from __future__ import annotations

import logging
import typing as T
from collections import abc

import fastparquet.writer
import httpx
import multifutures
import numpy as np
import numpy.typing as npt
import pandas as pd
import tenacity


if T.TYPE_CHECKING:
    from _typeshed import StrPath  # pyright: ignore [reportGeneralTypeIssues]


logger = logging.getLogger(__name__)


def _get_partition_key(df_or_ts: pd.DataFrame, partition_cols: T.Collection[str]) -> npt.NDArray[np.int_]:
    if hasattr(df_or_ts, "index"):
        partition_keys = np.vstack([getattr(df_or_ts.index, attr) for attr in partition_cols]).T
    else:
        partition_keys = np.vstack([getattr(df_or_ts, attr) for attr in partition_cols]).T
    return partition_keys


def _get_compression(compression_level: int) -> dict[str, T.Any]:
    return {
        "_default": {
            "type": "zstd",
            "args": {"level": compression_level},
        },
    }


def _merge_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    merged: pd.DataFrame = pd.concat([df1, df2])
    merged = merged[~merged.index.duplicated()].sort_index()
    return merged


def validate_paritition_columns(df: pd.DataFrame, partition_cols: abc.Collection[str]) -> None:
    for key in partition_cols:
        if not hasattr(df.index, key):
            raise ValueError(f"Partition key cannot be extracted from the Dataframe's index: {key}")


def _create_partition_columns(df: pd.DataFrame, partition_cols: abc.Collection[str]) -> pd.DataFrame:
    df = df.assign(**{key: getattr(df.index, key) for key in partition_cols})
    return df


def _to_parquet_create(
    df: pd.DataFrame,
    uri: StrPath,
    partition_cols: T.Collection[str],
    compression_level: int = 0,
    custom_metadata: dict[str, str] | None = None,
) -> None:
    fastparquet.write(
        filename=uri,
        data=df,
        compression=_get_compression(compression_level=compression_level),  # pyright: ignore [reportGeneralTypeIssues]
        file_scheme="hive",
        write_index=True,
        partition_on=partition_cols,
        append=False,
        stats=True,  # pyright: ignore [reportGeneralTypeIssues]
        custom_metadata=custom_metadata,
    )


def _to_parquet_append(
    *,
    pf: fastparquet.ParquetFile,
    df: pd.DataFrame,
    uri: StrPath,
    partition_cols: T.Collection[str],
    compression_level: int = 0,
) -> None:
    # Partition keys are tuples like these: `(2022,)` or `(2022, 03)`
    # Parquet row_group indexing follows the insertion order, not the index column
    # This means that in order to retrieve the row_group with the most recent timestamp
    # we need to check the statistics values.
    statistics = T.cast(dict[str, T.Any], pf.statistics)
    pf_index_of_most_recent_ts = np.argmax(statistics["max"]["time"])
    pf_most_recent_ts = pd.to_datetime(statistics["max"]["time"][pf_index_of_most_recent_ts])
    pf_most_recent_partition_key = tuple(_get_partition_key(pf_most_recent_ts, partition_cols)[0])
    df_first_partition_key = tuple(_get_partition_key(df.index[0], partition_cols)[0])

    if df_first_partition_key < pf_most_recent_partition_key:
        logger.debug("df_first_partition_key: %s", df_first_partition_key)
        logger.debug("pf_most_recent_partition_key: %s", pf_most_recent_partition_key)
        raise ValueError("Can't append data with earlier date")

    if df_first_partition_key == pf_most_recent_partition_key:
        # the dataframe we want to append has some data that use the same partition key
        # as the data that are stored in the parquet file. We need to:
        # 1. retrieve the existing data
        # 2. Merge them with the portion of the df that has the same partition key
        # 4. Write the parquet file with `append="overwrite"`
        # 5. Write the remaining df with `append=True`
        existing_df = pf[pf_index_of_most_recent_ts].to_pandas()
        same_partition_key = np.all(
            pf_most_recent_partition_key == _get_partition_key(df, partition_cols),
            axis=1,
        )
        df_merged = _merge_dataframes(
            df1=existing_df,
            df2=T.cast(pd.DataFrame, df[same_partition_key]),
        )
        fastparquet.write(
            filename=uri,
            data=df_merged,
            file_scheme="hive",
            compression=_get_compression(compression_level=compression_level),  # pyright: ignore [reportGeneralTypeIssues]
            partition_on=partition_cols,
            append="overwrite",   # pyright: ignore [reportGeneralTypeIssues]
        )  # fmt: skip
        # Filter out the rows of df that we already wrote to disk
        df = T.cast(pd.DataFrame, df[~same_partition_key])
    # just append the remaining rows
    if not df.empty:
        fastparquet.write(
            filename=uri,
            data=df,
            file_scheme="hive",
            compression=_get_compression(compression_level=compression_level),  # pyright: ignore [reportGeneralTypeIssues]
            partition_on=partition_cols,
            append=True,
        )  # fmt: skip


def to_parquet(
    *,
    df: pd.DataFrame,
    uri: StrPath,
    partition_cols: T.Collection[str],
    compression_level: int = 0,
    custom_metadata: dict[str, str] | None = None,
) -> None:
    # Sanity check
    if df.empty:
        raise ValueError("Empty dataframe!")

    # normalize dataframe
    df = df.sort_index()
    df = _create_partition_columns(df=df, partition_cols=partition_cols)

    try:
        pf = fastparquet.ParquetFile(str(uri))
    except FileNotFoundError:
        # The parquet file does not exist. Create it from scratch
        _to_parquet_create(
            df=df,
            uri=uri,
            partition_cols=partition_cols,
            compression_level=compression_level,
            custom_metadata=custom_metadata,
        )
    else:
        # The parquet file exists. Append data
        _to_parquet_append(
            pf=pf,
            df=df,
            uri=uri,
            partition_cols=partition_cols,
            compression_level=compression_level,
        )


def _before_sleep(retry_state: T.Any) -> None:  # pragma: no cover
    logger.warning(
        "Retrying %s: attempt %s ended with: %s",
        retry_state.fn,
        retry_state.attempt_number,
        retry_state.outcome,
    )


RETRY = tenacity.retry(
    stop=(tenacity.stop_after_delay(90) | tenacity.stop_after_attempt(10)),
    wait=tenacity.wait_random(min=2, max=10),
    retry=tenacity.retry_if_exception_type(httpx.TransportError),
    before_sleep=_before_sleep,
)


def _fetch_url(
    url: str,
    client: httpx.Client,
    ioc_code: str = "",
) -> str:
    try:
        response = client.get(url)
    except Exception:
        logger.warning("Failed to retrieve: %s", url)
        raise
    data = response.text
    return data


@RETRY
def fetch_url(
    url: str,
    client: httpx.Client,
    rate_limit: multifutures.RateLimit | None = None,
    ioc_code: str = "",
) -> str:
    if rate_limit is not None:
        while rate_limit.reached():
            multifutures.wait()  # pragma: no cover
    return _fetch_url(
        url=url,
        client=client,
        ioc_code=ioc_code,
    )
