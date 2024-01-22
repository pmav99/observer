from __future__ import annotations

import logging
import typing as T

import fastparquet
import geopandas as gpd
import pandas as pd

from observer.azclients import CredentialAIO
from observer.azclients import get_obs_fs
from observer.azclients import get_storage_options
from observer.settings import get_settings


logger = logging.getLogger(__name__)


def _get_compression(compression_level: int) -> dict[str, T.Any]:
    return {
        "_default": {
            "type": "zstd",
            "args": {"level": compression_level},
        },
    }


def _get_metadata_uri() -> str:
    settings = get_settings()
    uri = f"az://{settings.container_name}/ioc/metadata.parquet"
    return uri


def _get_station_uri(ioc_code: str) -> str:
    settings = get_settings()
    uri = f"az://{settings.container_name}/ioc/stations/{ioc_code}.parquet"
    return uri


def get_ioc_metadata(*, credential: CredentialAIO | None = None) -> pd.DataFrame:
    """
    Return the IOC metadata from Blob
    """
    settings = get_settings()
    storage_options = get_storage_options(credential=credential)
    df = pd.read_parquet(
        _get_metadata_uri(),
        storage_options=storage_options,
        engine=settings.engine,
    )
    if "geometry" not in df.columns:
        df = df.assign(geometry=gpd.points_from_xy(df.lon, df.lat))
    return df


def write_ioc_metadata(*, df: pd.DataFrame, credential: CredentialAIO | None = None) -> None:
    logger.info("IOC Meta: Starting upload")
    # GeoPandas (i.e. the `geometry` column) only supports using pyarrow as the engine for to_parquet.
    # We are using fastparquet, so we need to drop it
    if "geometry" in df:
        df = df.drop(columns="geometry")
    settings = get_settings()
    uri = _get_metadata_uri()
    storage_options = get_storage_options(credential=credential)
    df.to_parquet(  # type: ignore[call-overload]
        uri,
        index=True,
        storage_options=storage_options,
        engine=settings.engine,
        compression=_get_compression(compression_level=0),  # type: ignore
        append=False,
        custom_metadata={"saved_at": pd.Timestamp.now("utc").isoformat()},
    )
    logger.info("IOC Meta: Finished upload")


def write_ioc_df(
    *,
    df: pd.DataFrame,
    ioc_code: str,
    compression_level: int = 0,
    credential: CredentialAIO | None = None,
    append: bool = False,
    custom_metadata: dict[str, str] | None = None,
) -> None:
    logger.debug("%s: Starting upload", ioc_code)
    settings = get_settings()
    uri = _get_station_uri(ioc_code)
    storage_options = get_storage_options(credential=credential)
    if "year" not in df.columns:
        df = df.assign(year=df.index.year)  # type: ignore[attr-defined]
    df.to_parquet(  # type: ignore[call-overload]
        uri,
        partition_cols=["year"],
        index=True,
        storage_options=storage_options,
        engine=settings.engine,
        compression=_get_compression(compression_level=compression_level),  # type: ignore
        append=append,
        custom_metadata=custom_metadata,
    )
    logger.info("%s: Finished upload", ioc_code)


def get_ioc_parquet_file(
    ioc_code: str,
    *,
    credential: CredentialAIO | None = None,
    **kwargs: dict[str, T.Any],
) -> fastparquet.ParquetFile:
    uri = _get_station_uri(ioc_code)
    fs = get_obs_fs(credential=credential)
    pf = fastparquet.ParquetFile(uri, fs=fs, **kwargs)  # type: ignore
    return pf


def get_ioc_df(
    ioc_code: str,
    *,
    no_years: int = 2,
    credential: CredentialAIO | None = None,
    **kwargs: dict[str, T.Any],
) -> pd.DataFrame:
    pf = get_ioc_parquet_file(ioc_code=ioc_code, credential=credential, **kwargs)
    df: pd.DataFrame = pf[-no_years:].to_pandas(columns=pf.columns)
    return df


def list_ioc_stations(
    *,
    credential: CredentialAIO | None = None,
) -> list[str]:
    fs = get_obs_fs(credential=credential)
    existing = [
        parquet.split("/")[-1].split(".")[0] for parquet in fs.ls(f"{get_settings().container_name}/ioc/stations")
    ]
    return existing
