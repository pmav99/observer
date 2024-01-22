# ruff: noqa: PLR2004: Magic value used in comparison
from __future__ import annotations

import fastparquet
import httpx
import multifutures
import pandas as pd
import pytest

from observer import tools
from observer.tools import to_parquet


def test_fetch_url():
    url = "https://google.com"
    response = tools._fetch_url(url, client=httpx.Client())
    assert "The document has moved" in response


def test_fetch_url_failure():
    url = "http://localhost"
    with pytest.raises(httpx.ConnectError) as exc:
        tools._fetch_url(url, client=httpx.Client(timeout=0))
    assert "in progress" in str(exc)


def test_fetch_url_full():
    url = "https://google.com"
    response = tools.fetch_url(url, client=httpx.Client(), rate_limit=multifutures.RateLimit())
    assert "The document has moved" in response


def test_to_parquet_single_partition_per_append(tmp_path) -> None:
    # Not the cleanest test ever, but OK...
    # We create one dataframe and we split it into groups of 3 rows each
    # We call to_parquet on each group and we assert that the resulting parquet file
    # has the expected structure (i.e. the correct number of row_groups)
    filename = str(tmp_path / "archive.parquet")
    partition_cols = ["year"]
    df = pd.DataFrame(
        data={
            "time": pd.to_datetime(
                [
                    "2021-01-01", "2021-02-02", "2021-03-03",
                    "2021-10-01", "2021-11-02", "2021-12-03",
                    "2022-01-01", "2022-02-02", "2022-03-03",
                    "2023-01-01", "2023-02-02", "2023-03-03",
                    "2023-10-01", "2023-11-02", "2023-12-03",
                    "2024-01-01", "2024-02-02", "2024-03-03",
                ],
            ),
            "rad": [
                100, 101, 102,
                200, 201, 202,
                300, 301, 302,
                400, 401, 402,
                500, 501, 502,
                600, 601, 602,
            ],
        },
    ).set_index("time")  # fmt: skip

    for i in range(len(df) // 3):
        end_index = (i + 1) * 3
        to_parquet(
            df=df.iloc[end_index - 3 : end_index],
            uri=filename,
            partition_cols=partition_cols,
        )

        pf = fastparquet.ParquetFile(filename)
        assert pf.file_scheme == "hive"
        pf_info = pf.info
        assert pf_info["rows"] == end_index
        assert pf_info["row_groups"] == df.index[:end_index].year.nunique()  # pyright: ignore [reportGeneralTypeIssues]
        assert pf_info["partitions"] == partition_cols
        assert pf_info["columns"] == ["time", "rad"]


def test_to_parquet_multiple_partitions_per_append(tmp_path) -> None:
    # Not the cleanest test ever, but OK...
    # We create one dataframe and we split it into groups of 3 rows each
    # We call to_parquet on each group and we assert that the resulting parquet file
    # has the expected structure (i.e. the correct number of row_groups)
    filename = str(tmp_path / "archive.parquet")
    partition_cols = ["year"]
    df = pd.DataFrame(
        data={
            "time": pd.to_datetime(
                [
                    "2021-01-01", "2021-02-02", "2022-01-01",
                    "2022-02-02", "2022-03-03", "2022-04-04",
                    "2022-05-05", "2023-01-01", "2023-02-02",
                ],
            ),
            "rad": [
                100, 101, 102,
                200, 201, 202,
                300, 301, 302,
            ],
        },
    ).set_index("time")  # fmt: skip

    # Create
    to_parquet(
        df=df.iloc[0:3],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 3
    assert pf_info["row_groups"] == 2
    assert pf_info["partitions"] == partition_cols
    assert pf_info["columns"] == ["time", "rad"]

    # Append, without new partitions
    to_parquet(
        df=df.iloc[3:6],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 6
    assert pf_info["row_groups"] == 2

    # Append, with new partitions
    to_parquet(
        df=df.iloc[6:],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 9
    assert pf_info["row_groups"] == 3


def test_to_parquet_multiple_partitions_per_append_and_multiple_partition_columns(tmp_path) -> None:
    # Not the cleanest test ever, but OK...
    # We create one dataframe and we split it into groups of 3 rows each
    # We call to_parquet on each group and we assert that the resulting parquet file
    # has the expected structure (i.e. the correct number of row_groups)
    filename = str(tmp_path / "archive.parquet")
    partition_cols = ["year", "month"]
    df = pd.DataFrame(
        data={
            "time": pd.to_datetime(
                [
                    "2021-01-01", "2021-01-02", "2021-02-01",
                    "2021-02-02", "2021-03-03", "2021-04-04",
                    "2021-05-05", "2023-01-01", "2023-01-02",
                ],
            ),
            "rad": [
                100, 101, 102,
                200, 201, 202,
                300, 301, 302,
            ],
        },
    ).set_index("time")  # fmt: skip

    # Create
    to_parquet(
        df=df.iloc[0:3],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 3
    assert pf_info["row_groups"] == 2
    assert pf_info["partitions"] == partition_cols
    assert pf_info["columns"] == ["time", "rad"]

    # Append, without new partitions
    to_parquet(
        df=df.iloc[3:6],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 6
    assert pf_info["row_groups"] == 4

    # Append, with new partitions
    to_parquet(
        df=df.iloc[6:],
        uri=filename,
        partition_cols=partition_cols,
    )
    pf = fastparquet.ParquetFile(filename)
    assert pf.file_scheme == "hive"
    pf_info = pf.info
    assert pf_info["rows"] == 9
    assert pf_info["row_groups"] == 6
