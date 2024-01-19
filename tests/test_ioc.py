from __future__ import annotations

import unittest.mock

import httpx
import limits
import multifutures
import pandas as pd
import pytest

import observer.ioc.scraper as scraper
from observer.ioc.scraper import scrape_ioc
from observer.ioc.scraper import scrape_ioc_station


def test_generate_urls():
    ioc_code = "AAA"
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-06-01")
    urls = scraper.generate_urls(
        ioc_code=ioc_code,
        start_date=start_date,
        end_date=end_date,
    )
    assert len(urls) == 6
    assert all(isinstance(url, str) for url in urls)
    assert all(ioc_code in url for url in urls)
    assert scraper.ioc_date(start_date) in urls[0]
    assert scraper.ioc_date(end_date) in urls[-1]


def test_generate_urls_raises_when_end_date_before_start_date():
    start_date = pd.Timestamp("2023-01-01")
    end_date = pd.Timestamp("2023-01-01")
    with pytest.raises(ValueError) as exc:
        scraper.generate_urls(
            ioc_code="aaaa",
            start_date=start_date,
            end_date=end_date,
        )
    assert str(exc.value) == f"'end_date' must be after 'start_date': {end_date} vs {start_date}"


def test_resolve_rate_limit_returns_object_as_is():
    rate_limit = multifutures.RateLimit()
    resolved = scraper._resolve_rate_limit(rate_limit=rate_limit)
    assert resolved is rate_limit


def test_resolve_http_client_returns_object_as_is():
    http_client = httpx.Client(timeout=httpx.Timeout(timeout=10, read=30))
    resolved = scraper._resolve_http_client(http_client=http_client)
    assert resolved is http_client


@unittest.mock.patch("observer.ioc.scraper.fetch_url")
def test_scrape_ioc_empty_responses(mocked_fetch_url):
    ioc_code = "blri"
    start_date = pd.Timestamp("2023-09-01")
    end_date = pd.Timestamp("2023-12-10")
    # The period between start_date and end_date should hit 4 URLs
    mocked_fetch_url.side_effect = [
        f"""[{{"error":"code '{ioc_code}' not found"}}]""",
        f"""[{{"error":"code '{ioc_code}' not found"}}]""",
        '[{"error":"Incorrect code"}]',
        "[]",
    ]
    data = scrape_ioc(
        ioc_codes=[ioc_code],
        start_date=start_date,
        end_date=end_date,
    )
    assert len(data) == 1
    assert ioc_code in data
    assert isinstance(data[ioc_code], pd.DataFrame)
    assert data[ioc_code].empty


@unittest.mock.patch("observer.ioc.scraper.fetch_url")
def test_scrape_ioc_normal_call(mocked_fetch_url):
    ioc_code = "acnj"
    start_date = pd.Timestamp("2022-03-12T11:04:00")
    end_date = pd.Timestamp("2022-03-12T11:06:00")
    mocked_fetch_url.side_effect = [
        """ [\
        {"slevel":0.905,"stime":"2022-03-12 11:04:00","sensor":"wls"},
        {"slevel":0.906,"stime":"2022-03-12 11:05:00","sensor":"wls"},
        {"slevel":0.896,"stime":"2022-03-12 11:06:00","sensor":"wls"}
        ]""",
    ]
    data = scrape_ioc(
        ioc_codes=[ioc_code],
        start_date=start_date,
        end_date=end_date,
    )
    assert len(data) == 1
    assert ioc_code in data
    assert isinstance(data[ioc_code], pd.DataFrame)
    assert not data[ioc_code].empty
    assert len(data[ioc_code]) == 3


@unittest.mock.patch("observer.ioc.scraper.fetch_url")
def test_scrape_ioc_duplicated_timestamps(mocked_fetch_url):
    ioc_code = "acnj"
    start_date = pd.Timestamp("2022-03-12T11:04:00")
    end_date = pd.Timestamp("2022-03-12T11:06:00")
    mocked_fetch_url.side_effect = [
        """ [\
        {"slevel":0.905,"stime":"2022-03-12 11:04:00","sensor":"wls"},
        {"slevel":0.906,"stime":"2022-03-12 11:05:00","sensor":"wls"},
        {"slevel":0.906,"stime":"2022-03-12 11:05:00","sensor":"wls"},
        {"slevel":0.906,"stime":"2022-03-12 11:05:00","sensor":"wls"},
        {"slevel":0.896,"stime":"2022-03-12 11:06:00","sensor":"wls"}
        ]""",
    ]
    data = scrape_ioc(
        ioc_codes=[ioc_code],
        start_date=start_date,
        end_date=end_date,
    )
    assert len(data) == 1
    assert ioc_code in data
    df = data[ioc_code]
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 3
    assert df.wls.max() == 0.906
    assert df.wls.min() == 0.896
    assert df.wls.median() == 0.905


@unittest.mock.patch("observer.ioc.scraper.fetch_url")
def test_scrape_ioc_station_normal_call(mocked_fetch_url):
    ioc_code = "acnj"
    start_date = pd.Timestamp("2022-03-12T11:04:00")
    end_date = pd.Timestamp("2022-03-12T11:06:00")
    mocked_fetch_url.side_effect = [
        """ [\
        {"slevel":0.905,"stime":"2022-03-12 11:04:00","sensor":"wls"},
        {"slevel":0.906,"stime":"2022-03-12 11:05:00","sensor":"wls"},
        {"slevel":0.896,"stime":"2022-03-12 11:06:00","sensor":"wls"}
        ]""",
    ]
    df = scrape_ioc_station(
        ioc_code=ioc_code,
        start_date=start_date,
        end_date=end_date,
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 3
    assert df.wls.max() == 0.906
    assert df.wls.min() == 0.896
    assert df.wls.median() == 0.905
