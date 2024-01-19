from __future__ import annotations

import collections
import io
import itertools
import logging
import typing as T

import httpx
import limits
import multifutures
import pandas as pd
import searvey.ioc
import tenacity

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ioc-sealevelmonitoring.org/service.php?query=data&timestart={timestart}&timestop={timestop}&code={ioc_code}"

IOC_URL_TS_FORMAT = "%Y-%m-%dT%H:%M:%S"
IOC_JSON_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def ioc_date(ts: pd.Timestamp) -> str:
    formatted = ts.strftime(IOC_URL_TS_FORMAT)
    return formatted


def generate_urls(
    ioc_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[str]:
    if end_date <= start_date:
        raise ValueError(f"'end_date' must be after 'start_date': {end_date} vs {start_date}")
    duration = end_date - start_date
    periods = duration.days // 30 + 2
    urls = []
    date_range = pd.date_range(start_date, end_date, periods=periods, unit="us", inclusive="both")
    for start, stop in itertools.pairwise(date_range):
        timestart = ioc_date(start)
        timestop = ioc_date(stop)
        url = BASE_URL.format(ioc_code=ioc_code, timestart=timestart, timestop=timestop)
        urls.append(url)
    return urls


def my_before_sleep(retry_state: T.Any) -> None:
    logger.warning(
        "Retrying %s: attempt %s ended with: %s",
        retry_state.fn,
        retry_state.attempt_number,
        retry_state.outcome,
    )


@tenacity.retry(
    stop=(tenacity.stop_after_delay(90) | tenacity.stop_after_attempt(10)),
    wait=tenacity.wait_random(min=2, max=10),
    retry=tenacity.retry_if_exception_type(httpx.TransportError),
    before_sleep=my_before_sleep,
)
def fetch_url(
    url: str,
    client: httpx.Client,
    rate_limit: multifutures.RateLimit,
    ioc_code: str = "",
) -> str:
    while rate_limit.reached(identifier="IOC"):
        multifutures.wait()

    try:
        response = client.get(url)
    except Exception:
        logger.warning("Failed to retrieve: %s", url)
        raise
    data = response.text
    return data


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.sensor.isin(searvey.ioc.IOC_STATION_DATA_COLUMNS.values())]
    df = df.assign(stime=pd.DatetimeIndex(pd.to_datetime(df.stime.str.strip(), format=IOC_JSON_TS_FORMAT)))
    df = df.rename(columns={"stime": "time"})
    # Occasionaly IOC contains complete garbage. E.g. duplicate timestamps on the same sensor. We should drop those.
    # https://www.ioc-sealevelmonitoring.org/service.php?query=data&timestart=2022-03-12T11:03:40&timestop=2022-04-11T09:04:26&code=acnj
    duplicated_timestamps = df[["time", "sensor"]].duplicated()
    if duplicated_timestamps.sum() > 0:
        df = df[~duplicated_timestamps]
        logger.warning("%s: Dropped duplicates: %d rows", df.attrs["ioc_code"], duplicated_timestamps.sum())
    df = df.pivot(index="time", columns="sensor", values="slevel")
    df._mgr.items.name = ""  # type: ignore[attr-defined]
    return df


def parse_json(content: str, ioc_code: str) -> pd.DataFrame:
    df = pd.read_json(content, orient="records")
    df.attrs["ioc_code"] = ioc_code
    df = normalize_df(df)
    return df


def _resolve_rate_limit(rate_limit: multifutures.RateLimit | None = None) -> multifutures.RateLimit:
    if rate_limit is None:
        rate_limit = multifutures.RateLimit(rate_limit=limits.parse("5/second"))
    return rate_limit


def _resolve_http_client(http_client: httpx.Client | None = None) -> httpx.Client:
    if http_client is None:
        timeout = httpx.Timeout(timeout=10, read=30)
        http_client = httpx.Client(timeout=timeout)
    return http_client


def retrieve_ioc_data(
    ioc_codes: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rate_limit: multifutures.RateLimit,
    http_client: httpx.Client,
    n_threads: int,
) -> list[multifutures.FutureResult]:
    kwargs = []
    for ioc_code in ioc_codes:
        for url in generate_urls(ioc_code=ioc_code, start_date=start_date, end_date=end_date):
            kwargs.append(
                dict(
                    ioc_code=ioc_code,
                    url=url,
                    client=http_client,
                    rate_limit=rate_limit,
                ),
            )
    with http_client:
        logger.debug("Starting data retrieval")
        results = multifutures.multithread(func=fetch_url, func_kwargs=kwargs, check=False, n_workers=n_threads)
        logger.debug("Finished data retrieval")
    multifutures.check_results(results)
    return results


def parse_ioc_responses(
    ioc_responses: list[multifutures.FutureResult],
    n_processes: int,
) -> list[multifutures.FutureResult]:
    # Parse the json files using pandas
    # This is a CPU heavy process, so let's use multiprocess
    # Not all the urls contain data, so let's filter them out
    kwargs = []
    for result in ioc_responses:
        ioc_code = result.kwargs["ioc_code"]  # type: ignore[index]
        # if a url doesn't have any data instead of a 404, it returns an empty list `[]`
        if result.result == "[]":
            continue
        # For some stations though we get a json like this:
        #    '[{"error":"code \'blri\' not found"}]'
        #    '[{"error":"code \'bmda2\' not found"}]'
        # we should ignore these, too
        elif result.result == f"""[{{"error":"code '{ioc_code}' not found"}}]""":
            continue
        # And if the IOC code does not match some pattern (5 letters?) then we get this error
        elif result.result == '[{"error":"Incorrect code"}]':
            continue
        else:
            kwargs.append(dict(ioc_code=ioc_code, content=io.StringIO(result.result)))
    logger.debug("Starting JSON parsing")
    results = multifutures.multiprocess(parse_json, func_kwargs=kwargs, check=False, n_workers=n_processes)
    multifutures.check_results(results)
    logger.debug("Finished JSON parsing")
    return results


def group_results(
    ioc_codes: list[str],
    parsed_responses: list[multifutures.FutureResult],
) -> dict[str, pd.DataFrame]:
    # Group per IOC code
    df_groups = collections.defaultdict(list)
    for item in parsed_responses:
        df_groups[item.kwargs["ioc_code"]].append(item.result)  # type: ignore[index]

    # Concatenate dataframes and remove duplicates
    dataframes: dict[str, pd.DataFrame] = {}
    for ioc_code in ioc_codes:
        if ioc_code in df_groups:
            df_group = df_groups[ioc_code]
            df = pd.concat(df_group)
            df = df.sort_index()
            logger.debug("%s: Total timestamps : %d", ioc_code, len(df))
            df = df[~df.index.duplicated()]
            logger.debug("%s: Unique timestamps: %d", ioc_code, len(df))
        else:
            logger.warning("%s: No data. Creating a dummy dataframe", ioc_code)
            df = pd.DataFrame(columns=["time"], dtype="datetime64[ns]").set_index("time")
        dataframes[ioc_code] = df
        logger.debug("%s: Finished conversion to pandas", ioc_code)

    return dataframes


def scrape_ioc(
    *,
    ioc_codes: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rate_limit: multifutures.RateLimit | None = None,
    http_client: httpx.Client | None = None,
    n_threads: int = 5,
    n_processes: int = multifutures.MAX_AVAILABLE_PROCESSES,
) -> dict[str, pd.DataFrame]:
    logger.info("Starting scraping: %s - %s", start_date, end_date)
    rate_limit = _resolve_rate_limit(rate_limit=rate_limit)
    http_client = _resolve_http_client(http_client=http_client)

    # Fetch json files from the IOC website
    # We use multithreading in order to be able to use RateLimit + to take advantage of higher performance
    ioc_responses: list[multifutures.FutureResult] = retrieve_ioc_data(
        ioc_codes=ioc_codes,
        start_date=start_date,
        end_date=end_date,
        rate_limit=rate_limit,
        http_client=http_client,
        n_threads=n_threads,
    )

    # Parse the json files using pandas
    # This is a CPU heavy process, so we are using multiprocessing here
    parsed_responses: list[multifutures.FutureResult] = parse_ioc_responses(
        ioc_responses=ioc_responses,
        n_processes=n_processes,
    )

    # OK, now we have a list of dataframes. We need to group them per ioc_code, concatenate them and remove duplicates
    dataframes = group_results(ioc_codes=ioc_codes, parsed_responses=parsed_responses)
    return dataframes


def scrape_ioc_station(
    *,
    ioc_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    rate_limit: multifutures.RateLimit | None = None,
    http_client: httpx.Client | None = None,
    n_threads: int = 5,
    n_processes: int = multifutures.MAX_AVAILABLE_PROCESSES,
) -> pd.DataFrame:
    logger.info("%s: Starting scraping: %s - %s", ioc_code, start_date, end_date)
    df = scrape_ioc(
        ioc_codes=[ioc_code],
        start_date=start_date,
        end_date=end_date,
        rate_limit=rate_limit,
        http_client=http_client,
        n_threads=n_threads,
        n_processes=n_processes,
    )[ioc_code]
    logger.info("%s: Finished scraping: %s - %s", ioc_code, start_date, end_date)
    return df
