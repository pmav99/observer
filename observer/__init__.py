from __future__ import annotations

import importlib.metadata

import fastparquet  # noqa: F401=unused import

from .azclients import Credential
from .azclients import CredentialAIO
from .azclients import get_credential
from .azclients import get_credential_aio
from .azclients import get_obs_client
from .azclients import get_obs_fs
from .azclients import get_storage_options
from .ioc import get_ioc_df
from .ioc import get_ioc_metadata
from .ioc import get_ioc_parquet_file
from .ioc import list_ioc_stations
from .ioc import scrape_ioc
from .ioc import scrape_ioc_station
from .ioc import write_ioc_df
from .notify import notify_error
from .notify import notify_info
from .settings import get_settings
from .settings import Settings
from .tools import to_parquet

__version__ = importlib.metadata.version(__name__)


__all__: list[str] = [
    "__version__",
    # azclients
    "Credential",
    "CredentialAIO",
    "get_credential",
    "get_credential_aio",
    "get_obs_client",
    "get_obs_fs",
    "get_storage_options",
    # ioc
    "get_ioc_df",
    "get_ioc_metadata",
    "get_ioc_parquet_file",
    "list_ioc_stations",
    "scrape_ioc",
    "scrape_ioc_station",
    "write_ioc_df",
    # notify
    "notify_error",
    "notify_info",
    # settings
    "get_settings",
    "Settings",
    # tools
    "to_parquet",
]
