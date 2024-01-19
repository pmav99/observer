from __future__ import annotations

import logging
import typing as T

import adlfs
import azure.identity.aio
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import ContainerClient

from .settings import get_settings

logger = logging.getLogger(__name__)

Credential: T.TypeAlias = azure.identity.ChainedTokenCredential
CredentialAIO: T.TypeAlias = azure.identity.aio.ChainedTokenCredential


def get_credential() -> Credential:
    credential_chain = (
        azure.identity.AzureCliCredential(),
        azure.identity.ManagedIdentityCredential(),
    )
    credential = azure.identity.ChainedTokenCredential(*credential_chain)
    return credential


def get_credential_aio() -> CredentialAIO:
    credential_chain = (
        azure.identity.aio.AzureCliCredential(),
        azure.identity.aio.ManagedIdentityCredential(),
    )
    credential = azure.identity.aio.ChainedTokenCredential(*credential_chain)
    return credential


def get_keyvault_client(
    keyvault_url: str | None = None,
    credential: Credential | None = None,
) -> SecretClient:
    if not keyvault_url:
        keyvault_url = get_settings().keyvault_url
    if credential is None:
        credential = get_credential()
    client = SecretClient(vault_url=keyvault_url, credential=credential)
    return client


def get_blob_client(
    storage_account: str = "",
    credential: Credential | None = None,
) -> BlobServiceClient:
    if not storage_account:
        storage_account = get_settings().storage_account
    account_url = f"https://{storage_account}.blob.core.windows.net"
    if credential is None:
        credential = get_credential()
    client = BlobServiceClient(account_url=account_url, credential=credential)
    return client


def get_obs_client(
    storage_account: str = "",
    container_name: str = "",
    credential: Credential | None = None,
) -> ContainerClient:
    settings = get_settings()
    if not storage_account:
        storage_account = settings.storage_account
    if not container_name:
        container_name = settings.container_name
    blob_client = get_blob_client(storage_account=storage_account, credential=credential)
    container_client = blob_client.get_container_client(container=container_name)
    return container_client


def get_storage_options(
    credential: CredentialAIO | None = None,
) -> dict[str, T.Any]:
    settings = get_settings()
    if credential is None:
        credential = get_credential_aio()
    return dict(account_name=settings.storage_account, credential=credential)


def get_obs_fs(
    credential: CredentialAIO | None = None,
) -> adlfs.AzureBlobFileSystem:
    storage_options = get_storage_options(credential=credential)
    fs = adlfs.AzureBlobFileSystem(**storage_options)
    return fs
