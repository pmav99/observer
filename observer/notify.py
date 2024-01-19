from __future__ import annotations

import functools
import logging

import httpx

from observer.azclients import get_keyvault_client

logger = logging.getLogger(__name__)


@functools.lru_cache
def _get_notify_info_url() -> str:
    client = get_keyvault_client()
    url = client.get_secret("seareport-info-webhook-url").value
    assert url is not None
    return url


@functools.lru_cache
def _get_notify_error_url() -> str:
    client = get_keyvault_client()
    url = client.get_secret("seareport-error-webhook-url").value
    assert url is not None
    return url


def notify_info(*, message: str, title: str = "") -> None:
    url = _get_notify_info_url()
    if not title:
        title = "INFO"
    data = {"text": f"**{title}**\n\n {message}"}
    try:
        response = httpx.post(url, json=data, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("Failed to send INFO notification:\n%s", data)


def notify_error(*, message: str, title: str = "") -> None:
    url = _get_notify_error_url()
    if not title:
        title = "ERROR"
    data = {"text": f"**{title}**\n\n {message}"}
    try:
        response = httpx.post(url, json=data, timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        logger.exception("Failed to send ERROR notification:\n%s", data)
