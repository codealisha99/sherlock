"""BYOK HTTP handlers: list, save, and delete the caller's provider keys."""

from __future__ import annotations

from typing import Optional

from app.saas.byok import delete_user_key, list_user_keys, provider_catalog, save_user_key
from app.saas.http import HTTPError, Request, Response


def _opt_str(body: dict, field: str) -> Optional[str]:
    value = body.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPError(400, f"'{field}' must be a string.")
    return value


def list_keys(req: Request) -> Response:
    user = req.require_user()
    return Response(200, {"keys": list_user_keys(user["id"]), "providers": provider_catalog()})


def save_key(req: Request) -> Response:
    user = req.require_user()
    body = req.json()
    provider = _opt_str(body, "provider")
    api_key = _opt_str(body, "api_key")
    if not provider or not api_key:
        raise HTTPError(400, "'provider' and 'api_key' are required.")
    try:
        key = save_user_key(
            user["id"],
            provider,
            api_key,
            model=_opt_str(body, "model"),
            base_url=_opt_str(body, "base_url"),
        )
    except ValueError as exc:  # validation messages never contain the key
        raise HTTPError(400, str(exc)) from None
    return Response(201, {"key": key})


def delete_key(req: Request) -> Response:
    user = req.require_user()
    provider = req.path_params.get("provider", "")
    if not delete_user_key(user["id"], provider):
        raise HTTPError(404, "No key stored for that provider.")
    return Response(200, {"status": "deleted"})
