"""HTTP handlers for AI triage and platform capabilities."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.saas.db import get_store
from app.saas.http import HTTPError, Request, Response
from app.saas.ingest_routes import _owned_repo_id
from app.saas.llm_platform import platform_model_name, resolve_gateway
from app.saas.triage import MAX_EXPLICIT, run_triage

NO_MODEL_MESSAGE = "No AI model available. Add your own API key in Settings."


def capabilities(req: Request) -> Response:
    """Public. With a session, also reports whether the caller has a BYOK key."""
    model = platform_model_name()
    body: Dict[str, Any] = {"platform_model": model is not None, "platform_model_name": model}
    if req.user is not None:
        body["byok"] = get_store().get_api_key(req.user["id"]) is not None
    return Response(body=body)


def _parse_ids(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(i, str) and i for i in raw):
        raise HTTPError(400, "'finding_ids' must be a list of finding ids")
    unique = list(dict.fromkeys(raw))
    if len(unique) > MAX_EXPLICIT:
        raise HTTPError(400, f"At most {MAX_EXPLICIT} findings can be triaged per request")
    return unique


def triage(req: Request) -> Response:
    user = req.require_user()
    repo_id = _owned_repo_id(req)
    finding_ids = _parse_ids(req.json().get("finding_ids"))
    gateway, _source = resolve_gateway(user)
    if gateway is None:
        return Response(status=409, body={"status": "error", "error": NO_MODEL_MESSAGE, "code": "no_model"})
    return Response(body=run_triage(user, repo_id, gateway, finding_ids))
