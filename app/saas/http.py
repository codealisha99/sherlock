"""Framework-free request/response types shared by all SaaS route handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


class HTTPError(Exception):
    """Raise from a handler to return a JSON error with the given status."""

    def __init__(self, status: int, message: str, headers: Tuple[Tuple[str, str], ...] = ()):
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    query: Mapping[str, List[str]]
    headers: Mapping[str, str]
    body: bytes
    client_ip: str
    user: Optional[Dict[str, Any]] = None
    path_params: Mapping[str, str] = field(default_factory=dict)

    def json(self) -> Dict[str, Any]:
        if not self.body:
            return {}
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HTTPError(400, "Invalid JSON payload")
        if not isinstance(data, dict):
            raise HTTPError(400, "JSON body must be an object")
        return data

    def q(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self.query.get(name, [default])[0]

    def require_user(self) -> Dict[str, Any]:
        if not self.user:
            raise HTTPError(401, "Authentication required")
        return self.user


@dataclass(frozen=True)
class Response:
    status: int = 200
    body: Union[Dict[str, Any], List[Any], bytes, str] = field(default_factory=dict)
    headers: Tuple[Tuple[str, str], ...] = ()
    content_type: str = "application/json; charset=utf-8"

    def encoded(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        if isinstance(self.body, str):
            return self.body.encode("utf-8")
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")
