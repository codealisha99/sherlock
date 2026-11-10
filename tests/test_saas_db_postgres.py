"""SaasStore against a real Postgres (e.g. a throwaway Neon branch).

Skipped unless SHERLOCK_TEST_DATABASE_URL is set -- never point it at production.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

import pytest

from app.saas.db import SaasStore, utc_now

URL = os.getenv("SHERLOCK_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="SHERLOCK_TEST_DATABASE_URL not set")


@pytest.fixture(scope="module")
def store():
    s = SaasStore(database_url=URL)
    assert s.dialect == "postgres"
    return s


def _email() -> str:
    return f"pg-{uuid.uuid4().hex[:10]}@example.com"


def test_users_unique_email_and_update(store):
    email = _email()
    user = store.create_user(email, "hash")
    assert store.get_user_by_email(email)["id"] == user["id"]
    with pytest.raises(ValueError, match="email_taken"):
        store.create_user(email, "hash")
    store.update_user(user["id"], plan="pro", failed_logins=2)
    assert store.get_user(user["id"])["plan"] == "pro"


def test_sessions_expiry_and_cascade(store):
    user = store.create_user(_email(), "hash")
    store.create_session("tok-live-" + user["id"], user["id"], timedelta(hours=1))
    store.create_session("tok-dead-" + user["id"], user["id"], timedelta(seconds=-1))
    assert store.get_session_user("tok-live-" + user["id"])["id"] == user["id"]
    assert store.get_session_user("tok-dead-" + user["id"]) is None
    store.delete_user_sessions(user["id"])
    assert store.get_session_user("tok-live-" + user["id"]) is None


def test_api_keys_upsert_never_lists_secret(store):
    user = store.create_user(_email(), "hash")
    store.upsert_api_key(user["id"], "openai", "enc-1", "…1111", None, "m1")
    store.upsert_api_key(user["id"], "openai", "enc-2", "…2222", None, "m2")
    listed = store.list_api_keys(user["id"])
    assert len(listed) == 1 and listed[0]["key_hint"] == "…2222"
    assert "encrypted_key" not in listed[0]
    assert store.get_api_key(user["id"], "openai")["encrypted_key"] == "enc-2"
    assert store.delete_api_key(user["id"], "openai") is True
    assert store.delete_api_key(user["id"], "openai") is False


def test_repo_owner_idempotent_and_jobs(store):
    user = store.create_user(_email(), "hash")
    store.add_repo_owner("repo-pg", user["id"], "https://github.com/a/b")
    store.add_repo_owner("repo-pg", user["id"], "https://github.com/a/b")  # ON CONFLICT DO NOTHING
    assert store.list_user_repo_ids(user["id"]) == ["repo-pg"]
    job = store.create_job(user["id"], "https://github.com/a/b")
    store.update_job(job["id"], status="done", stage="done", progress=100,
                     repo_id="repo-pg", analysis_id="an-1", result={"stats": {"files": 3}})
    got = store.get_job(job["id"])
    assert got["result"] == {"stats": {"files": 3}} and got["status"] == "done"
    assert store.user_owns_analysis(user["id"], "an-1")
    assert store.count_active_jobs(user["id"]) == 0


def test_usage_and_payments_and_webhooks(store):
    user = store.create_user(_email(), "hash")
    store.record_usage(user["id"], "analysis")
    store.record_usage(user["id"], "analysis")
    assert store.count_usage_since(user["id"], "analysis", utc_now() - timedelta(minutes=5)) == 2
    sid = "cs_" + uuid.uuid4().hex
    store.create_payment(user["id"], sid, "pro", 1900, "usd")
    store.update_payment_status(sid, "paid")
    assert store.get_payment_by_session(sid)["status"] == "paid"
    evt = "evt_" + uuid.uuid4().hex
    assert store.mark_webhook_processed(evt) is True
    assert store.mark_webhook_processed(evt) is False
    store.unmark_webhook(evt)
    assert store.mark_webhook_processed(evt) is True


def test_schema_init_is_idempotent():
    SaasStore(database_url=URL)
    SaasStore(database_url=URL)
