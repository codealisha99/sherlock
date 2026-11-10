"""Test defaults: legacy tests run in single-user local mode with an isolated SaaS DB.

SaaS tests opt into auth with monkeypatch.setenv("SHERLOCK_AUTH_REQUIRED", "true").
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="sherlock_test_")
# Never touch the developer's real index DB from tests.
os.environ["SHERLOCK_INDEX_DB"] = os.path.join(_TMP, "sherlock.db")
os.environ.setdefault("SHERLOCK_AUTH_REQUIRED", "false")
os.environ.setdefault("SHERLOCK_SAAS_DB", os.path.join(_TMP, "saas.db"))
os.environ.setdefault("SHERLOCK_SECRET_KEY", "test-secret-key-not-for-production-0123456789")
