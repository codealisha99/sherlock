#!/bin/sh
# Railway mounts volumes owned by root: fix ownership, then drop to the app user.
set -eu
mkdir -p "$SHERLOCK_DATA_DIR"
chown -R app:app "$SHERLOCK_DATA_DIR"
# setpriv keeps the environment: reset HOME, or libpq probes /root/.postgresql
# (EACCES there is fatal, unlike a missing file) and every Postgres connect fails.
exec setpriv --reuid=app --regid=app --init-groups --inh-caps=-all env HOME=/home/app USER=app "$@"
