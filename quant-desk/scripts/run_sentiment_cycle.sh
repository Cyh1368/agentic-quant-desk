#!/usr/bin/env bash
# One sentiment research cycle. Invoked by PM2 on the 4h cadence.
# --allow-unreviewed-terms: remove this flag once terms_review in
# config/sentiment.yaml is completed (the gate then passes on its own).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m quantdesk.sentiment_cycle --allow-unreviewed-terms
