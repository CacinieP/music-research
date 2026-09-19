#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/prepare_site.py
python -m mkdocs build --strict
python scripts/check_site.py
python scripts/check_content.py
node scripts/check_math.cjs
python scripts/prepare_wiki.py
