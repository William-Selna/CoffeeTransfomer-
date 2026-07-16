#!/usr/bin/env bash
# Bundle the repo into a release tarball. By default includes the vendored HTE
# data but EXCLUDES the large corpora (data/raw), generated outputs (runs,
# data/processed), git, and caches. Pass --with-raw to include data/raw too
# (big — only if you want the fetched PubChem/USPTO in the bundle).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NAME="coffee-transformer-$(basename "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)")"
OUT="${NAME}.tar.gz"
WITH_RAW="${1:-}"

EXCLUDES=(
  --exclude='.git'
  --exclude='runs'
  --exclude='data/processed'
  --exclude='data/cache'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='.venv'
  --exclude='*.egg-info'
  --exclude='.pytest_cache'
  --exclude='*.tar.gz'
)
if [ "$WITH_RAW" != "--with-raw" ]; then
  EXCLUDES+=(--exclude='data/raw')
fi

# build into a temp file OUTSIDE the repo so tar never reads its own output
TMP="$(mktemp -d)"
tar czf "$TMP/$OUT" "${EXCLUDES[@]}" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
mv "$TMP/$OUT" "$ROOT/$OUT"
rmdir "$TMP" 2>/dev/null || true
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
[ "$WITH_RAW" = "--with-raw" ] || echo "(large corpora in data/raw excluded; pass --with-raw to include)"
