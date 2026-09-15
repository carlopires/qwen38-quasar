#!/usr/bin/env bash
#
# Pin every source to an immutable revision and (optionally) download it.
#
# Handoff section 12. Resolves HF revisions and the NInfer git SHA, writes
# provenance.lock.json, and reports any drift from the revisions recorded when
# the execution plan was written.
#
# Usage:
#   scripts/bootstrap_sources.sh              # resolve + write lock, no download
#   scripts/bootstrap_sources.sh --download   # also fetch the sources
#   scripts/bootstrap_sources.sh --check      # fail if anything drifted
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOWNLOAD=0
CHECK=0
for arg in "$@"; do
  case "$arg" in
    --download) DOWNLOAD=1 ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

EFFICIENTTHINK_REPO="nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2"
QUASAR_REFERENCE_REPO="QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4"
OPEN_PERFECTBLEND_REPO="mlabonne/open-perfectblend"
NINFER_LOCAL="${NINFER_LOCAL:-$HOME/code/ninfer}"

echo "== environment =="
command -v uv >/dev/null || { echo "uv is required" >&2; exit 1; }
if command -v hf >/dev/null; then
  echo "hf CLI: $(command -v hf)"
  hf auth whoami >/dev/null 2>&1 && echo "hf auth: ok" || echo "hf auth: not logged in (public sources still work)"
else
  echo "hf CLI: not found (downloads will use huggingface_hub via uv)"
fi

if [[ ! -d "$NINFER_LOCAL/.git" ]]; then
  echo "NInfer checkout not found at $NINFER_LOCAL" >&2
  echo "Set NINFER_LOCAL, or clone carlopires/ninfer-rtx5090-mobile there first." >&2
  exit 1
fi

echo
echo "== resolving revisions =="
uv run python - "$@" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from qwen38_quasar.provenance import (
    EFFICIENTTHINK_REPO,
    OPEN_PERFECTBLEND_REPO,
    QUASAR_REFERENCE_REPO,
    build_lock,
    git_sha,
    read_lock,
    resolve_hf_revision,
    write_lock,
)

root = Path.cwd()
lock_path = root / "provenance.lock.json"
ninfer_local = Path(os.environ.get("NINFER_LOCAL", Path.home() / "code" / "ninfer"))

resolved = {
    "efficientthink": resolve_hf_revision(EFFICIENTTHINK_REPO),
    "quasar_reference": resolve_hf_revision(QUASAR_REFERENCE_REPO),
    "open_perfectblend": resolve_hf_revision(OPEN_PERFECTBLEND_REPO, kind="dataset"),
    "ninfer": git_sha(ninfer_local),
}

for name, revision in resolved.items():
    print(f"  {name:20} {revision}")

lock = build_lock(
    efficientthink=resolved["efficientthink"],
    quasar_reference=resolved["quasar_reference"],
    open_perfectblend=resolved["open_perfectblend"],
    ninfer=resolved["ninfer"],
    generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
)

drift = []
if lock_path.exists():
    previous = read_lock(lock_path)
    for name, pin in lock.sources.items():
        if name in previous.sources and previous.sources[name].revision != pin.revision:
            drift.append(
                f"  {name}: {previous.sources[name].revision} -> {pin.revision}"
            )

if drift:
    print("\n!! revision drift detected since the last lock:")
    print("\n".join(drift))
    mode = os.environ.get("BOOTSTRAP_MODE", "")
    if mode == "check":
        print("\nRefusing to continue in --check mode.", file=sys.stderr)
        raise SystemExit(1)

write_lock(lock, lock_path)
print(f"\nwrote {lock_path}")
PY

if [[ "$CHECK" == "1" ]]; then
  BOOTSTRAP_MODE=check uv run python - <<'PY'
from pathlib import Path
lock = Path("provenance.lock.json")
print("lock present:", lock.exists())
import json
data = json.loads(lock.read_text())
for name, pin in sorted(data["sources"].items()):
    print(f"  {name:20} {pin['revision']}")
PY
fi

if [[ "$DOWNLOAD" == "1" ]]; then
  echo
  echo "== downloading sources =="
  mkdir -p models datasets
  echo "-- EfficientThink BF16 (~55 GB, 12 shards)"
  hf download "$EFFICIENTTHINK_REPO" --include 'BF16/**' --local-dir models/efficientthink-source
  echo "-- QUASAR schema oracle (config and tokenizer only; weights not needed)"
  hf download "$QUASAR_REFERENCE_REPO" --include 'config.json' --include 'model.safetensors.index.json' \
    --local-dir models/quasar-qwen38-reference
  echo "-- Open-PerfectBlend"
  hf download "$OPEN_PERFECTBLEND_REPO" --repo-type dataset --local-dir datasets/open-perfectblend
  echo
  echo "downloads complete; verify hashes against provenance.lock.json before training"
else
  echo
  echo "Skipping downloads (pass --download to fetch sources)."
fi

echo
echo "== NInfer baseline =="
git -C "$NINFER_LOCAL" rev-parse HEAD
git -C "$NINFER_LOCAL" status --short | head -5 || true
echo "Recorded in provenance.lock.json. Work on branch feat/efficientthink-quasar, not master."
