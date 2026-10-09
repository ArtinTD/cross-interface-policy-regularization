#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC=$HERE/verl_patch
DST=$HERE/toolrl
BASE=8cee13ec0ca72f0461da372a93a6fd8140dbb840
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

[ -d "$SRC/verl" ] || { echo "FATAL: no overlay at $SRC" >&2; exit 1; }
if [ ! -f "$DST/verl/trainer/main_ppo.py" ]; then
  echo "no ToolRL checkout at $DST. Clone it at the pinned commit:" >&2
  echo "  git clone https://github.com/qiancheng0/ToolRL.git $DST && git -C $DST checkout $BASE" >&2
  exit 1
fi
have=$(git -C "$DST" rev-parse HEAD 2>/dev/null || echo none)
if [ "$have" != "$BASE" ]; then
  echo "WARN: toolrl/ is at $have, the overlay was cut against $BASE." >&2
  echo "      These are whole-file replacements: re-cut them against the new base before applying." >&2
  [ "${ALLOW_BASE_MISMATCH:-0}" = 1 ] || exit 1
fi

n=0; drift=0
while IFS= read -r rel; do
  s=$SRC/$rel; d=$DST/$rel
  if [ -f "$d" ] && cmp -s "$s" "$d"; then continue; fi
  drift=$((drift + 1))
  if [ "$CHECK" = 1 ]; then echo "  differs: $rel"; continue; fi
  mkdir -p "$(dirname "$d")"; cp -f "$s" "$d"; n=$((n + 1))
done < <(cd "$SRC" && find verl -type f)

if [ "$CHECK" = 1 ]; then
  [ "$drift" = 0 ] && { echo "overlay applied and identical ($(cd "$SRC" && find verl -type f | wc -l | tr -d ' ') files)"; exit 0; }
  echo "overlay NOT applied: $drift file(s) differ. Run: bash scripts/apply_verl_patch.sh"; exit 1
fi
echo "overlay applied: $n file(s) written, $(( $(cd "$SRC" && find verl -type f | wc -l) - n )) already current"
