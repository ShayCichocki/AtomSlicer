#!/bin/bash
# Time ref vs new AtomSlicer on fixtures, compare waypoints, append to METAL_LOG.md.
# Usage: test/run_bench.sh "change description" fixture.stl [fixture2.stl ...]
# Env:   REF=path/to/ref/slicer   (default build-ref/slicer)
#        NEW=path/to/new/slicer   (default build/slicer)
#        ARGS="-d 0.4 -h 0.5 -o 1 -i 0"   slicer options (-W is added)
#        REUSE_REF=1   reuse a cached ref run for the same fixture+ARGS
#        NOLOG=1       print the table only, do not append to the log
#        PY=python3    interpreter with numpy+scipy;  LOG=file  log to append to (default METAL_LOG.md)
#        FLOOR=path    CPU build with -ffp-contract=off (default build-perturb/slicer): its
#                      output vs ref is the float noise floor (cached per fixture)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-python3}"              # needs numpy + scipy
LOG="${LOG:-$ROOT/METAL_LOG.md}"   # benchmark log to append to
REF="${REF:-$ROOT/build-ref/slicer}"
NEW="${NEW:-$ROOT/build/slicer}"
ARGS="${ARGS:--d 0.4 -h 0.5 -o 1 -i 0}"
FLOOR="${FLOOR:-$ROOT/build-perturb/slicer}"
CHANGE="$1"; shift
OUT="$HERE/out"; mkdir -p "$OUT"
TAG=$(echo "$ARGS" | tr -c 'A-Za-z0-9' '_')
ROWS=""
ALLPASS=1
for STL in "$@"; do
  FX=$(basename "$STL" .stl)
  R="$OUT/ref_${FX}_${TAG}"; N="$OUT/new_${FX}_${TAG}"
  if [ "${REUSE_REF:-0}" != 1 ] || [ ! -s "$R.fxwp" ]; then
    /usr/bin/time -l "$REF" "$STL" $ARGS -W "$R.fxwp" > "$R.log" 2> "$R.time" || { echo "ref failed on $FX"; exit 2; }
  fi
  FARG=""
  if [ -x "$FLOOR" ]; then
    F="$OUT/floor_${FX}_${TAG}"
    [ -s "$F.fxwp" ] || "$FLOOR" "$STL" $ARGS -W "$F.fxwp" > "$F.log" 2>&1
    FARG="--floor $F.fxwp"
  fi
  /usr/bin/time -l "$NEW" "$STL" $ARGS -W "$N.fxwp" > "$N.log" 2> "$N.time" || { echo "new failed on $FX"; tail -5 "$N.log"; exit 2; }
  "$PY" "$HERE/compare.py" "$R.fxwp" "$N.fxwp" --logs "$R.log" "$N.log" $FARG --json > "$N.cmp.json"
  [ $? -eq 0 ] || ALLPASS=0
  "$PY" "$HERE/compare.py" "$R.fxwp" "$N.fxwp" --logs "$R.log" "$N.log" $FARG | tail -2
  ROWS="$ROWS$("$PY" "$HERE/bench_table.py" "$FX" "$R.log" "$N.log" "$N.cmp.json" "$R.time" "$N.time")"$'\n'
done
LOAD=$(uptime | sed 's/.*load averages*: //')
BLOCK="
### $(date '+%Y-%m-%d %H:%M') — $CHANGE

Args: \`$ARGS\`. New binary: \`${NEW#$ROOT/}\`. Load average at end: $LOAD (other slicing jobs share the CPU, so CPU timings are noisy).

| fixture | stage | ref s | new s | speedup / note |
|---|---|---|---|---|
$ROWS"
echo "$BLOCK"
if [ "${NOLOG:-0}" != 1 ]; then echo "$BLOCK" >> "$LOG"; fi
[ $ALLPASS = 1 ]
