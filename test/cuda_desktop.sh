#!/bin/bash
# Build the three binaries needed to prove the CUDA backend on an NVIDIA machine (Linux or
# WSL2), then run the fixtures through all of them and compare the waypoints.
#   build-cpu/    plain CPU build (the reference)
#   build-floor/  CPU build with -ffp-contract=off: its difference from build-cpu is the
#                 pipeline's own float noise floor (see docs/METAL_LOG.md)
#   build-cuda/   -DATOM_CUDA=ON
# Usage (from the repository root):  test/cuda_desktop.sh [fixture.stl ...]
# Env: ARCH=89 (CMAKE_CUDA_ARCHITECTURES; "native" works when the GPU is visible)
#      JOBS=8  ARGS="-d 0.4 -h 0.5 -o 1 -i 0"  PY=python3  SKIP_BUILD=1  EXACT=1
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ARCH="${ARCH:-89}"
JOBS="${JOBS:-8}"
PY="${PY:-python3}"
ARGS="${ARGS:--d 0.4 -h 0.5 -o 1 -i 0}"
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)

if [ "${SKIP_BUILD:-0}" != 1 ]; then
  echo "== toolchain"
  cmake --version | head -1
  nvcc --version | tail -2 || { echo "nvcc not on PATH (export PATH=/usr/local/cuda/bin:\$PATH)"; exit 2; }
  nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv || true
  cmake -S . -B build-cpu   "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release
  cmake -S . -B build-floor "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-ffp-contract=off
  cmake -S . -B build-cuda  "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release -DATOM_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
  cmake --build build-cpu   -j "$JOBS"
  cmake --build build-floor -j "$JOBS"
  cmake --build build-cuda  -j "$JOBS"
fi

echo "== smoke test: cube6 on the GPU build"
ATOM_CUDA_VERBOSE=1 build-cuda/slicer test/cube6.stl $ARGS -W /tmp/atom_cube6_cuda.fxwp 2>&1 | grep -E '^\[cuda\]|Total time' | head -40 || true

mkdir -p test/out-cuda
EX=()
[ "${EXACT:-1}" = 1 ] && EX=(--exact)
FIX=("$@")
[ ${#FIX[@]} -eq 0 ] && FIX=(test/cube6.stl test/cube20.stl test/holed_block.stl test/curved_bar.stl)
"$PY" test/cuda_equiv.py --cpu build-cpu/slicer --floor build-floor/slicer --gpu build-cuda/slicer "${EX[@]}" \
      --args "$ARGS" --out test/out-cuda --fixtures "${FIX[@]}" | tee test/out-cuda/report.md
