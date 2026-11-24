#!/bin/bash
# Self-check of the CUDA backend on a machine with no CUDA toolkit: Homebrew clang parses
# src/cuda/gpu_cuda.cu as CUDA against stub/cuda_runtime.h.
#   1. host side:   --cuda-host-only  -fsyntax-only  (host code, launch syntax, API calls)
#   2. device side: --cuda-device-only -S for sm_89 -> PTX (real NVPTX code generation of
#      every kernel, catches host-only calls from device code, recursion, etc.)
#   3. a look at the PTX: no approximate math, explicit fma only, stack frames.
# What it does NOT check: nvcc's own front end (EDG), linking against the real cudart,
# ptxas/SASS, and anything at run time.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../src"
CLANG="${CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
OUT="${OUT:-$HERE/out}"
mkdir -p "$OUT"
# -D__CUDACC__ and -include cuda_runtime.h mimic what nvcc does implicitly.
COMMON=(-x cuda -D__CUDACC__ -include cuda_runtime.h -std=c++17 -nocudainc -nocudalib --cuda-gpu-arch=sm_89 -I "$HERE/stub" -DATOM_CUDA=1 -Wall -Wno-unknown-cuda-version -ffp-contract=off -Xclang -target-sdk-version=12.4)
echo "== host side (clang $("$CLANG" --version | head -1 | sed 's/.*version //'))"
"$CLANG" "${COMMON[@]}" --cuda-host-only -fsyntax-only "$SRC/cuda/gpu_cuda.cu"
echo "   ok"
echo "== device side -> PTX (sm_89)"
"$CLANG" "${COMMON[@]}" --cuda-device-only -O3 -S -o "$OUT/gpu_cuda_sm89.ptx" "$SRC/cuda/gpu_cuda.cu"
echo "   ok: $OUT/gpu_cuda_sm89.ptx ($(wc -l < "$OUT/gpu_cuda_sm89.ptx") lines)"
P="$OUT/gpu_cuda_sm89.ptx"
echo "== PTX inspection"
echo "   kernels (.entry):        $(grep -c '^\.visible \.entry\|\.entry ' "$P" || true)"
grep -o '\.entry [a-zA-Z0-9_]*' "$P" | sed 's/^/     /'
echo "   approximate ops (.approx, should be 0 outside libdevice calls): $(grep -c '\.approx' "$P" || true)"
echo "   ftz ops (should be 0):   $(grep -c '\.ftz' "$P" || true)"
echo "   fma.rn.f32:              $(grep -c 'fma\.rn\.f32' "$P" || true)"
echo "   fma.rn.f64:              $(grep -c 'fma\.rn\.f64' "$P" || true)"
echo "   unresolved libdevice calls (resolved by nvcc/libdevice in a real build):"
grep -o 'call[^;]*__nv_[a-z0-9_]*' "$P" | grep -o '__nv_[a-z0-9_]*' | sort | uniq -c | sed 's/^/     /' || true
echo "   local stack per kernel (.local depot bytes):"
grep -o '__local_depot[0-9]*\[[0-9]*\]' "$P" | sed 's/^/     /' || true
