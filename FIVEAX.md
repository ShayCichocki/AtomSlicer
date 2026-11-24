# fiveax fork of AtomSlicer

AtomSlicer (Giovanni Cocco and Inria, MIT, https://github.com/iota97/AtomSlicer) with additions for a
five-axis FDM pipeline. Everything is opt-in: a default build is upstream's CPU code.

## Additions
- **Metal GPU backend** (`-DATOM_METAL=ON`, macOS): the grid red-black Gauss-Seidel smoothers (tool
  orientations, tri-phasor layers/tangents), atom extraction and the SDF cells run as Metal compute
  kernels. Shaders are compiled at run time (no offline Metal toolchain needed). `ATOM_METAL=0` in the
  environment forces the CPU path; `ATOM_METAL_EXACT=1` keeps cos/sin/atan2 on the CPU.
- **`-F field.fxfd`**: external layer-normal + road-direction field (e.g. from a stress analysis), used
  through AtomSlicer's support-free normal objective. Format in `src/extfield.h`. Note the convention:
  the tangent slot is the in-layer direction ACROSS the roads (n x road).
- **`-W out.fxwp`**: raw waypoints (position, nozzle axis, deposit flag) in the STL frame, for an
  external exporter/verifier.
- **Tests** (`test/`): fixtures, `compare.py` (CPU vs GPU waypoint comparison with a float-noise-floor
  verdict), `run_bench.sh`, `make_fixtures.py`, `make_field.py`.

## Build
```
cmake -S . -B build-ref -G Ninja -DCMAKE_BUILD_TYPE=Release            # CPU reference
cmake -S . -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DATOM_METAL=ON
cmake -S . -B build-perturb -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-ffp-contract=off   # float floor
```

## How equivalence is judged
GPU and CPU builds are not bit-identical (summation order, fused multiply-adds). Small float differences can
flip discrete choices (which atom, which path route), moving whole path pieces by about a bead. So the GPU
build is compared against the CPU build AND against the float noise floor (a CPU build that differs only by
`-ffp-contract=off`): it passes when it moves the plan no more than 2x what rounding alone does, with volume
and layer count strict. Downstream, the fiveax pipeline checks export verification and FEA load factors.

Measured (Apple M5 Pro, 0.2 mm layers): on a 20 cm^3 test hook the GPU stages are 2-10x faster
(e.g. tri-phasor 31 s -> 4 s, atom extraction 18 s -> 1.8 s), and the GPU plans sit inside the float floor
(e.g. position p99 0.2026 mm vs 0.2025 mm floor). Whole runs are limited by CPU-only stages (partitioning,
toolpath generation).
