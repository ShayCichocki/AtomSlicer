# CPU-stage speed-ups (fiveax-speedups)

Three CPU stages of AtomSlicer made faster with **byte-identical output** (checked with `cmp` on the
`.fxwp` waypoints against the unmodified build, CPU and Metal, with and without an `-F` field).

| Stage | What was slow | Change | Indicative speed-up |
|---|---|---|---|
| Partitioning | after every committed layer, the point BVH over all remaining atoms was rebuilt with a hash lookup per point per level (~85 % of the time); cone queries repeated on every backtrack; the loop branch re-flooded the whole region per attempt | `src/partbvh.h`: a partition-only BVH that builds the same tree without hash lookups (packed points, child bounds in the split pass, parallel subtrees); exact caches for blocker identity, blocked/viable state and flood neighbours; order-free tests for rejected loop-branch starts; incremental viable-set update | 20x on a 2.5M-atom test hook (2410 s -> 117 s, 4 threads), 7-12x on fixtures |
| Toolpath generation | one layer at a time, each with `Parallel::For` over a few thousand vertices (94 % of thread time idle) | one layer per pool task (largest first), order-dependent stitching still serial in layer order; nested `Parallel` calls run inline on workers; flat adjacency arrays in the stitcher | 2.3-6x |
| SDF generation | per cell: BVH closest point + 5 parity rays | warm-start bound from the previous cell (prunes only far subtrees; same leaf order, same winner; falls back if nothing beats it); SAH tree for ray candidates, then the original tests; CPU and Metal | ~1.7x |

Measured together (Apple M5 Pro, machine busy, so indicative): cube20 70 s -> 20 s and curved_bar with a
field 80 s -> 16 s on the Metal build; 90 s -> 38 s and 98 s -> 43 s on the CPU build.

Notes:
- Exactness of the partition depends on reproducing the original `unordered_set` iteration orders (insert
  sequences and bucket counts), so the BVH is still rebuilt exactly as before after each layer; see the
  comments in `src/partition.cpp`. `ATOM_PART_VERIFY=1` recomputes every cached answer the original way.
- `Stitcher::connectRegion` in the original code has a benign-looking race: one branch writes a buffer that
  other elements of the same pass read. Nested runs use the pool's lockstep order so the output matches; a
  proper fix (double buffering) would change output and is left to upstream.
- `ATOM_THREADS=n` caps the thread pool. `ATOM_PART_DUMP`, `ATOM_DUMP_TOOLPATH` write stage inputs for the
  `tools/partbench` harness. Memory: up to one layer per thread in flight during toolpath generation.
