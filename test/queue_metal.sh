#!/bin/bash
# Sequential benchmark queue for the Metal build (run after the step-2 bench finished).
cd /Users/scichocki/personal/cad-box/five-axis-printer/vendor/AtomSlicer-metal
T=/private/tmp/claude-501/-Users-scichocki-personal-cad-box/d0a9da66-f3bd-4df0-b164-0068f285f903/tasks
until grep -q "rc=" $T/berp7go9h.output 2>/dev/null; do sleep 10; done
while pgrep -f "build-perturb/slicer" >/dev/null; do sleep 10; done
REUSE_REF=1 NEW=build-metal/slicer test/run_bench.sh "steps 3-5: Metal (ATOM_METAL=ON): grid smoothers + extractAtoms + SDF on GPU" test/cube20.stl test/holed_block.stl test/curved_bar.stl
echo "metal bench rc=$?"
ATOM_METAL_EXACT=1 REUSE_REF=1 NEW=build-metal/slicer test/run_bench.sh "Metal, ATOM_METAL_EXACT=1 (cos/sin/atan2 kernels kept on CPU)" test/cube20.stl test/holed_block.stl
echo "exact bench rc=$?"
ARGS="-d 0.4 -h 0.5 -o 0 -i 0 -F test/field_tilt.fxfd" NEW=build-metal/slicer test/run_bench.sh "Metal with the fiveax -F external field (-o 0)" test/curved_bar.stl
echo "field bench rc=$?"
echo QUEUE_DONE
