# Building and checking the CUDA AtomSlicer on the Windows desktop (RTX 4090)

Status: **the CUDA build has never been compiled by nvcc or run on an NVIDIA GPU.** The Mac has no NVIDIA GPU and no toolkit. What was checked on the Mac is in `CUDA_LOG.md`: the same host code and kernels ran through a CPU emulation of CUDA, and clang compiled the `.cu` file for sm_89. The steps below are the first real test. Run the equivalence script (step 5) before using the GPU build for anything.

Source: `five-axis-printer/vendor/AtomSlicer-cuda-wip/`. This is a copy of the Metal fork with the CUDA backend added. It is not committed to git yet; that waits for your go-ahead. Copy the whole folder to the desktop, for example with `scp -r`, a USB stick or a OneDrive folder. Inside WSL, put it under `~/`, not `/mnt/c/...`: building on the Windows filesystem is several times slower.

## What you need

| | version | why |
|---|---|---|
| GPU | RTX 4090 (Ada, compute capability 8.9 = `sm_89`), 24 GB | the build defaults to `CMAKE_CUDA_ARCHITECTURES=89` |
| Windows NVIDIA driver | recent Game Ready or Studio driver. CUDA 12.x needs R525 or newer, CUDA 12.8 needs R570 or newer, CUDA 13.x needs R580 or newer (driver minimums are from memory: check the toolkit release notes for the version you install) | in WSL2 the driver comes from Windows |
| CUDA toolkit | **11.8 is the first release that knows `sm_89`; use 12.4 or newer** (any 12.x or 13.x) | nvcc |
| CMake | 3.18 or newer (3.24 or newer for `CMAKE_CUDA_ARCHITECTURES=native`) | Ubuntu 24.04 ships 3.28 |
| C++ compiler | g++ 11 to 13 (WSL / Linux) or MSVC 2022 (native Windows) | must be a host compiler your nvcc version supports |
| Python 3 + numpy + scipy | for `test/compare.py` | |

`CMAKE_CUDA_ARCHITECTURES=native` makes CMake ask the driver which GPU is installed, so it only works when the GPU is visible while CMake configures. The default `89` needs no GPU at configure time. Use `-DCMAKE_CUDA_ARCHITECTURES=86` for an RTX 30xx card and `"89;120"` to cover a 50xx card as well. A binary without code for the card it runs on prints `probe kernel failed ... using CPU` at start-up and then runs entirely on the CPU. It still produces a correct result, just slowly.

## Route A (recommended): WSL2 + Ubuntu 24.04

Why: upstream AtomSlicer and our fork are developed with GCC/clang and CMake on Unix-like systems, the scripts here are bash, and CUDA on WSL2 is supported by NVIDIA. Upstream also says it builds with Visual Studio (route B), but that is untested with our changes.

1. **Windows side** (PowerShell as administrator): `wsl --install -d Ubuntu-24.04`, then reboot. Install or update the normal NVIDIA Windows driver. **Do not install an NVIDIA driver inside Ubuntu.**
2. **Check that the GPU is visible in WSL**:
   ```bash
   nvidia-smi          # should list the RTX 4090; this tool comes from the Windows driver
   ```
3. **Toolkit inside Ubuntu.** Use NVIDIA's WSL-Ubuntu repository and the `cuda-toolkit-*` package only. The `cuda`, `cuda-12`/`cuda-13` and `cuda-drivers` packages would try to install a Linux driver.
   ```bash
   sudo apt-get update && sudo apt-get install -y build-essential cmake ninja-build git python3-venv
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   apt-cache search --names-only '^cuda-toolkit-1[23]-[0-9]+$'   # see which versions exist
   sudo apt-get install -y cuda-toolkit-12-8                      # or the newest 12.x / 13.x listed
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc && source ~/.bashrc
   nvcc --version
   ```
4. **Python for the comparison script**:
   ```bash
   python3 -m venv ~/atomvenv && ~/atomvenv/bin/pip install numpy scipy
   ```
5. **Build everything and prove equivalence** (from the copied folder):
   ```bash
   cd ~/AtomSlicer-cuda-wip
   PY=~/atomvenv/bin/python JOBS=16 test/cuda_desktop.sh
   ```
   The script builds three binaries: `build-cpu/` (the CPU reference), `build-floor/` (the CPU reference with `-ffp-contract=off`, used as the float noise floor) and `build-cuda/` (`-DATOM_CUDA=ON`). It runs a smoke test on `test/cube6.stl` with per-kernel timings, then runs cube6, cube20, holed_block and curved_bar through the CPU build, the floor build, the CUDA build and the CUDA build with `ATOM_CUDA_EXACT=1`. It writes `test/out-cuda/report.md` with two tables: a compare table (verdict, position, axis, volume, coverage, layers, backend line) and a timing table (per stage, wall clock and peak RSS). The whole run takes roughly 10 to 30 minutes, most of it the CPU reference. Send that file back.

   Doing it by hand:
   ```bash
   cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DATOM_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build-cuda -j 16
   ATOM_CUDA_VERBOSE=1 build-cuda/slicer test/cube6.stl -d 0.4 -h 0.5 -o 1 -i 0 -W /tmp/cube6.fxwp
   ```
   The first stderr line must be `[cuda] using NVIDIA GeForce RTX 4090 (compute 8.9, 24.0 GB)`. With `ATOM_CUDA_VERBOSE=1` you then get one `[cuda] <kernel> ... ms (incl. copies)` line per GPU call.

### Reading the result

- `PASS`, identical `True`: byte-identical to the CPU build.
- `FLOOR-PASS`: the plans differ, but by no more than twice what a pure compiler-flag change (`build-floor`) causes, and volume, axis and layer count are inside the strict tolerances. This is the verdict the Metal build got, and it is the expected one here. See "Expected differences" below.
- `FAIL`: something is wrong. Send `test/out-cuda/*.gpu.log` and the summary JSON.
- `GPU-NOT-USED`: no `[cuda] using` line was found (no device, wrong architecture, or `ATOM_CUDA=0` set). The comparison is then meaningless.

### Expected differences vs the desktop CPU build

The GPU kernels copy the floating-point operation order of the *Mac* CPU build (Apple clang, where `a*b + c` inside one expression becomes one fused multiply-add). On the desktop, g++ with `-march=native` fuses multiply-adds by its own rules (`-ffp-contract=fast`), and CUDA's `cosf`, `sinf` and `atan2f` differ from glibc's by an ulp now and then. So **byte-identical output is not expected**. A FLOOR-PASS is. `ATOM_CUDA_EXACT=1` keeps the three cos/sin/atan2 kernels on the CPU, which shows how much of any difference comes from the transcendental functions.

## Route B: native Windows (MSVC + CUDA for Windows). Untested.

Upstream AtomSlicer's README says it builds on Windows with CMake and Visual Studio, and the authors publish Windows binaries. Our fork adds nothing POSIX-only to the CPU or CUDA paths: the POSIX allocator is Metal-only and the CPU emulation is not built here. Expect warnings, because the CMake file passes GCC-style flags (`-Wall -O3 -march=native`), which MSVC warns about and ignores.
1. Install Visual Studio 2022 with "Desktop development with C++", then the CUDA toolkit for Windows (12.4 or newer). It adds the VS integration.
2. In an "x64 Native Tools Command Prompt for VS 2022":
   ```bat
   cmake -S . -B build-cuda -G "Visual Studio 17 2022" -A x64 -DATOM_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build-cuda --config Release -j 16
   cmake -S . -B build-cpu -G "Visual Studio 17 2022" -A x64
   cmake --build build-cpu --config Release -j 16
   build-cuda\Release\slicer.exe test\cube6.stl -d 0.4 -h 0.5 -o 1 -i 0 -W cube6.fxwp
   ```
3. Equivalence: `python test\cuda_equiv.py --cpu build-cpu\Release\slicer.exe --gpu build-cuda\Release\slicer.exe --exact`. Without a floor build you get only the strict verdict; a floor build needs MSVC's `/fp:strict` or similar, and that has not been worked out.

## Run-time switches

| variable | effect |
|---|---|
| `ATOM_CUDA=0` | everything on the CPU (same binary) |
| `ATOM_CUDA_EXACT=1` | keep `tri_smooth_layers`, `tri_smooth_triphasor` and `tri_extract_atoms` (cos/sin/atan2) on the CPU |
| `ATOM_CUDA_SKIP=sdf,normal` | keep kernels whose name starts with a listed prefix on the CPU |
| `ATOM_CUDA_VERBOSE=1` | per-call GPU timings, copies included |
| `ATOM_CUDA_MIN_CELLS=N` | smallest grid level sent to the GPU (default 32768) |
| `ATOM_CUDA_DEVICE=N` | pick a GPU (default 0) |

## Notes and limits

- **Memory:** a grid level is copied to the GPU, stays there for all its sweeps, and is copied back. Device memory is about the largest grid × 40 bytes, plus 16 bytes per cell during atom extraction (4 for the SDF). 24 GB is therefore enough for grids of about 400 million cells. If an allocation fails, that call runs on the CPU and the run carries on.
- **Display watchdog:** if the 4090 also drives your monitor, Windows kills any GPU kernel that runs longer than about 2 s (TDR), and that applies under WSL2 too. Every launch here is one red-black half-sweep or a chunk of at most 4 M cells, so no launch should come close. If one does, the error is printed, the GPU is switched off for the rest of the run, and the CPU finishes the job.
- **Determinism:** the CUDA code has no atomics in its arithmetic (only a "something changed" flag), so repeated GPU runs should be bit-identical to each other. The script does not check this yet; running it twice and comparing the two `gpu` outputs checks it.
- Partitioning and toolpath generation stay on the CPU, as in the Metal build. On small parts they are most of the wall time (see METAL_LOG).
