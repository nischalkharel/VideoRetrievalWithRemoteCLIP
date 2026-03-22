# CRC Setup Reference — P1_Serial (Video Embedding Pipeline)

## Connection

```bash
ssh nik96@h2p.crc.pitt.edu
```

Home directory: `/ihome/ageorge/nik96`
Project directory: `~/ece2166/Project`

## Project Directory Structure

```
~/ece2166/Project/
├── CMakeLists.txt
├── run_serial.slurm
├── src/
│   ├── P1_Serial.cpp        # Main code (compiled with ABI=0 for LibTorch)
│   ├── cv_wrapper.h          # C-linkage header for OpenCV
│   └── cv_wrapper.cpp        # Compiled with ABI=1 to match OpenCV
├── build/                    # CMake build directory
├── models/
│   └── RemoteCLIP-ViT-L-14_scripted.pt  (1.2 GB)
└── data/
    ├── short_video.mp4       (398 MB)
    └── short_video_embeddings.pt  (output)
```

## Modules to Load

```bash
module purge
module load gcc/12.2.0
module load python/pytorch_251_311_cu124
module load cuda/12.4.1
module load cmake/3.27.7          # only needed for cmake step
```

**Do NOT** load `opencv/4.8.0` — it was built without FFMPEG/videoio support. The OpenCV bundled inside the PyTorch module (`/software/rhel9/manual/install/python/pytorch_251_311_cu124/lib/`) has full video support.

## The ABI Problem (Key Learning)

LibTorch forces `_GLIBCXX_USE_CXX11_ABI=0` (old GCC string ABI). OpenCV was built with `ABI=1` (new ABI). This means any function using `std::string` in its signature cannot link across the two libraries — the mangled symbols don't match.

**Solution:** A thin C wrapper (`cv_wrapper.cpp`) compiled with `ABI=1` exposes OpenCV functions through `extern "C"` (no C++ types cross the boundary). The main code compiled with `ABI=0` calls OpenCV only through these C functions.

The preprocessing (resize, BGR→RGB, normalize) was moved to pure PyTorch tensor ops so only video I/O (open, read, release) goes through the wrapper.

This is a **Linux/GCC-only issue** — MSVC on Windows doesn't have this ABI split, which is why it compiled fine on your PC.

## Building

```bash
cd ~/ece2166/Project

# Load modules (no cmake needed after first build)
module purge
module load gcc/12.2.0
module load python/pytorch_251_311_cu124
module load cuda/12.4.1
module load cmake/3.27.7

# Build
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..
make
```

After first build, you only need `make` unless you change CMakeLists.txt.

The cmake warnings (nvtx3, kineto, shorthash) are harmless — just missing optional components.

## Running Jobs

Jobs must run on GPU compute nodes, not login nodes. Submit via Slurm.

### Slurm File (run_serial.slurm)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --time=0:30:00
#SBATCH --job-name=p1_serial

module purge
module load gcc/12.2.0
module load python/pytorch_251_311_cu124
module load cuda/12.4.1

echo "Node: $(hostname)"
echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader

cd /ihome/ageorge/nik96/ece2166/Project
./build/P1_Serial
```

### GPU Partitions (from most to least available)

| Partition | GPUs | Notes |
|-----------|------|-------|
| `l40s` | L40S | Good availability, worked quickly |
| `rtx6k` | RTX 6000 | Had 5 idle nodes |
| `a100` | A100 | Popular, often long queue |
| `a100_nvlink` | A100 NVLink | Multi-GPU, usually busy |
| `h200` | H200 | Newest, very limited |

If one partition has long wait, switch to another.

### Useful Commands

```bash
sbatch run_serial.slurm                # Submit job
squeue -M gpu -u $USER                 # Check job status (PD=pending, R=running)
scancel -M gpu <jobid>                 # Cancel job
cat slurm-<jobid>.out                  # Read output after job finishes
crc-sinfo | grep -i gpu               # Check GPU node availability
```

## SCP Commands (Windows → CRC)

```
scp "C:\path\to\file" nik96@h2p.crc.pitt.edu:/ihome/ageorge/nik96/ece2166/Project/src/
```

CRC → Windows:
```
scp nik96@h2p.crc.pitt.edu:/ihome/ageorge/nik96/ece2166/Project/data/output.pt .
```

## Key Paths on CRC

| What | Path |
|------|------|
| PyTorch/LibTorch root | `/software/rhel9/manual/install/python/pytorch_251_311_cu124` |
| LibTorch cmake config | `...above.../lib/python3.11/site-packages/torch/share/cmake` |
| OpenCV libs (with videoio) | `...above.../lib/libopencv_*.so` |
| OpenCV headers | `...above.../include/opencv4` |
| System OpenCV (broken videoio) | `/usr/lib64/libopencv_*.so`, `/usr/include/opencv4` |
| Spack OpenCV module (no videoio) | Don't use — built with all features OFF |

## Next Steps

- Review slurm output to verify serial baseline runs correctly
- Write the OpenMP parallel version (producer-consumer pattern with multiple CPU threads feeding GPU)
- For the parallel version, use `--ntasks-per-node=16` and `export OMP_NUM_THREADS=N` in the slurm script
- Part 2: vector retrieval comparison (serial CPU, FAISS GPU, Gemini APU)
