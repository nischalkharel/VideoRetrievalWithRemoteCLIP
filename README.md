# RemoteCLIP Video Retrieval Pipeline

A high-performance text-based video retrieval system for satellite and remote sensing footage, built on [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP). The pipeline embeds video frames using RemoteCLIP's vision encoder, then retrieves relevant clips from natural language queries via cosine similarity search. Implementations span serial CPU, parallel CPU (OpenMP), and GPU (CUDA/cuBLAS) backends, with benchmarking across all platforms.

Developed as part of research at the [NSF SHREC Center](https://nsf-shrec.org/), University of Pittsburgh.

---

## Overview

The project is structured in two phases:

### Phase 1 — Parallel Video Embedding

Extracts frames from satellite/remote sensing video and computes RemoteCLIP embeddings in parallel using a producer-consumer architecture.

- **C++ implementation** using LibTorch + OpenCV
- **OpenMP parallelism** with multiple producer threads feeding a shared GPU consumer
- TorchScript model export (with fused attention kernels disabled for compatibility)
- Scales up to ~4× speedup with 4 producer threads before hitting GPU saturation

### Phase 2 — Text-Based Video Retrieval

Given a natural language query, encodes the text with RemoteCLIP's text encoder and performs brute-force k-nearest neighbor (kNN) cosine similarity search against the precomputed frame embeddings.

Retrieval is implemented across three backends:

| Backend | Implementation | Key Detail |
|---------|---------------|------------|
| Serial CPU | C++ | Baseline single-threaded dot product |
| Parallel CPU | C++ with OpenMP | Scales near-linearly up to 16 cores |
| GPU | Python with `torch.mv()` / cuBLAS | ~185× faster than serial (warm) |

---

## Key Findings

- **OpenMP Scaling**: Near-linear speedup up to 16 cores on a dual-socket system. Performance degrades at 32 cores due to NUMA cross-socket memory access overhead.
- **NUMA-Aware Pinning**: Binding threads to a single NUMA node with `OMP_PROC_BIND=close` and `OMP_PLACES=cores` achieves ~38× speedup over serial.
- **GPU Cold-Start Overhead**: While GPU warm search is ~185× faster than serial CPU, end-to-end GPU execution (including CUDA initialization and PCIe data transfer) is *slower* than CPU serial — a significant result for latency-sensitive edge/space deployments.
- **OpenMP + PyTorch Conflict**: PyTorch internally calls `omp_set_dynamic(1)`, which silently overrides the user-specified thread count. Fixed by explicitly calling `omp_set_dynamic(0)` before the OpenMP parallel region.

---

## Tech Stack

- **Languages**: C++, Python
- **Libraries**: LibTorch, OpenCV, PyTorch, OpenMP
- **Model**: RemoteCLIP (ViT-L/14 backbone, TorchScript export)
- **Build**: CMake + Ninja (CUDA 12.6, LibTorch 2.10.0)
- **HPC**: University of Pittsburgh CRC cluster (SLURM, multi-node)

---

## Project Structure

```
├── cpp/
│   ├── embedding/          # Phase 1: parallel video embedding pipeline
│   │   ├── main.cpp        # Producer-consumer OpenMP pipeline
│   │   └── CMakeLists.txt
│   └── retrieval/          # Phase 2: kNN retrieval (serial + OpenMP)
│       ├── retrieval.cpp   # Brute-force cosine similarity search
│       └── CMakeLists.txt
├── python/
│   ├── embed_video.py      # Python embedding baseline
│   ├── retrieval_gpu.py    # GPU retrieval via torch.mv()
│   └── export_model.py     # TorchScript export script
├── scripts/
│   ├── benchmark.sh        # Automated benchmarking across backends
│   └── slurm/              # SLURM job submission scripts for CRC
└── README.md
```

> **Note**: The directory structure above reflects the intended layout. Some paths may differ on the current branch.

---

## Building

### Prerequisites

- CMake ≥ 3.18
- CUDA Toolkit ≥ 12.0
- LibTorch (C++ distribution, ≥ 2.0)
- OpenCV ≥ 4.0

### Compile

```bash
mkdir build && cd build
cmake .. -G Ninja \
    -DCMAKE_PREFIX_PATH=/path/to/libtorch \
    -DOpenCV_DIR=/path/to/opencv
ninja
```

### Running

```bash
# Phase 1: Embed video frames
./embedding --video /path/to/video.mp4 --model /path/to/remoteclip.pt --threads 4

# Phase 2: Retrieve by text query
./retrieval --embeddings /path/to/embeddings.bin --query "buildings near coastline" --top_k 5
```

---

## Benchmarking

Benchmarks were conducted on Pitt's CRC cluster (dual-socket nodes with NVIDIA L40S GPUs).

| Backend | Configuration | Relative Speedup |
|---------|--------------|-------------------|
| Serial CPU | 1 thread | 1× (baseline) |
| OpenMP | 16 cores, NUMA-pinned | ~38× |
| OpenMP | 32 cores, cross-socket | Degrades (NUMA penalty) |
| GPU (warm) | `torch.mv()` / cuBLAS | ~185× |
| GPU (cold, end-to-end) | Including CUDA init + PCIe | < 1× (slower than serial) |

---

## Upcoming Work

- **GSI Gemini II APU**: Benchmarking retrieval on GSI's analog processing unit using the GNLPY/Copperhead APIs. Investigating float32 vs. binary (LSH) dot product support for approximate nearest neighbor search. *(In progress)*

---

## Acknowledgments

This work is supported by the NSF Center for Space, High-Performance, and Resilient Computing (SHREC) at the University of Pittsburgh.

RemoteCLIP: Chen et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing," IEEE TGRS, 2024.
