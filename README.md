# GPU 2D Convolution Optimization

Optimize 2D convolution on NVIDIA GPUs through progressive optimizations: memory coalescing, on-chip memory, register tiling, and concurrent execution.

---

## Quick Start

```bash
# 1. Build
make clean && make

# 2. Test baseline
./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=naive --iters=10 --verify

# 3. Implement your optimizations in src/conv_kernels.cu

# 4. Test your implementation
./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=variant1 --iters=10 --verify

# 5. Profile
ncu --set full -o variant1.ncu-rep \
    ./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=variant1 --iters=1
```

---

## Project Structure

```
hpca-assignment-2025/
├── src/
│   ├── conv_kernels.cu         # ⭐ YOUR IMPLEMENTATION HERE
│   ├── main.cu                 # Benchmark driver (provided)
│   ├── cpu_reference.cpp       # CPU reference (provided)
│   └── utils.cu                # Utilities (provided)
├── include/
│   ├── conv_kernels.cuh        # Function declarations
│   └── timers.h                # Timing utilities
├── Makefile                    # Build configuration
├── README.md                   # This file
└── ASSIGNMENT.md               # Detailed instructions
```

---

## Building

```bash
# Standard build
make clean && make

# Debug build (for cuda-gdb)
make clean && make DEBUG=1

# Clean only
make clean
```

---

## Running

### Command Format
```bash
./gpu_conv [options]
```

### Common Options
- `--n=N` - Batch size (default: 8)
- `--h=H` - Image height (default: 1024)
- `--w=W` - Image width (default: 1024)
- `--k=K` - Kernel size, must be odd (default: 5)
- `--impl=NAME` - Implementation: `naive`, `variant1`, `variant2`, `variant3`, `variant4`, `bonus`
- `--iters=I` - Timing iterations (default: 5)
- `--verify` - Compare with CPU reference

### Standard Test Configuration
```bash
./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=<name> --iters=10 --verify
```
---

## Profiling

### Basic Profiling
```bash
# Profile your implementation
ncu --set full -o variant1.ncu-rep \
    ./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=variant1 --iters=1

# View in GUI
ncu-ui variant1.ncu-rep
```

### Compare Implementations
```bash
# Profile baseline and variant
ncu --set full -o baseline.ncu-rep ./gpu_conv ... --impl=naive --iters=1
ncu --set full -o variant1.ncu-rep ./gpu_conv ... --impl=variant1 --iters=1

# Compare side-by-side
ncu-ui baseline.ncu-rep variant1.ncu-rep
```

### Useful Metrics
```bash
# Memory bandwidth
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./gpu_conv ... --impl=variant1 --iters=1

# L1 cache hit rate
ncu --metrics l1tex__t_sector_hit_rate.pct \
    ./gpu_conv ... --impl=variant2 --iters=1

# Bank conflicts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    ./gpu_conv ... --impl=variant2 --iters=1
```

---

## Verification

**Always use `--verify` during development:**
```bash
./gpu_conv --impl=variant1 --verify
```

**Expected output:**
```
✓ PASS: Results match CPU reference
Maximum absolute difference vs CPU: 0.00000024
```

**If you see large differences (>0.001), you have a bug!**

---

## Development Workflow

```bash
# 1. Edit code
vim src/conv_kernels.cu

# 2. Build
make clean && make

# 3. Test with small input first
./gpu_conv --n=2 --h=256 --w=256 --k=5 --impl=variant1 --verify

# 4. Test with target config
./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=variant1 --iters=10 --verify

# 5. Profile
ncu --set full -o variant1.ncu-rep \
    ./gpu_conv --n=16 --h=2048 --w=2048 --k=11 --impl=variant1 --iters=1

# 6. Iterate
```

---

## Compare All Implementations

```bash
# Create compare.sh
cat > compare.sh << 'EOF'
#!/bin/bash
CONFIG="--n=16 --h=2048 --w=2048 --k=11 --iters=10"
for impl in naive variant1 variant2 variant3 variant4; do
    echo "=== $impl ==="
    ./gpu_conv $CONFIG --impl=$impl --verify | grep -E "(Average|Throughput)"
done
EOF

chmod +x compare.sh
./compare.sh
```

---

## Debugging

### Test with Small Inputs
```bash
./gpu_conv --n=1 --h=64 --w=64 --k=3 --impl=variant1 --verify
```

### Use CUDA Debugger
```bash
make clean && make DEBUG=1
cuda-gdb ./gpu_conv
# In gdb:
(cuda-gdb) run --n=1 --h=64 --w=64 --k=3 --impl=variant1 --verify
```

---

## Troubleshooting

**Build fails:**
```bash
# Check CUDA is in PATH
nvcc --version
export PATH=/usr/local/cuda/bin:$PATH

# Rebuild from scratch
make clean && make
```

**Wrong results:**
- Test with smaller inputs
- Check boundary handling
- Verify `__syncthreads()` usage

**Out of memory:**
- Reduce batch size: `--n=4`
- Reduce dimensions: `--h=1024 --w=1024`

---

## Submission Checklist

- [ ] All variants compile: `make clean && make`
- [ ] All variants pass verification: `--verify`
- [ ] Performance targets met (see table above)
- [ ] Code is well-commented
- [ ] Report includes profiling analysis
- [ ] NCU reports included (`.ncu-rep` files)

---

## Resources

- **Detailed Instructions:** See `ASSIGNMENT.md`
- **CUDA Documentation:** https://docs.nvidia.com/cuda/
- **Nsight Compute Guide:** https://docs.nvidia.com/nsight-compute/

---

## Getting Help

- Check `ASSIGNMENT.md` for implementation hints
- Review provided baseline implementation
- Use NCU profiler to identify bottlenecks
- Office hours: [TBD]

---

**Start coding in `src/conv_kernels.cu` and aim for 10× speedup! 🚀**
