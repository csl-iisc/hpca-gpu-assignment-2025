# CUDA 2D Convolution Optimization Assignment

## Overview
Optimize a 2D convolution kernel through a series of progressive improvements, analyzing performance at each stage using NVIDIA profiling tools.

## Submission Requirements

### File Structure
```
<last_5_digits_of_SR_ID>/
├── report.pdf
└── conv_kernels.cu
```

### Files to Submit
1. **conv_kernels.cu** - Your implementation with all optimized variants
2. **report.pdf** - Your analysis report

---

## Assignment Tasks

### Variant 1: Global-Memory Access Pattern

**Objective:** Restructure the thread/data mapping so that the hardware can merge per-thread loads into fewer, fuller memory transactions.

**Goal:**
- Increase effective global-memory bandwidth by maximizing bytes used per transaction and minimizing transactions per warp.

**What to Measure (before & after):**
- Global load efficiency / requested vs. delivered bytes
- DRAM read throughput (GB/s) and "transactions per request" counters
- Kernel time and MPix/s

**Hints (discovery-oriented):**
- Inspect how a warp's threads (lane 0..31) walk the image in memory. Are neighboring lanes reading neighboring addresses, or are they striding?
- Revisit your mapping from (threadIdx.x, threadIdx.y) → (row, col). Which dimension in memory is contiguous, and do lanes advance along it?
- Consider block shapes where each warp spans one logical row of the tile rather than splitting a warp across multiple rows
- The order of inner loops matters: move the loop that advances along the contiguous memory dimension into the per-lane direction
- When alignment permits, loading wider types (e.g., 16-byte aligned chunks) reduces the number of memory transactions. Handle tails safely.

---

### Variant 2: On-Chip Memory (Shared + Constant)

**Approach:**

Begin by profiling the naive convolution implementation using NVIDIA Nsight Compute. Record key metrics such as memory bandwidth utilization, cache hit rates, IPC, and other relevant performance indicators.

Next, study the various GPU memory types — both on-chip and off-chip — and discuss their access latencies and bandwidths. Explain which of these memories are being used by the naive convolution kernel and how.

Then, implement Variant 2 by modifying the kernel to make use of different on-chip memory spaces. Specifically, explore the use of shared memory and constant memory to improve data reuse and reduce global memory traffic.

After your optimization, re-profile the kernel and report changes in cache utilization, bandwidth utilization, and overall performance.

Finally, observe and explain an interesting phenomenon: certain optimizations may increase memory bandwidth utilization while decreasing cache hit rates, yet still lead to better performance. Provide a detailed reasoning for why this happens, relating it to reduced cache dependence, more efficient data reuse, and improved throughput across the GPU memory hierarchy.

---

### Variant 3: Register-Level Optimization and Data Locality

**Approach:**

In this task, you will investigate the role of the GPU's register file and how exploiting data locality at the thread level can further improve performance beyond what shared and global memory optimizations achieve.

Begin by profiling your previous variant and examine metrics related to register utilization, instruction-level parallelism (ILP), and arithmetic efficiency. Observe how many registers are used per thread and whether memory operations still dominate execution time.

Next, study how the GPU register file serves as the fastest storage resource available to each thread. Think about ways to reuse data already loaded into registers to reduce redundant memory accesses and improve computational intensity. Consider whether each thread could perform more useful work by computing multiple nearby output elements rather than just one.

Modify the kernel to take advantage of this thread-level reuse and the available registers. After your optimization, re-profile and report changes in achieved FLOP/s, register utilization, and memory bandwidth usage.

Finally, discuss in your report how locality within the register file and the reuse of data across computations can reduce memory pressure and improve throughput. Relate your findings to the GPU's execution model and to the balance between register usage, occupancy, and ILP.

---

### Variant 4 (Bonus): Multi-Stream Concurrent Execution

**Objective:** Use Nsight Systems to understand the end-to-end timeline and then improve throughput by overlapping independent work with CUDA streams.

**Goal:**
- Reduce idle gaps on the copy and compute engines by overlapping operations (e.g., host to device transfers with kernel execution) across a large batch.

**What to Examine in Nsight Systems (Before):**
- Are H2D/D2H copies serialized with kernel launches?
- Do copy engines (C/E) or SMs sit idle between batches?
- Where are the longest gaps on the timeline (host prep, copies, kernels)?

**What to Measure (before & after):**
- End-to-end time per full batch; GPU utilization (%), copy engine utilization
- Degree of overlap visible on the NSYS timeline (copies concurrent with kernels)
- Any change in kernel performance (avoid starving compute with too many small chunks)

---

## Report Format

Your report should include the following for each variant:

### 1. Implementation Strategy
- Description of your approach
- Key design decisions and tradeoffs

### 2. Performance Analysis
- Profiling data from Nsight Compute/Systems
- Relevant metrics as specified in each variant
- Speedup comparisons
- Timeline visualizations (for Variant 4)

### 3. Discussion
- Why the optimization works
- Explanation of observed phenomena
- Relationship to GPU architecture and memory hierarchy
- Insights gained from profiling

---

## Evaluation Criteria

- **Correctness:** Kernels produce correct output
- **Performance:** Speedup achieved for each variant
- **Code Quality:** Readability, comments, proper resource management
- **Analysis Depth:** Understanding of optimizations, profiling data interpretation, and architectural insights

---

## Resources
- CUDA C Programming Guide
- NVIDIA Nsight Compute Documentation
- NVIDIA Nsight Systems Documentation
- CUDA Best Practices Guide

---

## Submission Deadline
Details on how and when to upload the assignment will be shared soon.

## Academic Integrity
All code must be your own work. Plagiarism will result in consequences per institute policy.
