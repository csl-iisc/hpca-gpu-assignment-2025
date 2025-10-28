
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "../include/conv_kernels.cuh"

// =============================================================================
// BASELINE IMPLEMENTATION (PROVIDED - DO NOT MODIFY)
// =============================================================================
// This is an intentionally inefficient implementation for comparison purposes.

static __device__ __forceinline__ size_t idx3(int batch_index, int row, int col,
                                               int height, int width) {
    return static_cast<size_t>(batch_index) * height * width +
           static_cast<size_t>(row) * width +
           col;
}

__global__ void kernel_conv2d_baseline(const float* __restrict__ input_images,
                                       const float* __restrict__ kernel,
                                       float* __restrict__ output_images,
                                       int batch_size, int height, int width,
                                       int kernel_size) {
    int batch_index = blockIdx.z;

    int row = blockIdx.y * blockDim.y + threadIdx.x;  // Note: threadIdx.x for row
    int col = blockIdx.x * blockDim.x + threadIdx.y;  // Note: threadIdx.y for col

    if (batch_index >= batch_size || row >= height || col >= width) {
        return;
    }

    int radius = (kernel_size - 1) / 2;
    float accumulated_value = 0.0f;

    for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
        int input_row = row + kernel_row - radius;
        if (input_row < 0 || input_row >= height) continue;

        for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
            int input_col = col + kernel_col - radius;
            if (input_col < 0 || input_col >= width) continue;

            float input_pixel = input_images[idx3(batch_index, input_row, input_col,
                                                   height, width)];
            float kernel_weight = kernel[kernel_row * kernel_size + kernel_col];
            accumulated_value += input_pixel * kernel_weight;
        }
    }

    output_images[idx3(batch_index, row, col, height, width)] = accumulated_value;
}

void conv2d_baseline(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    dim3 threads_per_block(16, 16, 1);
    dim3 blocks_per_grid(
        (width + threads_per_block.x - 1) / threads_per_block.x,
        (height + threads_per_block.y - 1) / threads_per_block.y,
        batch_size
    );

    kernel_conv2d_baseline<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input, kernel, output, batch_size, height, width, kernel_size
    );
}

// =============================================================================
// VARIANT 1: GLOBAL-MEMORY ACCESS PATTERN (BANDWIDTH EFFICIENCY)
// =============================================================================
// TODO: Restructure the thread/data mapping so that the hardware can merge
//       per-thread loads into fewer, fuller memory transactions.
//
// GOAL:
// - Increase effective global-memory bandwidth by maximizing bytes used per
//   transaction and minimizing transactions per warp.
//
// WHAT TO MEASURE (before & after):
// - L1TEX/L2: average bytes used per sector (aim ~32/32 for 32B sectors).
// - Global load efficiency / requested vs. delivered bytes.
// - DRAM read throughput (GB/s) and “transactions per request” counters.
// - Kernel time and MPix/s.
//
// HINTS (discovery-oriented):
// - Inspect how a warp’s threads (lane 0..31) walk the image in memory.
//   Are neighboring lanes reading neighboring addresses, or are they striding?
// - Revisit your mapping from (threadIdx.x, threadIdx.y) → (row, col).
//   Which dimension in memory is contiguous, and do lanes advance along it?
// - Consider block shapes where each warp spans one logical row of the tile
//   rather than splitting a warp across multiple rows.
// - The order of inner loops matters: move the loop that advances along the
//   contiguous memory dimension into the per-lane direction.
// - When alignment permits, loading wider types (e.g., 16-byte aligned chunks)
//   reduces the number of memory transactions. Handle tails safely.
//
// =============================================================================


void conv2d_variant1(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    // TODO: Configure and launch your kernel

    // Your code here
}

// =============================================================================
// VARIANT 2: ON-CHIP MEMORY (SHARED + CONSTANT)
// =============================================================================
//
// In this task, you will explore the different levels of the GPU memory hierarchy
// and analyze how they impact performance.
//
// Begin by profiling the naive convolution implementation using NVIDIA Nsight Compute.
// Record key metrics such as memory bandwidth utilization, cache hit rates, IPC, and
// other relevant performance indicators.
//
// Next, study the various GPU memory types — both on-chip and off-chip — and discuss
// their access latencies and bandwidths. Explain which of these memories are being
// used by the naive convolution kernel and how.
//
// Then, implement Variant 2 by modifying the kernel to make use of different on-chip
// memory spaces. Specifically, explore the use of shared memory and constant memory
// to improve data reuse and reduce global memory traffic.
//
// After your optimization, re-profile the kernel and report changes in cache
// utilization, bandwidth utilization, and overall performance.
//
// Finally, observe and explain an interesting phenomenon: certain optimizations may
// increase memory bandwidth utilization while decreasing cache hit rates, yet still
// lead to better performance. Provide a detailed reasoning for why this happens,
// relating it to reduced cache dependence, more efficient data reuse, and improved
// throughput across the GPU memory hierarchy.
//
// =============================================================================


void conv2d_variant2(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {

    // Your code here
}

// =============================================================================
// VARIANT 3: REGISTER-LEVEL OPTIMIZATION AND DATA LOCALITY
// =============================================================================
//
// In this task, you will investigate the role of the GPU’s register file and
// how exploiting data locality at the thread level can further improve
// performance beyond what shared and global memory optimizations achieve.
//
// Begin by profiling your previous variant and examine metrics related to
// register utilization, instruction-level parallelism (ILP), and arithmetic
// efficiency. Observe how many registers are used per thread and whether
// memory operations still dominate execution time.
//
// Next, study how the GPU register file serves as the fastest storage resource
// available to each thread. Think about ways to reuse data already loaded into
// registers to reduce redundant memory accesses and improve computational
// intensity. Consider whether each thread could perform more useful work by
// computing multiple nearby output elements rather than just one.
//
// Modify the kernel to take advantage of this thread-level reuse and the
// available registers. After your optimization, re-profile and report changes
// in achieved FLOP/s, register utilization, and memory bandwidth usage.
//
// Finally, discuss in your report how locality within the register file and
// the reuse of data across computations can reduce memory pressure and improve
// throughput. Relate your findings to the GPU’s execution model and to the
// balance between register usage, occupancy, and ILP.
//
// =============================================================================


void conv2d_variant3(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {

    // Your code here
}

// =============================================================================
// BONUS: MULTI-STREAM CONCURRENT EXECUTION
// =============================================================================
//
// In this task, use Nsight Systems to understand the end-to-end timeline and
// then improve throughput by overlapping independent work with CUDA streams.
//
// GOAL:
// - Reduce idle gaps on the copy and compute engines by overlapping operations
//   (e.g., host to device transfers with kernel execution) across a large batch.
//
// WHAT TO EXAMINE IN NSIGHT SYSTEMS (BEFORE):
// - Are H2D/D2H copies serialized with kernel launches?
// - Do copy engines (C/E) or SMs sit idle between batches?
// - Where are the longest gaps on the timeline (host prep, copies, kernels)?
//
// WHAT TO MEASURE (before & after):
// - End-to-end time per full batch; GPU utilization (%), copy engine utilization.
// - Degree of overlap visible on the NSYS timeline (copies concurrent with kernels).
// - Any change in kernel performance (avoid starving compute with too many small chunks).
//
// =============================================================================


void conv2d_variant4(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size,
                     cudaStream_t stream) {
    // TODO: Configure and launch

    // Your code here
}

// =============================================================================
// BONUS ROUND
// =============================================================================


void conv2d_bonus(const float* input, const float* kernel, float* output,
                  int batch_size, int height, int width, int kernel_size,
                  int chunk_size, int num_streams) {
    // TODO: Implement multi-stream version
    // You can reuse any of your previous kernels

    // Your code here
}