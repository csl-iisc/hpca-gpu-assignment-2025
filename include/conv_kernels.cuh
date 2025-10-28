#pragma once
#include <cuda_runtime.h>

// Students fill these. Each should produce correct output but faster performance
// as they explore different optimization ideas.

void conv2d_baseline (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);

void conv2d_variant1 (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);

void conv2d_variant2 (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);

void conv2d_variant3 (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);

void conv2d_variant4 (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);

void conv2d_variant5 (const float* I, const float* K, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);
   

// Optional exploration / creative extension.
void conv2d_bonus (const float* I, const float* K1, const float* K2, float* O,
                      int N, int H, int W, int k, cudaStream_t stream);
