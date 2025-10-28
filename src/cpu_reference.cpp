#include <vector>
#include <cmath>
#include <algorithm>

// CPU reference implementation of 2D convolution with SAME padding
// This performs zero-padded convolution on a batch of single-channel images
// 
// Parameters:
//   input_images: Input image batch [N × H × W] stored as flat array
//   kernel: Convolution kernel [k × k] stored as flat array
//   output_images: Output image batch [N × H × W] stored as flat array
//   batch_size: Number of images in the batch
//   height: Height of each image in pixels
//   width: Width of each image in pixels
//   kernel_size: Size of the square convolution kernel (must be odd)
void cpu_conv2d_same(const float* input_images, const float* kernel, 
                     float* output_images,
                     int batch_size, int height, int width, int kernel_size) {
    // Calculate kernel radius (how far it extends from center)
    // For k=5, radius=2, so kernel covers positions [-2, -1, 0, +1, +2]
    int radius = (kernel_size - 1) / 2;
    
    // Lambda function to convert 3D coordinates to 1D array index
    // Images are stored sequentially: [Image0, Image1, ..., ImageN]
    auto calculate_index = [height, width](int batch_idx, int row, int col) {
        return static_cast<size_t>(batch_idx) * height * width + 
               static_cast<size_t>(row) * width + 
               col;
    };
    
    // Process each image in the batch
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        
        // Process each row of the image
        for (int output_row = 0; output_row < height; ++output_row) {
            
            // Process each column of the image
            for (int output_col = 0; output_col < width; ++output_col) {
                
                // Accumulator for the convolution sum at this pixel
                float convolution_sum = 0.0f;
                
                // Slide the kernel over the input image
                for (int kernel_row = 0; kernel_row < kernel_size; ++kernel_row) {
                    for (int kernel_col = 0; kernel_col < kernel_size; ++kernel_col) {
                        
                        // Calculate corresponding input image coordinates
                        // Subtract radius to center the kernel
                        int input_row = output_row + kernel_row - radius;
                        int input_col = output_col + kernel_col - radius;
                        
                        // Check if input coordinates are within image bounds
                        // If outside bounds, treat as zero (zero padding for SAME convolution)
                        bool within_bounds = (input_row >= 0) && (input_row < height) &&
                                           (input_col >= 0) && (input_col < width);
                        
                        if (within_bounds) {
                            // Get input pixel value
                            float input_pixel = input_images[
                                calculate_index(batch_idx, input_row, input_col)
                            ];
                            
                            // Get kernel weight
                            float kernel_weight = kernel[kernel_row * kernel_size + kernel_col];
                            
                            // Multiply and accumulate
                            convolution_sum += input_pixel * kernel_weight;
                        }
                        // else: out of bounds, contributes 0 (implicit zero padding)
                    }
                }
                
                // Write the computed convolution result to output
                output_images[calculate_index(batch_idx, output_row, output_col)] = convolution_sum;
            }
        }
    }
}

// Utility function to compute the maximum absolute difference between two arrays
// Used to verify correctness by comparing GPU output against CPU reference
// Returns the L-infinity norm (maximum absolute difference)
//
// Parameters:
//   array_a: First array to compare
//   array_b: Second array to compare
//   num_elements: Number of elements in both arrays
//
// Returns:
//   Maximum absolute difference found between any pair of corresponding elements
float max_abs_diff(const float* array_a, const float* array_b, size_t num_elements) {
    float max_difference = 0.0f;
    
    for (size_t i = 0; i < num_elements; ++i) {
        float absolute_difference = std::fabs(array_a[i] - array_b[i]);
        max_difference = std::max(max_difference, absolute_difference);
    }
    
    return max_difference;
}