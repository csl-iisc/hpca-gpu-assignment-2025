#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <getopt.h>

#include "timers.h"
#include "../include/conv_kernels.cuh"

// Forward declarations
void cpu_conv2d_same(const float* input, const float* kernel, float* output,
                     int batch_size, int height, int width, int kernel_size);
float max_abs_diff(const float* array_a, const float* array_b, size_t num_elements);

// Utility functions
bool load_images_bin(const std::string& path, std::vector<float>& output, 
                     int& batch_size, int& height, int& width);
void gen_random_images(std::vector<float>& output, int batch_size, int height, 
                       int width, unsigned seed = 42);

static void print_usage(const char* program_name) {
    std::printf("Usage: %s [options]\n", program_name);
    std::puts("  --n=N             batch size (default 8)\n"
              "  --h=H             image height (default 1024)\n"
              "  --w=W             image width (default 1024)\n"
              "  --k=K             kernel size - must be odd (default 5)\n"
              "  --impl=NAME       implementation: baseline|variant1|variant2|variant3|variant4|variant5|bonus\n"
              "  --iters=I         number of timing iterations (default 5)\n"
              "  --verify          compare GPU results with CPU reference\n"
              "  --images=PATH     load images from binary file (otherwise generate random)\n"
              "  --batch=B         batch size for streams (default 16)\n"
              "  --streams=S       number of CUDA streams (default 2)\n");
}

int main(int argc, char** argv) {
    // Default parameters
    int batch_size = 8;
    int height = 1024;
    int width = 1024;
    int kernel_size = 5;
    int num_iterations = 5;
    bool run_verification = false;
    std::string implementation = "naive";
    std::string image_path = "";

    // Command line option definitions
    const option long_options[] = {
        {"n",       required_argument, nullptr, 'n'},
        {"h",       required_argument, nullptr, 'h'},
        {"w",       required_argument, nullptr, 'w'},
        {"k",       required_argument, nullptr, 'k'},
        {"impl",    required_argument, nullptr, 'i'},
        {"iters",   required_argument, nullptr, 't'},
        {"verify",  no_argument,       nullptr, 'v'},
        {"images",  required_argument, nullptr, 'f'},
        {0, 0, 0, 0}
    };

    // Parse command line arguments
    while (true) {
        int option_char = getopt_long(argc, argv, "", long_options, nullptr);
        if (option_char == -1) {
            break;
        }
        
        switch (option_char) {
            case 'n': batch_size = std::atoi(optarg); break;
            case 'h': height = std::atoi(optarg); break;
            case 'w': width = std::atoi(optarg); break;
            case 'k': kernel_size = std::atoi(optarg); break;
            case 'i': implementation = optarg; break;
            case 't': num_iterations = std::atoi(optarg); break;
            case 'v': run_verification = true; break;
            case 'f': image_path = optarg; break;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    // Validate kernel size
    if (kernel_size % 2 == 0 || kernel_size <= 0) {
        std::fprintf(stderr, "Error: kernel size must be odd and positive\n");
        return 1;
    }

    // ===== Load or generate input images =====
    std::vector<float> host_images;
    int loaded_batch_size = batch_size;
    int loaded_height = height;
    int loaded_width = width;
    
    if (!image_path.empty() && load_images_bin(image_path, host_images, 
                                                loaded_batch_size, loaded_height, loaded_width)) {
        batch_size = loaded_batch_size;
        height = loaded_height;
        width = loaded_width;
        std::printf("Loaded images: batch=%d height=%d width=%d\n", 
                    batch_size, height, width);
    } else {
        gen_random_images(host_images, batch_size, height, width, 1234);
        std::printf("Generated random images: batch=%d height=%d width=%d\n", 
                    batch_size, height, width);
    }

    // ===== Initialize kernel (convolution filter) =====
    std::vector<float> host_kernel(kernel_size * kernel_size);
    float kernel_value = 1.0f / (kernel_size * kernel_size);
    for (int i = 0; i < kernel_size * kernel_size; ++i) {
        host_kernel[i] = kernel_value;
    }

    // ===== Allocate host output buffers =====
    std::vector<float> host_output(batch_size * height * width);
    std::vector<float> host_reference_output;

    // ===== Allocate device memory =====
    float* device_input = nullptr;
    float* device_kernel = nullptr;
    float* device_output = nullptr;
    
    size_t image_bytes = static_cast<size_t>(batch_size) * height * width * sizeof(float);
    size_t output_bytes = image_bytes;
    size_t kernel_bytes = static_cast<size_t>(kernel_size) * kernel_size * sizeof(float);
    
    CK(cudaMalloc(&device_input, image_bytes));
    CK(cudaMalloc(&device_kernel, kernel_bytes));
    CK(cudaMalloc(&device_output, output_bytes));
    
    // Copy input data to device
    CK(cudaMemcpy(device_input, host_images.data(), image_bytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(device_kernel, host_kernel.data(), kernel_bytes, cudaMemcpyHostToDevice));

    // ===== Generate CPU reference if verification requested =====
    if (run_verification) {
        std::printf("Computing CPU reference (this may take a while)...\n");
        host_reference_output.resize(static_cast<size_t>(batch_size) * height * width);
        cpu_conv2d_same(host_images.data(), host_kernel.data(), host_reference_output.data(), 
                        batch_size, height, width, kernel_size);
    }

    // ===== Helper function to select and call the right kernel =====
    auto call_kernel = [&](const std::string& impl_name, cudaStream_t stream = 0) {
        if (impl_name == "naive" || impl_name == "baseline") {
            conv2d_baseline(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant1") {
            conv2d_variant1(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant2") {
            conv2d_variant2(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant3") {
            conv2d_variant3(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else if (impl_name == "variant4") {
            conv2d_variant4(device_input, device_kernel, device_output, 
                            batch_size, height, width, kernel_size, stream);
        } else {
            std::fprintf(stderr, "Unknown implementation: %s\n", impl_name.c_str());
            std::exit(2);
        }
    };

    // ===== Run benchmark with proper timing =====
    std::printf("\n========== BENCHMARK: %s ==========\n", implementation.c_str());
    std::printf("Configuration: N=%d, H=%d, W=%d, k=%d, iters=%d\n", 
                batch_size, height, width, kernel_size, num_iterations);
    std::printf("Total pixels per iteration: %.2f MPix\n", 
                static_cast<double>(batch_size) * height * width / 1e6);
    
    // Warmup runs (2-3 iterations to initialize GPU)
    std::printf("Running warmup...\n");
    for (int i = 0; i < 3; ++i) {
        call_kernel(implementation);
    }
    CK(cudaDeviceSynchronize());
    
    // Timed benchmark runs
    std::printf("Running timed iterations...\n");
    CudaEventTimer timer;
    std::vector<float> iteration_times;
    
    for (int iteration = 0; iteration < num_iterations; ++iteration) {
        timer.record_start();
        call_kernel(implementation);
        float elapsed_ms = timer.record_stop_and_elapsed_ms();
        iteration_times.push_back(elapsed_ms);
        
        std::printf("  Iteration %d: %.4f ms\n", iteration + 1, elapsed_ms);
    }
    
    // Calculate statistics
    float total_time = 0.0f;
    float min_time = iteration_times[0];
    float max_time = iteration_times[0];
    
    for (float time : iteration_times) {
        total_time += time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    float average_time = total_time / num_iterations;
    double megapixels = static_cast<double>(batch_size) * height * width / 1e6;
    double throughput = megapixels / (average_time / 1000.0);
    
    // Print results
    std::printf("\n========== RESULTS ==========\n");
    std::printf("IMPLEMENTATION: %s\n", implementation.c_str());
    std::printf("Average time:   %.4f ms\n", average_time);
    std::printf("Min time:       %.4f ms\n", min_time);
    std::printf("Max time:       %.4f ms\n", max_time);
    std::printf("Throughput:     %.2f MPix/s\n", throughput);
    std::printf("Ops per pixel:  %d (kernel size %dx%d)\n", 
                kernel_size * kernel_size, kernel_size, kernel_size);

    // Verify correctness against CPU reference
    if (run_verification) {
        std::printf("\n========== VERIFICATION ==========\n");
        CK(cudaMemcpy(host_output.data(), device_output, output_bytes, 
                      cudaMemcpyDeviceToHost));
        
        float max_difference = max_abs_diff(host_output.data(), 
                                            host_reference_output.data(), 
                                            static_cast<size_t>(batch_size) * height * width);
        
        std::printf("Maximum absolute difference vs CPU: %.8f\n", max_difference);
        
        if (max_difference < 1e-4) {
            std::printf("✓ PASS: Results match CPU reference\n");
        } else {
            std::printf("✗ FAIL: Results differ from CPU reference\n");
        }
    }

    // Cleanup
    cudaFree(device_input);
    cudaFree(device_kernel);
    cudaFree(device_output);
    
    std::printf("\n");
    return 0;
}