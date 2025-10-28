#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
#include <string>
#include <sys/stat.h>

// NVTX helpers for profiling (optional)
// If NVTX is not available, these become no-ops
#if __has_include(<nvtx3/nvToolsExt.h>)
  #include <nvtx3/nvToolsExt.h>
  
  struct NvtxRange { 
      NvtxRange(const char* name) { 
          nvtxRangePushA(name);
      } 
      
      ~NvtxRange() { 
          nvtxRangePop(); 
      } 
  };
  
  inline void nvtx_mark(const char* name) { 
      nvtxMarkA(name);
  }
#else
  struct NvtxRange { 
      NvtxRange(const char*) {} 
      ~NvtxRange() {} 
  };
  
  inline void nvtx_mark(const char*) {}
#endif

// Check if a file exists at the given path
bool file_exists(const std::string& path) { 
    struct stat file_status;
    int result = ::stat(path.c_str(), &file_status);
    return (result == 0);
}

// Load images from a binary file with format: [int32 N, H, W][N*H*W floats]
// Returns true on success, false on failure
// On success, populates out vector and sets N, H, W dimensions
bool load_images_bin(const std::string& path, std::vector<float>& out, 
                     int& N, int& H, int& W) {
    FILE* file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }
    
    // Read image dimensions
    int32_t num_images, height, width;
    size_t items_read = 0;
    
    items_read += fread(&num_images, sizeof(int32_t), 1, file);
    items_read += fread(&height, sizeof(int32_t), 1, file);
    items_read += fread(&width, sizeof(int32_t), 1, file);
    
    if (items_read != 3) {
        fclose(file);
        return false;
    }
    
    // Read image data
    size_t total_pixels = static_cast<size_t>(num_images) * height * width;
    out.resize(total_pixels);
    
    size_t pixels_read = fread(out.data(), sizeof(float), total_pixels, file);
    fclose(file);
    
    if (pixels_read != total_pixels) {
        return false;
    }
    
    // Set output dimensions
    N = num_images;
    H = height;
    W = width;
    
    return true;
}

// Generate random images if file is missing
// Creates N images of size H x W with random values between 0 and 1
void gen_random_images(std::vector<float>& out, int N, int H, int W, 
                       unsigned seed = 42) {
    std::mt19937 random_generator(seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    
    size_t total_pixels = static_cast<size_t>(N) * H * W;
    out.resize(total_pixels);
    
    for (float& pixel : out) {
        pixel = distribution(random_generator);
    }
}