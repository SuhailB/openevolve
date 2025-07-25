#include <iostream>
#include <random>
#include <cmath>

// Kernel-start
void gemmKernel(
  float A[256][256],
  float B[256][256],
  float C[256][256]
){
#pragma HLS ARRAY_PARTITION variable=A complete dim=2
#pragma HLS ARRAY_PARTITION variable=B complete dim=1
#pragma HLS ARRAY_PARTITION variable=C complete dim=2

  for(int i = 0; i < 256; ++i) {
    for(int j = 0; j < 256; ++j) {
      float sum = 0.0f;
      for(int k = 0; k < 256; ++k) {
#pragma HLS PIPELINE II=1
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
}
// Kernel-end

// Testbench-start
void gemmKernelGolden(
  float A[256][256],
  float B[256][256],
  float C[256][256]
){
  for(int i = 0; i < 256; ++i) {
    for(int j = 0; j < 256; ++j) {
      for(int k = 0; k < 256; ++k) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

int main() {
  // Initialize matrices A, B, and C
  float A[256][256] = {0};
  float B[256][256] = {0};
  float C[256][256] = {0};
  float C_golden[256][256] = {0};

  // Fill matrices A and B with random values
  for(int i = 0; i < 256; ++i) {
    for(int j = 0; j < 256; ++j) {
      A[i][j] = static_cast<float>(rand()) / RAND_MAX;
      B[i][j] = static_cast<float>(rand()) / RAND_MAX;
      C[i][j] = 0.0f; // Initialize C with zeros
      C_golden[i][j] = C[i][j]; // Copy initial C values to C_golden
    }
  }

  // Call the kernel function
  gemmKernel(A, B, C);

  // Call the golden function to compute the expected result
  gemmKernelGolden(A, B, C_golden);

  // Verify the result
  bool success = true;
  for(int i = 0; i < 256; ++i) {
    for(int j = 0; j < 256; ++j) {
      if (std::abs(C[i][j] - C_golden[i][j]) > 1e-5) {
        success = false;
        std::cout << "Mismatch at (" << i << ", " << j << "): "
                  << "C[" << i << "][" << j << "] = " << C[i][j]
                  << ", C_golden[" << i << "][" << j << "] = " << C_golden[i][j] << std::endl;
        break;
      }
    }
    if (!success) break;
  }
  if (success) {
    std::cout << "Test passed!" << std::endl;
  } else {
    std::cout << "Test failed!" << std::endl;
  }
  return success ? 0 : 1;
}
// Testbench-end