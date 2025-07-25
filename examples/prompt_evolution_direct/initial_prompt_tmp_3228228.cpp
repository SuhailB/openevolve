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
#pragma HLS DATAFLOW

  float local_A[16][256];
  float local_B[256][16];
  float local_C[16][256] = {0};

  for(int i = 0; i < 256; i += 16) {
    #pragma HLS LOOP_UNROLL factor=16
    for(int k = 0; k < 256; k++) {
      #pragma HLS PIPELINE II=1
      for(int j = 0; j < 16; j++) {
        local_A[j][k] = A[i+j][k];
        local_B[k][j] = B[k][i+j];
      }
    }

    for(int j = 0; j < 256; j++) {
      #pragma HLS PIPELINE II=1
      for(int k = 0; k < 256; k++) {
        for(int l = 0; l < 16; l++) {
          local_C[l][j] += local_A[l][k] * local_B[k][j];
        }
      }
    }

    for(int j = 0; j < 256; j++) {
      #pragma HLS PIPELINE II=1
      for(int l = 0; l < 16; l++) {
        C[i+l][j] = local_C[l][j];
      }
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
      C[i][j] = static_cast<float>(rand()) / RAND_MAX; // Initialize C with random values too
      C_golden[i][j] = C[i][j]; // Copy initial C values to C_golden
    }
  }

  // Call the optimized kernel
  gemmKernel(A, B, C);

  // Call the golden function to compute the expected result and print first 3 elements
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