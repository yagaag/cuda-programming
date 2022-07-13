// CUDA Programming practice: Image convolution

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <Magick++.h>

using namespace Magick;

// 8 x 8 convolutional mask
#define MASK_DIM 8

// Allocate mask in constant memory
__constant__ int mask[8 * 8];

// 2D Convolution Kernel
// Input:
//  matrix: Input matrix
//  result: Convolution result
//  N:      Dimensions of the matrices
__global__ void convolution_2d(int *matrix, int *result, int N) {
  // Calculate the global thread positions
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Starting index for calculation
  int start_r = row;
  int start_c = col;

  // Temp value for accumulating the result
  int temp = 0;

  // Iterate over all the rows
  for (int i = 0; i < MASK_DIM; i++) {
    // Go over each column
    for (int j = 0; j < MASK_DIM; j++) {
      // Accumulate result
      temp += matrix[(start_r + i) * N + (start_c + j)] * mask[i * MASK_DIM + j];
    }
  }

  // Write back the result
  result[row * N + col] = temp;
}

// Initializing a random number matrix as convolution kernel
void init_matrix(int *m, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      m[n * i + j] = rand() % 100;
    }
  }
}

// Loading an image
void init_image(int *m, int n) {
  try {
    InitializeMagick(*argv);
    Image img("test.bmp");
    for (int i=0; i<n; i++) {
      for (int j=0; i<n; j++) {
        ColorRGB rgb(img.pixelColor(i, j));
        m[(n * i) + j] = rgb.red();
      }
    }
  }
  catch ( Magick::Exception & error) {
    cerr << "Caught Magick++ exception: " << error.what() << endl;
  }
}

int main() {
  // Dimensions of the matrix (2 ^ 10 x 2 ^ 10)
  int N = 1 << 10;
  // Byte size for the matrix
  size_t bytes_n = N * N * sizeof(int);

  // Allocate the matrix and load an image in it
  int *matrix = new int[N * N];
  int *result = new int[N * N];
  init_image(matrix, N);

  // Size of the mask
  size_t bytes_m = MASK_DIM * MASK_DIM * sizeof(int);

  // Allocate the mask and initialize it
  int *h_mask = new int[MASK_DIM * MASK_DIM];
  init_matrix(h_mask, MASK_DIM);

  // Allocate device memory
  int *d_matrix;
  int *d_result;
  cudaMalloc(&d_matrix, bytes_n);
  cudaMalloc(&d_result, bytes_n);

  // Copy data to the device
  cudaMemcpy(d_matrix, matrix, bytes_n, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(mask, h_mask, bytes_m);

  // Calculate grid dimensions
  int THREADS = 16;
  int BLOCKS = (N + THREADS - 1) / THREADS;

  // Dimension launch args
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  // Perform 2D Convolution
  convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, N);

  // Copy the result back to the CPU
  cudaMemcpy(result, d_result, bytes_n, cudaMemcpyDeviceToHost);

  std::cout << "Convolution performed successfully";

  // Free memory
  delete[] matrix;
  delete[] result;
  delete[] h_mask;

  cudaFree(d_matrix);
  cudaFree(d_result);

  return 0;
}
