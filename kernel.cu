#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include "kernel.h"

/*
* Clamp function to ensure RGB values are within the valid range of 0-255. This is a common utility function used in color calculations.
*/
__device__ unsigned char clamp(int value, unsigned char min, unsigned char max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

/*
* Convert a temperature in Kelvin to an RGB color. This function uses a common approximation for converting color temperature to RGB values.
* The input is a double representing the temperature in Kelvin, and the output is an RGBtemp struct containing the corresponding RGB values.
* The function handles different ranges of temperature to calculate the appropriate red, green, and blue components, and uses the clamp function to ensure the resulting RGB values are valid.
*/
__device__ RGBtemp kelvinToRGB(double kelvin) {
    RGBtemp result;
    int blue = 0, red = 0, green = 0;
    double temp = kelvin / 100.0;
    if (temp <= 66.0) {
        red = 255;
        green = (int)(99.4708025861 * log(temp) - 161.1195681661);
        if (temp <= 19.0) {
            blue = 0;
        }
        else {
            blue = (int)(138.5177312231 * log(temp - 10.0) - 305.0447927307);
        }
    }
    else {
        red = (int)(329.698727446 * pow(temp - 60.0, -0.1332047592));
        green = (int)(288.1221695283 * pow(temp - 60.0, -0.0755148492));
        blue = 255;
    }
    result.B = clamp(blue, 0, 255);
    result.G = clamp(green, 0, 255);
    result.R = clamp(red, 0, 255);
    return result;
}

/*
* computeTemp kernel computes the new temperature values based on the average of the current cell and its 8 neighbors.
* It uses flat indexing to access the 2D grid stored in a 1D array.
* The kernel checks if the current thread corresponds to a valid cell (not on the border) and then calculates the average temperature for that cell, storing the result in the output array.
*/
__global__ void computeTemp(double* pa, double* pb, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int rows = M + 2;
    int cols = N + 2;

    if (i > 0 && j > 0 && i < M + 1 && j < N + 1) {
        double sum = 0.0;
        for (int k = i - 1; k <= i + 1; ++k) {
            for (int l = j - 1; l <= j + 1; ++l) {
                sum += pa[k * cols + l];
            }
        }
        pb[i * cols + j] = sum / 9.0;
    }
}

/*
* computeDiff kernel calculates the absolute difference between the new temperature values (pb) and the old temperature values (pa) for each cell.
*/
__global__ void computeDiff(double* pa, double* pb, double* pdiff, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int rows = M + 2;
    int cols = N + 2;

    if (i < rows && j < cols) {
        size_t idx = (size_t)i * cols + (size_t)j;
        pdiff[idx] = fabs(pb[idx] - pa[idx]);
        pa[idx] = pb[idx];
    }
}

/*
* computeRGBfromKelvin kernel converts the temperature values in pa to RGB colors using the kelvinToRGB function.
*/
__global__ void computeRGBfromKelvin(double* pa, RGBtemp* prgb, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int rows = M + 2;
    int cols = N + 2;

    if (i < rows && j < cols) {
        size_t idx = (size_t)i * cols + (size_t)j;
        prgb[idx] = kelvinToRGB(pa[idx]);
    }
}

/* Device-side state (internal to kernel.cu) */
static double* d_pa = nullptr;
static double* d_pb = nullptr;
static double* d_pdiff = nullptr;
static RGBtemp* d_prgb = nullptr;
static dim3 threadsPerBlock;
static dim3 numBlocks;

/*
* initializeDeviceMemory function allocates memory on the GPU for the temperature arrays and the RGB array, and copies the initial data from the host to the device.
*/
void initializeDeviceMemory(double* h_a, double* h_b, double* h_diff, RGBtemp* h_RGB, size_t cells, int M, int N) {
    cudaMalloc((void**)&d_pa, cells * sizeof(double));
    cudaMalloc((void**)&d_pb, cells * sizeof(double));
    cudaMalloc((void**)&d_pdiff, cells * sizeof(double));
    cudaMalloc((void**)&d_prgb, cells * sizeof(RGBtemp));

    cudaMemcpy(d_pa, h_a, cells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pb, h_b, cells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pdiff, h_diff, cells * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prgb, h_RGB, cells * sizeof(RGBtemp), cudaMemcpyHostToDevice);

    threadsPerBlock = dim3(16, 16);
    numBlocks = dim3((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
}

/*
* freeDeviceMemory function frees the allocated memory on the GPU to prevent memory leaks.
*/
void freeDeviceMemory() {
    cudaFree(d_pa);
    cudaFree(d_pb);
    cudaFree(d_pdiff);
    cudaFree(d_prgb);
}

/*
* runGpuStep function executes one iteration of the GPU computation. It launches the kernels to compute the new temperature values, the differences, and the RGB colors, and then copies the results back to the host.
*/
double runGpuStep(double* h_a, double* h_b, double* h_diff, RGBtemp* h_RGB, GLubyte* PixelBuffer, size_t rows, size_t cols, size_t width, size_t height, int iteration) {
    size_t cells = rows * cols;
    int M = (int)width;
    int N = (int)height;

    // Launch kernels - pass M and N so kernels use correct grid dimensions
    computeTemp << <numBlocks, threadsPerBlock >> > (d_pa, d_pb, M, N);
    cudaDeviceSynchronize();

    computeDiff << <numBlocks, threadsPerBlock >> > (d_pa, d_pb, d_pdiff, M, N);
    cudaDeviceSynchronize();

    computeRGBfromKelvin << <numBlocks, threadsPerBlock >> > (d_pa, d_prgb, M, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_diff, d_pdiff, cells * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a, d_pa, cells * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_RGB, d_prgb, cells * sizeof(RGBtemp), cudaMemcpyDeviceToHost);

    // Compute convergence sum on host (could also be done on device)
    double sum = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum += h_diff[i * cols + j];
        }
    }

    // Fill PixelBuffer with returned RGB data (host-side)
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            size_t src_idx = i * cols + j;
            size_t buf_idx = (i * width + j) * 3;
            PixelBuffer[buf_idx + 0] = h_RGB[src_idx].R;
            PixelBuffer[buf_idx + 1] = h_RGB[src_idx].G;
            PixelBuffer[buf_idx + 2] = h_RGB[src_idx].B;
        }
    }

    return sum;
}