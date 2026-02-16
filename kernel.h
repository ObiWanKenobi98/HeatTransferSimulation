#pragma once

#include <cstddef>
#include <GL/glut.h>
#include "rgb_temperature.h"

void initializeDeviceMemory(double* h_a, double* h_b, double* h_diff, RGBtemp* h_RGB, size_t cells, int M, int N);

void freeDeviceMemory();

double runGpuStep(double* h_a, double* h_b, double* h_diff, RGBtemp* h_RGB, GLubyte* PixelBuffer, size_t rows, size_t cols, size_t width, size_t height, int iteration);