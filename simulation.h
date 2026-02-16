#pragma once

#include "rgb_temperature.h"
#include <cstddef>
#include "GL/glut.h"

/*
* Simulation parameters parsed from command line arguments
*/
typedef struct {
    int M, N;
    double epsilon;
    int x0, y0;
    double source_temperature_celsius;
    double boundary_temperature_celsius;
    int max_iterations;
} SimulationParameters;

/*
* Per-window simulation context stored with FreeGLUT window data
*/
typedef struct {
    SimulationParameters* params;
    double* h_a;
    double* h_b;
    double* h_diff;
    RGBtemp* h_RGB;
    GLubyte* PixelBuffer;
    double sum;
    int iteration;
} SimulationContext;