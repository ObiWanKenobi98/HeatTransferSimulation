#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "GL/glut.h"

#include "kernel.h"
#include "main.h"

/**
* Parses command line arguments to initialize the SimulationParameters struct. Validates input and exits with an error message if any parameters are invalid.
*/
SimulationParameters* initializeSimulationParameters(int argc, char* argv[]) {
    if (argc != 9) {
        std::printf("Usage: %s <M> number_of_columns <N> number_of_rows <EPS> epsilon <x0> source_x <y0> source_y <initial_temp> source_temperature_celsius <boundary_temp> boundary_temperature_celsius <max_iterations> max_iterations\n", argv[0]);
        std::exit(EXIT_FAILURE);
    }
    int M = std::atoi(argv[1]);
    if (M <= 0) {
        std::printf("Invalid M value. Must be greater than 0.\n");
        std::exit(EXIT_FAILURE);
    }
    int N = std::atoi(argv[2]);
    if (N <= 0) {
        std::printf("Invalid N value. Must be greater than 0.\n");
        std::exit(EXIT_FAILURE);
    }
    double epsilon = std::atof(argv[3]);
    if (epsilon <= 0) {
        std::printf("Invalid EPS value. Must be greater than 0.\n");
        std::exit(EXIT_FAILURE);
    }
    int x0 = std::atoi(argv[4]);
    if (x0 < 1 || x0 > M) {
        std::printf("Invalid x0 value. Must be between 1 and M.\n");
        std::exit(EXIT_FAILURE);
    }
    int y0 = std::atoi(argv[5]);
    if (y0 < 1 || y0 > N) {
        std::printf("Invalid y0 value. Must be between 1 and N.\n");
        std::exit(EXIT_FAILURE);
    }
    double source_temperature_celsius = std::atof(argv[6]);
    double boundary_temperature_celsius = std::atof(argv[7]);
    int max_iterations = std::atoi(argv[8]);
    if (max_iterations <= 0) {
        std::printf("Invalid max_iterations value. Must be greater than 0.\n");
        std::exit(EXIT_FAILURE);
    }

    SimulationParameters* simulationParameters = new SimulationParameters();
    simulationParameters->M = M;
    simulationParameters->N = N;
    simulationParameters->epsilon = epsilon;
    simulationParameters->x0 = x0;
    simulationParameters->y0 = y0;
    simulationParameters->source_temperature_celsius = source_temperature_celsius;
    simulationParameters->boundary_temperature_celsius = boundary_temperature_celsius;
    simulationParameters->max_iterations = max_iterations;
    return simulationParameters;
}

/**
* Initializes the SimulationContext struct with the given SimulationParameters. Host memory pointers are initialized to nullptr and will be allocated in initializeSimulation().
*/
SimulationContext* initializeSimulationContext(SimulationParameters* simulationParameters)
{
    SimulationContext* simulationContext = new SimulationContext();
    simulationContext->params = simulationParameters;
    simulationContext->h_a = simulationContext->h_b = simulationContext->h_diff = nullptr;
    simulationContext->h_RGB = nullptr;
    simulationContext->PixelBuffer = nullptr;
    simulationContext->sum = 0.0;
    simulationContext->iteration = 0;
    return simulationContext;
}

/**
* Initializes host memory for the simulation, including the temperature grids and RGB buffer. Also initializes device memory and copies initial data to the GPU.
*/
void initializeSimulation(SimulationContext* ctx) {
    assert(ctx && ctx->params);
    int M = ctx->params->M;
    int N = ctx->params->N;
    size_t rows = (size_t)M + 2;
    size_t cols = (size_t)N + 2;
    size_t cells = rows * cols;

    initializeHostMemory(ctx);
    initializeDeviceMemory(ctx->h_a, ctx->h_b, ctx->h_diff, ctx->h_RGB, cells, M, N);

    ctx->PixelBuffer = new GLubyte[M * N * 3];
    ctx->sum = ctx->params->epsilon + 1.0;
    ctx->iteration = 0;
}

/**
* Allocates host memory for the temperature grids (h_a, h_b), the difference grid (h_diff), and the RGB buffer (h_RGB). Initializes the temperature grids with the boundary temperature, except for the source cell which is initialized with the source temperature. The difference grid is initialized to 0 and the RGB buffer is initialized to black.
*/
void initializeHostMemory(SimulationContext* ctx) {
    assert(ctx && ctx->params);
    int M = ctx->params->M;
    int N = ctx->params->N;
    int rows = M + 2;
    int cols = N + 2;
    int cells = rows * cols;

    ctx->h_a = new double[cells];
    ctx->h_b = new double[cells];
    ctx->h_diff = new double[cells];
    ctx->h_RGB = new RGBtemp[cells];

    double ts = ctx->params->source_temperature_celsius + 273.15;
    double tr = ctx->params->boundary_temperature_celsius + 273.15;

    int x0 = ctx->params->x0;
    int y0 = ctx->params->y0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            ctx->h_a[idx] = tr;
            ctx->h_b[idx] = tr;
            ctx->h_diff[idx] = 0.0;
            ctx->h_RGB[idx].R = 0;
            ctx->h_RGB[idx].G = 0;
            ctx->h_RGB[idx].B = 0;
        }
    }

    ctx->h_a[(size_t)x0 * cols + (size_t)y0] = ts;
    ctx->h_b[(size_t)x0 * cols + (size_t)y0] = ts;
}

/**
* Steps the simulation forward one iteration by calling the GPU kernel to compute the new temperature values and the sum of differences.
* If the sum of differences is less than or equal to epsilon, the simulation is considered converged and the function returns without updating the simulation state.
* Otherwise, it updates the simulation state and requests a redraw of the window.
*/
void stepSimulation(SimulationContext* ctx) {
    assert(ctx && ctx->params);
    std::printf("Iteration %d: sum of diffs = %f\n", ctx->iteration, ctx->sum);
    double epsilon = ctx->params->epsilon;
    if (ctx->sum <= epsilon) {
        return;
    }

    int M = ctx->params->M;
    int N = ctx->params->N;
    size_t rows = (size_t)M + 2;
    size_t cols = (size_t)N + 2;

    ctx->sum = runGpuStep(ctx->h_a, ctx->h_b, ctx->h_diff, ctx->h_RGB, ctx->PixelBuffer, rows, cols, M, N, ctx->iteration);

    glutPostRedisplay();
    ctx->iteration++;
}

/**
* Frees host memory allocated for the simulation, including the temperature grids, difference grid, RGB buffer, and pixel buffer. Also deletes the SimulationParameters and SimulationContext structs.
*/
void freeHostMemory(SimulationContext* ctx) {
    if (!ctx) return;
    delete[] ctx->h_a;
    ctx->h_a = nullptr;
    delete[] ctx->h_b;
    ctx->h_b = nullptr;
    delete[] ctx->h_diff;
    ctx->h_diff = nullptr;
    delete[] ctx->h_RGB;
    ctx->h_RGB = nullptr;
    delete[] ctx->PixelBuffer;
    ctx->PixelBuffer = nullptr;
    delete ctx->params;
    ctx->params = nullptr;
    delete ctx;
}