#include <cassert>
#include <cstdio>

#include "GL/glut.h"
#include "GL/freeglut_ext.h"

#include "main.h"
#include "glut_window.h"

/**
* Initializes the GLUT window and registers callbacks. The SimulationContext pointer is stored with the window using FreeGLUT's glutSetWindowData,
* allowing callbacks to access it without global variables.
*/
void initializeGlutWindow(int& argc, char* argv[], SimulationContext* ctx) {
    assert(ctx && ctx->params);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    int width = ctx->params->M;
    int height = ctx->params->N;
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Heat Transfer Simulation");
    glClearColor(0.0, 0.0, 0.0, 0);

    // store context pointer with the window (FreeGLUT API)
    glutSetWindowData(static_cast<void*>(ctx));

    // register plain function callbacks
    glutDisplayFunc(displayCallback);
    glutTimerFunc(0, timerCallback, 0);
}

/**
* Display callback for GLUT. Retrieves the SimulationContext pointer stored with the window and uses it to draw the current state of the simulation.
*/
void displayCallback() {
    // retrieve per-window context (FreeGLUT)
    SimulationContext* ctx = static_cast<SimulationContext*>(glutGetWindowData());
    if (!ctx) return;
    int width = ctx->params->M;
    int height = ctx->params->N;
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawPixels(width, height, GL_RGB, GL_UNSIGNED_BYTE, ctx->PixelBuffer);
    glutSwapBuffers();
}

/**
* Timer callback for GLUT. Retrieves the SimulationContext pointer stored with the window, advances the simulation by one step, and checks for convergence.
* If the simulation should continue, it registers itself again with glutTimerFunc to be called after 16ms (approximately 60 FPS).
* If the simulation has converged or reached the iteration limit, it prints a message and exits the GLUT main loop.
*/
void timerCallback(int value) {
    (void)value;
    SimulationContext* ctx = static_cast<SimulationContext*>(glutGetWindowData());
    if (!ctx) return;
    stepSimulation(ctx);
    // Continue only if not converged and iteration limit not reached
    bool notConverged = (ctx->sum > ctx->params->epsilon);
    bool underMaxIters = (ctx->iteration < ctx->params->max_iterations);

    if (notConverged && underMaxIters) {
        glutTimerFunc(16, timerCallback, 0);
    }
    else {
        std::printf("Simulation finished at iteration %d: sum=%f\n", ctx->iteration, ctx->sum);
        // print final temperature grid
        int M = ctx->params->M;
        int N = ctx->params->N;
        size_t rows = (size_t)M + 2;
        size_t cols = (size_t)N + 2;
        std::printf("Final temperature grid (in Celsius):\n");
        for (size_t i = 1; i <= M; i++) {
            for (size_t j = 1; j <= N; j++) {
                double temp_celsius = ctx->h_a[i * cols + j] - 273.15;
                std::printf("%6.2f ", temp_celsius);
            }
            std::printf("\n");
        }
        // Stop the GLUT main loop so cleanup in main() runs
        glutLeaveMainLoop();
    }
}