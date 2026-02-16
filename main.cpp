#include <cstdio>
#include <cstdlib>

#include "GL/glut.h"
#include "GL/freeglut_ext.h"

#include "kernel.h"
#include "main.h"
#include "glut_window.h"

/**
* Entry point of the program. Initializes the simulation parameters and context, sets up the OpenGL window, and starts the main loop.
* After the main loop exits, it frees any allocated memory and prints a message indicating that the simulation has ended.
*/
int main(int argc, char* argv[]) {
    SimulationParameters* simulationParameters = initializeSimulationParameters(argc, argv);
    SimulationContext* ctx = initializeSimulationContext(simulationParameters);
    std::printf("Starting the simulation.\n");

    initializeSimulation(ctx);
    initializeGlutWindow(argc, argv, ctx);
    glutMainLoop();

    freeDeviceMemory();
    freeHostMemory(ctx);
    std::printf("Simulation ended.\n");
    return 0;
}