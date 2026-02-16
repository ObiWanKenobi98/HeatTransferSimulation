#pragma once

#include "simulation.h"

/*
* Simulation lifecycle / helpers
*/
SimulationParameters* initializeSimulationParameters(int argc, char* argv[]);
SimulationContext* initializeSimulationContext(SimulationParameters* simulationParameters);

/*
*  Host memory initialization and cleanup, as well as stepping the simulation forward one iteration (kernel.cu)
*/
void initializeSimulation(SimulationContext* ctx);
void initializeHostMemory(SimulationContext* ctx);

void stepSimulation(SimulationContext* ctx);
void freeHostMemory(SimulationContext* ctx);