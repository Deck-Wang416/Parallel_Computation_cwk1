// Updated cwk2.c with parallel mean calculation

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "cwk2_extra.h"

int main(int argc, char **argv) {
    int i;
    int rank, numProcs;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if ((numProcs & (numProcs - 1)) != 0 || numProcs > 256) {
        if (rank == 0) printf("ERROR: Launch with a power of 2 processes (<=256).\n");
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    float *globalData = NULL;
    int globalSize = 0;
    if (rank == 0) {
        globalData = readDataFromFile(&globalSize);
        if (globalData == NULL) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        printf("Rank 0: Read in data set with %d floats.\n", globalSize);
    }

    int localSize;
    MPI_Bcast(&globalSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    localSize = globalSize / numProcs;
    float *localData = (float*)malloc(localSize * sizeof(float));
    MPI_Scatter(globalData, localSize, MPI_FLOAT, localData, localSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();
    
    // Compute local sum
    float localSum = 0.0f;
    for (i = 0; i < localSize; i++) {
        localSum += localData[i];
    }
    
    // Reduce to get global sum and compute mean
    float globalSum = 0.0f;
    MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    float mean = 0.0f;
    if (rank == 0) {
        mean = globalSum / globalSize;
    }
    
    // Broadcast mean to all processes
    MPI_Bcast(&mean, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Output results on rank 0
    if (rank == 0) {
        printf("Parallel Mean Computed: %f\n", mean);
    }
    
    free(localData);
    if (rank == 0) free(globalData);
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
