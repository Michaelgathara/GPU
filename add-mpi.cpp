#include <iostream>
#include <cmath>
#include <mpi.h>

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::cout << "World Size: " << world_size << ", World Rank: " << world_rank << std::endl;

    int N = 1000000000; // 1B elements -> N has to be divisible by world_size
    int local_N = N / world_size; // Elements per process

    std::cout << "N = " << N << ", local_N = " << local_N << std::endl;

    float *x = new float[local_N];
    float *y = new float[local_N];

    // Initialize x and y arrays on each process
    for (int i = 0; i < local_N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run add function locally on each process
    add(local_N, x, y);

    // Collect results on root process
    float *result = NULL;
    if (world_rank == 0) {
        result = new float[N];
    }

    MPI_Gather(y, local_N, MPI_FLOAT, result, local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Check for errors on root process
    if (world_rank == 0) {
        float maxError = 0.0f;
        for (int i = 0; i < N; i++) {
            maxError = fmax(maxError, fabs(result[i] - 3.0f));
        }
        std::cout << "Max error: " << maxError << std::endl;
        delete[] result;
    }

    // Free memory
    delete[] x;
    delete[] y;

    MPI_Finalize();

    return 0;
}
