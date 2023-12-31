/**
 * @file add-cpu.cpp
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 */
#include <math.h>
#include <iostream>

/**
 * Adopted from: https://developer.nvidia.com/blog/even-easier-introduction-cuda/
 */
// function to add the elements of two arrays

void add(int n, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1 << 30;  // 1M elements
    std::cout << "N = " << N << std::endl;

    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on elements on the CPU
    add(N, x, y);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    delete [] x;
    delete [] y;

    return 0;
}