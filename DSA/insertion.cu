/**
 * @file insertion.cu
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief Insertion sort using the GPU
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 */

#include <stdio.h>
#include <iostream>


// Can't use regular insertionSort, have to find a way to divide and combine GPU answer
__global__
void insertionSort(int *arr, int size) {

}



int main(void) {
    // Insertion sort with 500 Million numbers
    int N = 1 << 14;

    int *arr;
    cudaMallocManaged(&arr, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        arr[i] = N - i;
    }

    insertionSort<<<1, 1>>>(arr, N);

    cudaDeviceSynchronize();

    int amount_of_errors = 0;
    for (int i = 0; i < N; i++) {
        if (arr[i] == i + 1) {
            std::cout << "Error: " << i << " " << arr[i] << std::endl;
            amount_of_errors++;
        }
    }

    std::cout << "Amount of errors: " << amount_of_errors << std::endl;

    cudaFree(arr);

    return 0;
}