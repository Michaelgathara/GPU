/**
 * @file insertion.cpp
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief Insertion sort using the CPU
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>
#include <chrono>

std::vector<int> insertionSort(std::vector<int> arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= 0 && arr[j] > key) {
            arr[j+1] = arr[j];
            j--;
        }

        arr[j+1] = key;
    }

    return arr;
}

int main(void) {
    int N = 1 << 29;
    std::vector<int> arr(N);

    for (int i = 0; i < N; i++) {
        arr[i] = N - i;
    }

    auto start = std::chrono::steady_clock::now();
    std::vector<int> sorted = insertionSort(arr);
    auto end = std::chrono::steady_clock::now();

    for (int i = 0; i < N; i++) {
        if (sorted[i] != i + 1) {
            std::cout << "Error: " << i << " " << sorted[i] << std::endl;
            return 1;
        }
    }

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
}