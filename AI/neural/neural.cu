/**
 * @file neural_cuda.cpp
 * @author Michael Gathara (michael@michaelgathara.com)
 * @brief 
 * @version 0.1
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cuda_runtime.h>

__device__ 
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__global__ 
void hiddenLayerKernel(double* g_input, double* g_hidden_weights, double* g_hidden_values, double hidden_bias, int input_size, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < hidden_size) {
        double value = 0.0;
        for (int j = 0; j < input_size; j++) {
            value += g_input[j] * g_hidden_weights[i * input_size + j];
        }
        value += hidden_bias;
        g_hidden_values[i] = sigmoid(value);
    }
}

__global__ 
void outputLayerKernel(double* g_hidden_values, double* g_output_weights, double* g_output_value, double output_bias, int hidden_size) {
    double value = 0.0;
    for (int i = 0; i < hidden_size; i++) {
        value += g_hidden_values[i] * g_output_weights[i];
    }
    value += output_bias;
    *g_output_value = sigmoid(value);
}

class NeuralNetwork {
private:
    std::vector<std::vector<double>> hidden_weights;
    std::vector<double> output_weights;
    double hidden_bias;
    double output_bias;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis;
    double* g_hidden_weights;
    double* g_output_weights;

public:
    NeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) : gen(std::random_device{}()), dis(-1.0, 1.0) {
        hidden_weights.resize(hidden_nodes, std::vector<double>(input_nodes));
        for (int i = 0; i < hidden_nodes; i++) {
            for (int j = 0; j < input_nodes; j++) {
                hidden_weights[i][j] = dis(gen);
            }
        }

        output_weights.resize(output_nodes);
        for (int i = 0; i < output_nodes; i++) {
            output_weights[i] = dis(gen);
        }

        hidden_bias = dis(gen);
        output_bias = dis(gen);

        cudaMalloc(&g_hidden_weights, hidden_nodes * input_nodes * sizeof(double));
        cudaMalloc(&g_output_weights, output_nodes * sizeof(double));

        cudaMemcpy(g_hidden_weights, hidden_weights.data(), hidden_nodes * input_nodes * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(g_output_weights, output_weights.data(), output_nodes * sizeof(double), cudaMemcpyHostToDevice);
    }


    std::vector<double> forward(const std::vector<double>& input) {
        double* g_input;
        double* g_hidden_values;
        double* g_output_value;

        cudaMalloc(&g_input, input.size() * sizeof(double));
        cudaMalloc(&g_hidden_values, hidden_weights.size() * sizeof(double));
        cudaMalloc(&g_output_value, sizeof(double));

        cudaMemcpy(g_input, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);

        hiddenLayerKernel<<<(hidden_weights.size() + 255) / 256, 256>>>(g_input, g_hidden_weights, g_hidden_values, hidden_bias, input.size(), hidden_weights.size());
        cudaDeviceSynchronize();

        outputLayerKernel<<<1, 1>>>(g_hidden_values, g_output_weights, g_output_value, output_bias, hidden_weights.size());
        cudaDeviceSynchronize();

        double output_value;
        cudaMemcpy(&output_value, g_output_value, sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(g_input);
        cudaFree(g_hidden_values);
        cudaFree(g_output_value);

        return {output_value};
    }

    ~NeuralNetwork() {
        cudaFree(g_hidden_weights);
        cudaFree(g_output_weights);
    }
};

int main() {
    NeuralNetwork nn(5, 10, 1);
    std::vector<double> input = {0.5, 0.3, 0.2, 0.2, 0.1};
    std::vector<double> output = nn.forward(input);

    std::cout << "Output: " << output[0] << std::endl;

    return 0;
}
