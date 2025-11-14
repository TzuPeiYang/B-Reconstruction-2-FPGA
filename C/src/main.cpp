#include "model_weights.h"
#include <stdio.h>
#include <stddef.h>
#include <iostream>

#include "topk_test.h"

int main(int argc, char** argv) {
    
    // test inputs for model inference
    const int B = 1;
    const int D = 5;
    const int N = 10;
    const float pf_points[B][D][N] = {{{ -2.9732,  0.6687, -1.5152, -0.2692,  0.0974, -0.5925,  0.0400,
          -0.5660,  0.6632, -0.7361},
         {-1.4453, -0.8672, -0.2501, -0.6764, -0.2650, -1.1772, -0.2677,
           0.8057, -1.4662,  0.3231},
         {-1.0516,  1.2329,  0.3950,  1.1776, -0.1105,  0.7318, -0.3670,
           0.9206, -1.2518,  0.5360},
         { 2.0380,  0.3146, -0.0950, -0.8317, -0.3581, -0.5994, -1.1146,
           0.9128,  0.2906,  0.8735},
         {-2.2197,  0.5552,  1.1879,  2.6903,  0.5599,  0.9173,  0.4471,
           1.1378, -0.5362,  1.2142 }}};

    const int top_k = 3;
    int64_t indices[B][top_k][N];
    float values[B][top_k][N];

    entry(pf_points, values, indices);

    std::cout << "[";
    for (int i = 0; i < B; i++) {
        std::cout << "[";
        for (int j = 0; j < top_k; j++) {
            std::cout << "[";
            for (int k = 0; k < N; k++) {
                std::cout << values[i][j][k] << ", ";
            }
            std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    std::cout << "]" << std::endl;

    // test network
    
    
    /*
    float x[24] = {0.1, 0.5, 0.3,
                   0.4, 0.2, 0.9,
                   0.6, 0.8, 0.7,
                   0.15, 0.55, 0.35,
                   0.45, 0.25, 0.95,
                   0.65, 0.85, 0.75,
                   0.12, 0.52, 0.32,
                   0.42, 0.22, 0.92};
    int data_dim[3] = {2, 1, 12};
    float k[1] = {3};
    unsigned n_dim = 3;
    int axis = -1;
    int ax = axis < 0 ? axis + n_dim : axis;

	std::cout << "\t" << "// TopK with largest=1, sorted=1, other options not currently supported " << std::endl;
    
    std::cout << "\t" << "float min_heap[" << k[0] << "];" << std::endl;
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax) {
            std::cout << "\t" << "for (unsigned i" << i << " = 0; i" << i << " < " << data_dim[i] << "; i" << i << "++) {" << std::endl;
        }
    }
    std::cout << "\t\t" << "for (unsigned k = 0; k < " << k[0] << "; k++) {" << std::endl;
    std::cout << "\t\t" << "\tValues[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "k * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i << "] = -1e30f;" << std::endl;
        }
        else {
            std::cout << "k] = -1e30f;" << std::endl;
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "\t\t" << "\tIndices[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "k * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i << "] = -1;" << std::endl;
        }
        else {
            std::cout << "k] = -1;" << std::endl;
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "\t\t" << "}" << std::endl;
    
    std::cout << "\t\t" << "for (int k = 0; k < " << data_dim[ax] <<"; k++){" << std::endl;
    std::cout << "\t\t\t" << "for (int d = 0; d < " << k[0] << "; d++){" << std::endl;
    std::cout << "\t\t\t\t" << "if (X[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "k * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "k";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else {
                std::cout << data_dim[j] << " + ";
            }
        }
    }
    std::cout << "] > Values[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "d * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "d";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "]) {" << std::endl;
    std::cout << "\t\t\t\t\t" << "for (int s = " << k[0] - 1 << "; s > d; s--){" << std::endl;

    std::cout << "\t\t\t\t\t\t" << "Values[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "s * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "s";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "] = Values[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "(s - 1) * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "(s - 1)";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "];" << std::endl;

    std::cout << "\t\t\t\t\t\t" << "Indices[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "s * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "s";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "] = Indices[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "(s - 1) * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "(s - 1)";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "];" << std::endl;
    std::cout << "\t\t\t\t\t" << "}" << std::endl;

    std::cout << "\t\t\t\t\t" << "Values[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "d * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "d";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "] = X[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "k * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "k";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else {
                std::cout << data_dim[j] << " + ";
            }
        }
    }
    std::cout << "];" <<  std::endl;

    std::cout << "\t\t\t\t\t" << "Indices[";
    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax && i != n_dim - 1) {
            std::cout << "i" << i << " * ";
        }
        else if (i == ax && i != n_dim - 1) {
            std::cout << "d * ";
        }
        else if (i != ax && i == n_dim - 1) {
            std::cout << "i" << i;
        }
        else {
            std::cout << "d";
        }
        for (unsigned j = i + 1; j < n_dim; j++){
            if (j != ax && j != n_dim - 1) {
                std::cout << data_dim[j] << " * ";
            }
            else if (j == ax && j != n_dim - 1) {
                std::cout << k[0] << " * ";
            }
            else if (j != ax && j == n_dim - 1) {
                std::cout << data_dim[j] << " + ";
            }
            else {
                std::cout << k[0] << " + ";
            }
        }
    }
    std::cout << "] = k;" <<  std::endl;
    std::cout << "\t\t\t\t\t" << "break;" << std::endl;

    std::cout << "\t\t\t\t" << "}" << std::endl;
    std::cout << "\t\t\t" << "}" << std::endl;
    std::cout << "\t\t" << "}" << std::endl;

    for (unsigned i = 0; i < n_dim; i++){
        if (i != ax) {
            std::cout << "\t" << "}" << std::endl;
        }
    }
    */

    return 0;
}