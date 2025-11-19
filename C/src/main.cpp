#include "model_weights.h"
#include <stdio.h>
#include <stddef.h>
#include <iostream>

#include "network.h"
#include "pf_points.h"
#include "pf_features.h"

int main(int argc, char** argv) {
    
    float tensor_output[8][8][10];
    entry(pf_points, pf_features, tensor_output);

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 10; k++) {
                std::cout << tensor_output[i][j][k] << std::endl;
            }
        }
    }

    return 0;
}