#include <stdio.h>
#include <stddef.h>
#include <iostream>

#include "particlenet_complete.h"
#include "pf_points.h"
#include "pf_features.h"
#include "pf_mask.h"

int main(int argc, char** argv) {
    
    float tensor_output[1][4];
    entry(pf_points, pf_features, pf_mask, tensor_output);

    for (int i = 0; i < 4; i++) {
        std::cout << tensor_output[0][i] << " ";
    }
    std::cout << std::endl;

    return 0;
}