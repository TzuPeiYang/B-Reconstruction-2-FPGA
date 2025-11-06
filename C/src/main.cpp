#include "model_weights.h"
#include "edgeconv.h"
#include <stdio.h>
#include <stddef.h>

int main(int argc, char** argv) {
    // test inputs for model inference
    int B = 8;
    int D = 3;
    int in_channels = 4;
    int N = 10;
    float pf_points[B * D * N];
    float pf_features[B * in_channels * N];

    for (int i = 0; i < B * D * N; i++) {
        pf_points[i] = (float)i;
    }
    for (int i = 0; i < B * in_channels * N; i++) {
        pf_features[i] = (float)i;
    }

    // test network
    int layers = 1;
    int out_channels[1] = {8};
    float output[B * out_channels[layers - 1] * N];
    
    edge_conv_block(pf_points, pf_features, output, 1, 1);
    
    /*
    for (int i = 0; i < B; i++) {
        printf("[");
        for (int j = 0; j < out_channels[layers - 1]; j++) {
            printf("[");
            for (int k = 0; k < N; k++) {
                printf("%.4f, ", output[i * out_channels[layers - 1] * N + j * N + k]);
            }
            printf("]\n");
        }
        printf("]\n");
    }
    */

    return 0;
}