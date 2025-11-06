#include "relu.h"

void relu_forward(float* input, int B, int C, int H, int W) {
    for (int i = 0; i < B * C * H * W; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}