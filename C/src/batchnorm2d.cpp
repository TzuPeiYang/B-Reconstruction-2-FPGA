#include "batchnorm2d.h"
#include <math.h>

void batchnorm2d_forward(float* input, float* output, 
                         const float* weight, const float* bias, const float* mean, const float* var, float eps, 
                         int B, int C, int H, int W) {
    for (int n = 0; n < B; n++) {
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    output[n * C * H * W + c * H * W + y * W + x] = weight[c] * (input[n * C * H * W + c * H * W + y * W + x] - mean[c]) / sqrtf(var[c] + eps) + bias[c];
                }
            }
        }
    }
}