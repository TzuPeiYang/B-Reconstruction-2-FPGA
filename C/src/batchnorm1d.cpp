#include "batchnorm1d.h"
#include <math.h>

void batchnorm1d_forward(float* input, float* output, 
                         const float* weight, const float* bias, const float* mean, const float* var, float eps, 
                         int B, int C, int W) {
    for (int n = 0; n < B; n++) {
        for (int c = 0; c < C; c++) {
            float g = weight[c];
            float b = bias[c];
            float m = mean[c];
            float v = var[c];
            float inv_std = 1.0f / sqrtf(v + eps);
            for (int x = 0; x < W; x++) {
                int idx = n * C * W + c * W + x;
                float norm = (input[idx] - m) * inv_std;
                output[idx] = g * norm + b;
            }
        }
    }
}