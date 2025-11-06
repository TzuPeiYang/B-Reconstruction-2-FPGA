#include "conv2d.h"
#include <stdio.h>

void conv2d_forward(float* input, const float* weight, const float* bias, float* output, 
            int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2],
            int B, int H, int W) {
    // output size
    int H_out = (H + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
    int W_out = (W + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
    
    for (int n = 0; n < B; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oy = 0; oy < H_out; oy++) {
                for (int ox = 0; ox < W_out; ox++) {
                    float sum = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int ky = 0; ky < kernel_size[0]; ky++) {
                            for (int kx = 0; kx < kernel_size[1]; kx++) {
                                int iy = oy * stride[0] + ky - padding[0];
                                int ix = ox * stride[1] + kx - padding[1];
                                if (iy < 0 || iy >= H || ix < 0 || ix >= W)
                                    continue;
                                sum += input[n * in_channels * H * W + ic * H * W + iy * W + ix] * weight[oc * in_channels * kernel_size[0] * kernel_size[1] + ic * kernel_size[0] * kernel_size[1] + ky * kernel_size[1] + kx];
                                // printf("%.6f\n", input[n * in_channels * H * W + ic * H * W + iy * W + ix]);
                            }
                        }
                    }
                    output[n * out_channels * H_out * W_out + oc * H_out * W_out + oy * W_out + ox] = sum;
                }
            }
        }
    }
}