#include "conv1d.h"

void conv1d_forward(float* input, const float* weight, const float* bias, float* output, 
                    int in_channels, int out_channels, int kernel_size, int stride, int padding,
                    int B, int W) {
    // output size
    int W_out = (int)((float)(W + 2 * padding - kernel_size) / stride) + 1;
    
    for (int n = 0; n < B; n++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int ox = 0; ox < W_out; ox++) {
                float sum = bias ? bias[oc] : 0.0f;

                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kx = 0; kx < kernel_size; kx++) {
                        int ix = ox * stride + kx - padding;

                        if (ix < 0 || ix >= W)
                            continue;

                        int in_idx  = n * in_channels * W + ic * W + ix;
                        int w_idx   = oc * in_channels * kernel_size + ic * kernel_size + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }

                int out_idx = n * out_channels * W_out + oc * W_out + ox;
                output[out_idx] = sum;
            }
        }
    }
}