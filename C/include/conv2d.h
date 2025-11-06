#ifndef CONV2D_H
#define CONV2D_H

/*
    * Function to compute 2D convolution.
    * Parameters:
    *   input       : Input
    *   weight      : Weights
    *   bias        : Biases
    *   output      : Output
    *   in_channels : in channels
    *   out_channels: out channels
    *   kernel_size : Kernel size of the convolution
    *   stride      : Stride of the convolution
    *   padding     : Padding of the convolution
    *   B           : Batch size
    *   H           : Height of the input
    *   W           : Width of the input
*/
void conv2d_forward(float* input, const float* weight, const float* bias, float* output, 
            int in_channels, int out_channels, int kernel_size[2], int stride[2], int padding[2],
            int B, int H, int W);

#endif // CONV2D_H