#ifndef CONV1D_H
#define CONV1D_H

/*
    * Function to compute 2D convolution.
    * Parameters:
    *   input       : Input (B, in_channels, W)
    *   weight      : Weights (in_channels, out_channels, .., ..)
    *   bias        : Biases
    *   output      : Output
    *   in_channels : in channels
    *   out_channels: out channels
    *   kernel_size : Kernel size of the convolution
    *   stride      : Stride of the convolution
    *   padding     : Padding of the convolution
    *   B           : Batch size
    *   W           : Width of the input
*/
void conv1d_forward(float* input, const float* weight, const float* bias, float* output, 
                    int in_channels, int out_channels, int kernel_size, int stride, int padding,
                    int B, int W);

#endif // CONV1D_H