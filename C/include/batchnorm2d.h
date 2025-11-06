#ifndef BATCHNORM2D_H
#define BATCHNORM2D_H

/*
    * Function to compute edge convolution.
    * Parameters:
    *   input : Input (B, C, H, W)
    *   output: Output (B, C, H, W)
    *   weight: Scale parameter (C,)
    *   bias  : Shift parameter (C,)
    *   mean  : Running mean (C,)
    *   var   : Running variance (C,)
    *   eps   : Small value to avoid division by zero
    *   B     : Batch size
    *   C     : Number of channels
    *   H     : Height
    *   W     : Width
*/
void batchnorm2d_forward(float* input, float* output, 
                         const float* weight, const float* bias, const float* mean, const float* var, float eps, 
                         int B, int C, int H, int W);

#endif // BATCHNORM2D_H