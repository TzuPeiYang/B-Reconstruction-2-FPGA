#ifndef BATCHNORM1D_H
#define BATCHNORM1D_H

/*
    * Function to compute edge convolution.
    * Parameters:
    *   input : Input (B, C, W)
    *   output: Output (B, C, W)
    *   weight: Scale parameter (C,)
    *   bias  : Shift parameter (C,)
    *   mean  : Running mean (C,)
    *   var   : Running variance (C,)
    *   eps   : Small value to avoid division by zero
    *   B     : Batch size
    *   C     : Number of channels
    *   W     : Width
*/
void batchnorm1d_forward(float* input, float* output, 
                         const float* weight, const float* bias, const float* mean, const float* var, float eps, 
                         int B, int C, int W);

#endif // BATCHNORM2D_H