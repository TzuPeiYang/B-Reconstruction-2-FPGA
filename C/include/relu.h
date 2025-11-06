#ifndef RELU_H
#define RELU_H

/*
    * Function to compute rectified linear unit inplace (input is also the output).
    * Parameters:
    *   input      : Input, is also the output after ReLU
    *   B          : Batch size
    *   C          : Number of channels
    *   H          : Height
    *   W          : Width
*/
void relu_forward(float* input, int B, int C, int H, int W);

#endif // RELU_H