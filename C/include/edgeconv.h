#ifndef EDGE_CONV_H
#define EDGE_CONV_H

/*
    * Function to compute edge convolution.
    * Parameters:
    *   pf_points          : Pf_points of shape (B, D1, N)
    *   pf_features        : Pf_feautres of shape (B, in_channels, N)
    *   top_k              : Number of nearest neighbours to find
    *   in_channels        : Number of input features
    *   out_channels       : Number of output features (layers,)
    *   output             : Output
    *   conv2d_weights     : Weights for the internal conv2d layers, the layers are concatenated
    *   conv2d_biases      : Biases for the internal conv2d layers, the layers are concatenated
    *   batchnorm2d_weights: Weights for the internal batchnorm2d layers, the layers are concatenated
    *   batchnorm2d_biases : Biases for the internal batchnorm2d layers, the layers are concatenated
    *   batchnorm2d_means  : Running mean for the internal batchnorm2d layers, the layers are concatenated
    *   batchnorm2d_vars   : Running variance for the internal batchnorm2d layers , the layers are concatenated
    *   conv1d_weight      : Weight for the shortcut conv1d
    *   batchnorm1d_weight : Weight for the shortcut batchnorm1d layers
    *   batchnorm1d_bias   : Bias for the shortcut batchnorm1d layers
    *   batchnorm1d_mean   : Running mean for the shortcut batchnorm1d layers
    *   batchnorm1d_var    : Running variance for the shortcut batchnorm1d layers
    *   B                  : Batch size
    *   D                  : Dimension of point coordinates
    *   N                  : Number of points
    *   layers             : Number of layers
    *   do_batchnorm       : Whether to apply batch normalization
    *   do_activation      : Whether to apply activation function
*/
void edgeconv_forward(float* pf_points, float* pf_features, float* output, int do_batchnorm, int do_activation);

void edge_conv_block(float* pf_points, float* pf_features, float* output, int do_activation, int do_batchnorm);

#endif // EDGE_CONV_H