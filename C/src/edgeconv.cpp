#include "edgeconv.h"
#include "conv2d.h"
#include "conv1d.h"
#include "batchnorm2d.h"
#include "batchnorm1d.h"
#include "relu.h"

#include <stdio.h>
#include <stddef.h>

#include "model_weights.h"

static int kernel_size[2] = {1, 1};
static int stride[2] = {1, 1};
static int padding[2] = {0, 0};

template <int in_channels, int layers, int B, int D, int N, int top_k, 
          int* out_channels, 
          float* conv2d_weights, float* conv2d_biases,
          float* batchnorm2d_weights, float* batchnorm2d_biases, float* batchnorm2d_means, float* batchnorm2d_vars,
          const float* conv1d_weight,
          const float* sc_weight, const float* sc_bias, const float* sc_mean, const float* sc_var>
void edgeconv_forward(float* pf_points, float* pf_features, float* output, int do_batchnorm, int do_activation) {
    // Compute edge features
    // will not handle N < top_k !!!!
    static float edge_features[B * 2 * in_channels * N * top_k];

    // -------------------- Compute pairwise distance (squared) --------------------
    float pairwise_distance[B * N * N]; 
    for (int i = 0; i < B; i++){
        for (int j = 0; j < N; j++){
            for (int k = j; k < N; k++){
                pairwise_distance[i * N * N + j * N + k] = -(pf_points[i * D * N + 0 * N + j] - pf_points[i * D * N + 0 * N + k]) * (pf_points[i * D * N + 0 * N + j] - pf_points[i * D * N + 0 * N + k]) -
                                                        (pf_points[i * D * N + 1 * N + j] - pf_points[i * D * N + 1 * N + k]) * (pf_points[i * D * N + 1 * N + j] - pf_points[i * D * N + 1 * N + k]) -
                                                        (pf_points[i * D * N + 2 * N + j] - pf_points[i * D * N + 2 * N + k]) * (pf_points[i * D * N + 2 * N + j] - pf_points[i * D * N + 2 * N + k]);
                pairwise_distance[i * N * N + k * N + j] = pairwise_distance[i * N * N + j * N + k];
                #ifdef DEBUG
                    printf("pd[%d][%d][%d] = %.3f\n", i, j, k, pairwise_distance[i * N * N + j * N + k]);
                #endif                          
            }   
        }
    }
    // -------------------- Find k-nearest neighbours -------------------
    int indices[B * N * top_k];
    float values[top_k];
    for (int i = 0; i < B; i++){
        for (int j = 0; j < N; j++){
            // initialise top_k values
            for (int k = 0; k < top_k; k++){
                values[k] = -1e30f;
                indices[i * N * top_k + j * top_k + k] = -1;
            }
            for (int k = 0; k < N; k++){
                if (k == j) {
                    continue;
                }
                for (int d = 0; d < top_k; d++){
                    if (pairwise_distance[i * N * N + j * N + k] > values[d]){
                        // shift smaller elements down
                        for (int s = top_k - 1; s > d; s--) {
                            values[s] = values[s - 1];
                            indices[i * N * top_k + j * top_k + s] = indices[i * N * top_k + j * top_k + s - 1];
                        }
                        values[d] = pairwise_distance[i * N * N + j * N + k];;
                        indices[i * N * top_k + j * top_k + d] = k;
                        break;
                    }
                }
            }
            #ifdef DEBUG
                for (int d = 0; d < top_k; d++){
                    printf("%.3f ", values[d]);
                }
                printf("\n");
            #endif
        }
    }
    
    // -------------------- Gather features of k-nearest-neighbour --------------------
    int idx, ft_idx;
    float xi, xj;
    for (int i = 0; i < B; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < top_k; k++) {
                idx = indices[i * N * top_k + j * top_k + k];

                for (int d = 0; d < in_channels; d++) {
                    xi = pf_features[i * in_channels * N + d * N + j];
                    xj = pf_features[i * in_channels * N + d * N + idx];

                    ft_idx = i * (2 * in_channels) * N * top_k + d * N * top_k + j * top_k + k;
                    edge_features[ft_idx] = xi;
                    edge_features[ft_idx + in_channels * N * top_k] = xj - xi;
                }
            }
        }
    }

    #ifdef DEBUG
        printf("Edge features computed.\n");
        for (int i = 0; i < 2 * in_channels; i++) {
            printf("[");
            for (int j = 0; j < N; j++) {
                printf("[");
                for (int k = 0; k < top_k; k++) {
                    printf("%.3f, ", edge_features[i * N * top_k + j * top_k + k]);
                }
                printf("]");
            }
            printf("]\n");
        }
    #endif
    
    // -------------------- Compute network forward --------------------
    int offset_1d = 0;
    int offset_2d = 0;
    int max_channels;
    for (int i = 0; i < layers; i++) {
        if (out_channels[i] > max_channels) {
            max_channels = out_channels[i];
        }
    }
    // copy input to buffer
    float input_buffer[B * max_channels * N * top_k];
    for (int i = 0; i < B * 2 * in_channels * N * top_k; i++) {
        input_buffer[i] = edge_features[i];
    }

    // compute network layers
    for (int i = 0; i < layers; i++) {
        int in_ch, out_ch;
        if (i == 0) {
            in_ch = 2 * in_channels;
            out_ch = out_channels[0];
        }
        else {
            in_ch = out_channels[i - 1];
            out_ch = out_channels[i];
        }
        
        // apply conv2d
        float conv_output[B * out_ch * N * top_k];
        float conv_weight[out_ch * in_ch * 1 * 1];
        float conv_bias[out_ch];
        
        for (int j = 0; j < out_ch; j++) {
            if (do_batchnorm != 1) {
                conv_bias[j] = conv2d_biases[offset_1d + j]; // offset
            }
            for (int k = 0; k < in_ch; k++) {
                conv_weight[j * in_ch + k] = conv2d_weights[offset_2d + j * in_ch + k];
            }
        }
    
        conv2d_forward(input_buffer, conv_weight, conv_bias, conv_output, 
                       in_ch, out_ch, kernel_size, stride, padding,
                       B, N, top_k);
        
        for (int j = 0; j < B; j++) {
            for (int k = 0; k < out_ch; k++) {
                for (int l = 0; l < N; l++) {
                    for (int m = 0; m < top_k; m++) {
                        printf("%.3f ", conv_output[j * out_ch * N * top_k + k * N * top_k + l * top_k + m]);
                    }
                    printf("\n");
                }
            }
        }
        
        // apply batchnorm and activation
        if (do_batchnorm == 1) {
            float batchnorm_output[B * out_ch * N * top_k];
            float batchnorm_bias[out_ch];
            float batchnorm_weight[out_ch];
            float batchnorm_mean[out_ch];
            float batchnorm_var[out_ch];
            for (int j = 0; j < out_ch; j++) {
                batchnorm_bias[j] = batchnorm2d_biases[offset_1d + j];
                batchnorm_weight[j] = batchnorm2d_weights[offset_1d + j];
                batchnorm_mean[j] = batchnorm2d_means[offset_1d + j];
                batchnorm_var[j] = batchnorm2d_vars[offset_1d + j];
            }
            batchnorm2d_forward(conv_output, batchnorm_output,
                                batchnorm_weight, batchnorm_bias,
                                batchnorm_mean, batchnorm_var, 1e-5,
                                B, out_ch, N, top_k);
            // copy back to conv_output
            for (int j = 0; j < B * out_ch * N * top_k; j++) {
                conv_output[j] = batchnorm_output[j];
            }
        }
        if (do_activation == 1) {
            relu_forward(conv_output, B, out_ch, N, top_k);
        }
        offset_1d += out_ch;
        offset_2d += in_ch * out_ch;
        for (int i = 0; i < B * out_ch * N * top_k; i++) {
            input_buffer[i] = conv_output[i];
        }
        
    }
    
    // calculate mean over k neighbours
    for (int b = 0; b < B; b++) {
        for (int c = 0; c < out_channels[layers - 1]; c++) {
            for (int n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int k = 0; k < top_k; k++) {
                    sum += input_buffer[b * out_channels[layers - 1] * N * top_k + c * N * top_k + n * top_k + k];
                }
                output[b * out_channels[layers - 1] * N + c * N + n] = sum / top_k;
            }
        }
    }

    // shortcuts
    if (in_channels != out_channels[layers - 1]) {
        float conv_output[B * out_channels[layers - 1] * N];
        conv1d_forward(pf_features, conv1d_weight, NULL, conv_output, 
                       in_channels, out_channels[layers - 1], 1, 1, 0, B, N);

        float shortcut[B * out_channels[layers - 1] * N];
        batchnorm1d_forward(conv_output, shortcut, 
                            sc_weight, sc_bias, sc_mean, sc_var, 
                            1e-5, B, out_channels[layers - 1], N);
        for (int i = 0; i < B * out_channels[layers - 1] * N; i++) {
            output[i] += shortcut[i];
        }
    }
    else {
        for (int i = 0; i < B * in_channels * N; i++) {
            output[i] += pf_features[i];
        }
    }
    if (do_activation == 1) {
        relu_forward(output, B, out_channels[layers - 1], N, 1);
    }
    
}

void edge_conv_block(float* pf_points, float* pf_features, float* output, int do_activation, int do_batchnorm) {
    static int out_channels[1] = {8};
    edgeconv_forward<4, 1, 8, 3, 10, 3, 
                     out_channels, 
                     (float*)convs_0_weight, nullptr, 
                     (float*)bns_0_weight, (float*)bns_0_bias, (float*)bns_0_running_mean, (float*)bns_0_running_var, 
                     sc_weight, 
                     sc_bn_weight, sc_bn_bias, sc_bn_running_mean, sc_bn_running_var>(pf_points, pf_features, output, do_activation, do_batchnorm);
}