#pragma once
#include <ap_fixed.h>

/* Extern declarations for split weight tensors.
 * Each tensor is defined in weights_split/<name>.cpp
 */

extern const ap_fixed<18,5> tensor_fc_out_0_weight[4][32];
extern const ap_fixed<18,5> tensor_mod_edge_convs_0_output_bn_weight[32];
extern const ap_fixed<18,5> tensor_mod_edge_convs_0_output_bn_bias[32];
extern const ap_fixed<18,5> tensor_mod_edge_convs_0_output_bn_running_mean[32];
extern const ap_fixed<18,5> tensor_mod_edge_convs_0_output_bn_running_var[32];
extern const ap_fixed<18,5> tensor_mod_fc_0_0_weight[32][128];
extern const ap_fixed<18,5> tensor_mod_fc_0_0_bias[32];
extern const ap_fixed<18,5> tensor_onnx__Conv_169[32][8][1][1];
extern const ap_fixed<18,5> tensor_onnx__Conv_170[32];
extern const ap_fixed<18,5> tensor_onnx__Conv_172[32][32][1][1];
extern const ap_fixed<18,5> tensor_onnx__Conv_173[32];
extern const ap_fixed<18,5> tensor_onnx__Conv_175[32][32][1][1];
extern const ap_fixed<18,5> tensor_onnx__Conv_176[32];
extern const ap_fixed<18,5> tensor_onnx__Conv_178[32][4][1];
extern const ap_fixed<18,5> tensor_onnx__Conv_179[32];
extern const ap_fixed<18,5> tensor_onnx__Conv_181[128][32][1];
extern const ap_fixed<18,5> tensor_onnx__Conv_182[128];
