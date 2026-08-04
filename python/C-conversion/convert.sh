#!/bin/bash

ROOT_DIR='/home/tpyang/B-Reconstruction-2-FPGA/python/C-conversion/'
SUB_DIR='../GNN-part/pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/'

MODEL_SUFFIX='complete,1EC'

TEMP_DIR='patched_ONNX/'

# patch_onnx.py has to be modified for use case
python patch_onnx.py --model ${ROOT_DIR}${SUB_DIR}particlenet_${MODEL_SUFFIX}.onnx --out ${TEMP_DIR}particlenet_${MODEL_SUFFIX}_cleaned.onnx

# load callibration data and find range requirements for ap_fixed 

python export_calibration_data.py \
    --data-config  ${ROOT_DIR}${SUB_DIR}../config/data_config_${MODEL_SUFFIX}.yaml \
    --data-files   ${ROOT_DIR}${SUB_DIR}../../data/test.root \
    --output       calibration.npz \
    --n-events     2000

python find_range_requirements.py ${TEMP_DIR}particlenet_${MODEL_SUFFIX}_cleaned.onnx \
    --data calibration.npz \
    --input-names pf_points pf_features pf_mask \
    --n-samples 2000 \
    --hotspot-threshold 8 \
    --total-bits 32 \
    --csv ranges.csv


# convert to C code
/home/tpyang/onnx2c/build/onnx2c ${TEMP_DIR}particlenet_${MODEL_SUFFIX}_cleaned.onnx > ../../C/src/particlenet_complete.cpp
python generate_header_4_C_code.py --inpath ../../C/src/particlenet_complete.cpp --outpath ../../C/include/particlenet_complete.h

cp ../../C/src/particlenet_complete.cpp ../../hls/model_inference.cpp

python remove_union.py ../../hls/model_inference.cpp
python remove_constants.py ../../hls/model_inference.cpp

python simplify_1x1_conv.py ../../hls/model_inference.cpp
python simplify_gemm.py ../../hls/model_inference.cpp
python optimize_relu.py ../../hls/model_inference.cpp
python optimize_matmul.py ../../hls/model_inference.cpp

python add_inline_off.py ../../hls/model_inference.cpp

python apply_tree_reduction.py  ../../hls/model_inference.cpp  ../../hls/model_inference.cpp


# Step 1: classify every tensor and assign ap_fixed types
python assign_types.py ${TEMP_DIR}particlenet_${MODEL_SUFFIX}_cleaned.onnx \
    --ranges ranges.csv \
    --output tensor_types.json \
    --dump-groups          # optional: print every tensor and its classification

# Step 2: rewrite the C header
python rewrite_types.py ../../hls/model_inference.cpp tensor_types.json ../../hls/model_inference.cpp

# Step 3: patch the rest
python fix_types.py ../../hls/model_inference.cpp ../../hls/model_inference.cpp

python split_weights.py ../../hls/model_inference.cpp \
    --output      ../../hls/model_inference.cpp \
    --output-dir  ../../hls/weights \
    --weights-h   ../../hls/weights/weights.h \
    --min-elements 8

python fix_includes.py ../../hls/model_inference.cpp 

python fix_reshape.py ../../hls/model_inference.cpp ../../hls/model_inference.cpp
