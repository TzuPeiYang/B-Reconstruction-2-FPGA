#!/bin/bash

ROOT_DIR='/home/tpyang/B-Reconstruction-2-FPGA/python/C-conversion/'
SUB_DIR='../GNN-part/pure_B_plus_B_minus/gen_level_1B/with_vertex/training_log/'
ONNX_NAME='particlenet_complete'
TEMP_DIR='patched_ONNX/'

# patch_onnx.py has to be modified for use case

python patch_onnx.py --model ${ROOT_DIR}${SUB_DIR}${ONNX_NAME}.onnx --out ${TEMP_DIR}${ONNX_NAME}_cleaned.onnx
/home/tpyang/onnx2c/build/onnx2c ${TEMP_DIR}${ONNX_NAME}_cleaned.onnx > ../../C/src/${ONNX_NAME}.cpp
python generate_header_4_C_code.py --inpath ../../C/src/${ONNX_NAME}.cpp --outpath ../../C/include/${ONNX_NAME}.h