#!/bin/bash

# More safety, by turning some bugs into errors.
set -o errexit -o pipefail -o noclobber -o nounset

# ignore errexit with `&& true`
getopt --test > /dev/null && true
if [[ $? -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

# option --output/-o requires 1 argument
LONGOPTS=train,predict,graph,continue,regression,save,mixed
OPTIONS=tpgcrsm

# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
# -if getopt fails, it complains itself to stderr
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@") || exit 2
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

t=0 p=0  g=0 c=0 r=0 s=0 m=0
# now enjoy the options in order and nicely split until we see --
while true; do
    case "$1" in
        -t|--train)
            t=1
            shift
            ;;
        -c|--continue)
            c=1
            shift
            ;;
        -p|--predict)
            p=1
            shift
            ;;
        -g|--graph)
            g=1
            shift
            ;;
        -r|--regression)
            r=1
            shift
            ;;
        -m|--mixed)
            m=1
            shift
            ;;
        -s|--save)
            s=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

PREFIX='particlenet'
SUFFIX='complete'
ROOT_DIR='/home/tpyang/B-Reconstruction-2-FPGA/python/GNN-part'
SUB_DIR='pure_B_plus_B_minus/gen_level_1B/with_vertex/'
MODEL_CONFIG='config/'${PREFIX}'_'${SUFFIX}'.py'
DATA_CONFIG='config/data_config_'${SUFFIX}'.yaml'

SAMPLES_DIR='../data/'
PATH_TO_LOG='training_log/'

args=( --data-train ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'train_0*.root' \
    --data-val ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'test*.root' \
    --fetch-by-file --fetch-step 1 --num-workers 2 \
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX} \
    --gpus 0 --batch-size 256 --start-lr 5e-3 --num-epochs 200 --optimizer ranger \
    --log ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}.train.log \
    --tensorboard ${PREFIX} )
if [ $r -eq 1 ]; then 
    args+=( --regression-mode ) 
fi
if [ $m -eq 1 ]; then 
    args+=( --mixed-mode ) 
fi
if [ $c -eq 1 ]; then 
    args+=( --load-model-weights ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt ) 
fi

if [ $t -eq 1 ]; then
    weaver "${args[@]}"

    mv ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_best_epoch_state.pt ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt
    rm ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_epoch*.pt
fi

pred_args=( --predict \
    --data-test ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'test*.root' \
    --num-workers 1 \
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt  \
    --gpus 0 --batch-size 256 \
    --predict-output ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_predict_${SUFFIX}.root 
    )
if [ $r -eq 1 ]; then 
    pred_args+=( --regression-mode ) 
fi
if [ $m -eq 1 ]; then 
    pred_args+=( --mixed-mode ) 
fi
if [ $p -eq 1 ]; then
    weaver "${pred_args[@]}"
fi

if [ $g -eq 1 ]; then
    python plot_mass.py ${SUB_DIR}
fi

save_args=( 
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt \
    --onnx-opset 18 \
    --export-onnx ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.onnx
     )

if [ $s -eq 1 ]; then
    weaver "${save_args[@]}"
fi

