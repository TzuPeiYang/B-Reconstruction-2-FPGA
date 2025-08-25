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
LONGOPTS=train,predict,graph,continue,
OPTIONS=tpgc

# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
# -if getopt fails, it complains itself to stderr
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@") || exit 2
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

t=0 p=0  g=0 c=0
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
SUFFIX='drop_0.03'
ROOT_DIR='/home/tpyang/GNN-2-FPGA-B-Reco/python/'
SUB_DIR='incomplete-1-B/'

MODEL_CONFIG='config/particlenet_pf.py'
DATA_CONFIG='config/data-config.yaml'
SAMPLES_DIR='data/'
PATH_TO_LOG='training-log/'

if [[ $t -eq 1 && $c -eq 1 ]]; then
    python /home/tpyang/weaver-core/weaver/train.py \
    --data-train ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'train*.root' \
    --data-val ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'test.root' \
    --fetch-by-file --fetch-step 1 --num-workers 2 \
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX} \
    --gpus 0 --batch-size 256 --start-lr 1e-2 --num-epochs 200 --optimizer ranger \
    --log ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}.train.log \
    --tensorboard ${PREFIX} \
    --regression-mode \
    --load-model-weights  ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt

    mv ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_best_epoch_state.pt ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt
    rm ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_epoch*.pt
elif [ $t -eq 1 ]; then
    python /home/tpyang/weaver-core/weaver/train.py \
    --data-train ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'train*.root' \
    --data-val ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'test.root' \
    --fetch-by-file --fetch-step 1 --num-workers 2 \
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX} \
    --gpus 0 --batch-size 256 --start-lr 1e-2 --num-epochs 200 --optimizer ranger \
    --log ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}.train.log \
    --tensorboard ${PREFIX} \
    --regression-mode

    mv ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_best_epoch_state.pt ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt
    rm ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_epoch*.pt
fi

if [ $p -eq 1 ]; then
    python /home/tpyang/weaver-core/weaver/train.py --predict\
    --data-test ${ROOT_DIR}${SUB_DIR}${SAMPLES_DIR}'test.root'\
    --num-workers 1 \
    --data-config ${ROOT_DIR}${SUB_DIR}${DATA_CONFIG} \
    --network-config ${ROOT_DIR}${SUB_DIR}${MODEL_CONFIG} \
    --model-prefix ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_${SUFFIX}.pt  \
    --gpus 0 --batch-size 256 \
    --predict-output ${ROOT_DIR}${SUB_DIR}${PATH_TO_LOG}${PREFIX}_predict_${SUFFIX}.root \
    --regression-mode
fi

if [ $g -eq 1 ]; then
    python plot_mass.py ${SUB_DIR}
fi
