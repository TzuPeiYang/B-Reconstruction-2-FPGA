# B-Reconstruction to FPGA
Aiming to configure FPGAs to run graph neural networks (GNNs) for B-meson reconstruction. Now at the NN design phase, this repo relies on the [weaver-core](https://github.com/hqucms/weaver-core?tab=readme-ov-file) framework to train and test. Some of the files in weaver-core are modified for various purposes, so in order to use this repository, clone and set up the [forked-weaver](https://github.com/TzuPeiYang/weaver-core) instead. The C++ part is generated using [onnx2c](https://github.com/kraiskil/onnx2c/tree/master). This [forked-onnx2c](https://github.com/TzuPeiYang/onnx2c) has TopK node and Tile node implemented, which are used in our models.

## Dependencies
- python: pyhepmc (has to be installed with HepMC3), uproot, torch 
- onnx2c

## Folder structure
- C
  - bin
  - build
  - include
  - src

- python
  - GNN-part
    - data (hepmc3 files)
    - runs (training and validation loss)
    - all_B
    - pure_B_plus_B_minus
      - gen_level_1B
      - gen_level_2B
        - data (training and testing data)
        - with_eta_phi
        - with partial_vertex
        - with_vertex
          - config (data_config.yaml, model_config.py)
          - training_log (model files and prediction outputs)
  - C-conversion
    - onnx2c_test (some pytorch modules for testing)
    - patched_ONNX (where ONNX is saved after patching dynamic axes and unused attributes)

## Data
After unzipping the `data.tar.gz`, you will find existing MC events. The hepmc3 files `train_01.hepmc3`, `train_02.hepmc3`, `test.hepmc3` are pure $\Upsilon(4S) \rightarrow B^+ B^-$, the first two sets each contain 100k events, and the last one contain 10k events. `train_03.hepmc3` and `test_03.hepmc3` contains 200k and 20k events with the actual branching ratios, respectively.

## How to train
1. Use `hepmc_2_root.py` to generate training and testing data in root file format, for details see [Python code usages](README.md#python-code-usages).

2. The `python/GNN-part/train.sh` file depends on weaver-core to function.
```
PREFIX='particlenet'
SUFFIX='complete'
ROOT_DIR='/home/B-Reconstruction-2-FPGA/python/'
SUB_DIR='pure_B_plus_B_minus/gen_level_1B/with_partial_vertex/'
```

The shell script takes the following flags
- -t | --train: to train
- -c | --continue: continue with existing model
- -p | --predict: load model and output prediction in root files
- -r | --regression: train or predict regression, classification is defaulted if this is missing
- -m | --mixed: train with a regression head and a classification head
- -s | --save: save model to ONNX
- -g | --graph: plot calls `plot_mass.py` to plot all prediction outputs in the training_log directory

For example, to train regression, run 
```
./train.sh -tr
```

## How to convert to C code
In the `C-conversion` directory, there are python files and a shell script. Modify the `manual_shape_override` function in `patch_onnx.py` and the paths in the shell script, then run 
```
./convert.sh
```
It would generate a C++ file in `C/src/` and a header in `C/include/`. If the conversion goes wrong at the onnx2c level, then the ONNX file probably need more patching. 

## Python code usages
- `hepmc_2_root.py` \
The original data are in hepmc3 format, this piece of code converts hepmc3 file `./data/${filename}.hepmc3` to ready to train root file `./${directory}/data/${filename}.root`. The selections or any preprocessing are also done in this file.
```
python hepmc_2_root.py ${directory} ${filename}
```

- `add_mask_2_data.py` \
When training the network for masking, after prediction run this to add `pred_mask` branch to the original training data. For example, after prediction of masking for `${directory}/data/${filename}.root`, a `score.npz` will appear, then run 
```
python add_mask_2_data.py ${directory} ${filename}
```

- `plot_mass.py`
After the 4-momentum prediction of test data, a output root file will appear in `${directory}/training_log`, run the following to plot the resulting reconstructed mass.
```
python plot_mass.py ${directory}
```