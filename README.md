# B-Reconstruction to FPGA
Aiming to configure FPGAs to run graph neural networks (GNNs) for B-meson reconstruction. Now at the NN design phase, this repo relies on the [weaver-core](https://github.com/hqucms/weaver-core?tab=readme-ov-file) framework to train and test. Some of the files in weaver-core are modified for various purposes, so in order to use this repository, clone and set up the [forked-version](https://github.com/TzuPeiYang/weaver-core) instead. The C++ part is currently still empty except for loading data. 

## Dependencies
- python: pyhepmc (has to be installed with HepMC3), uproot, torch 
- C++: yaml-cpp (to load data for tests)

## Folder structure
- C
  - bin
  - build
  - lib
  - src
  - test

- python
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

## Data
After unzipping the `data.tar.gz`, you will find existing MC events. The hepmc3 files `train_01.hepmc3`, `train_02.hepmc3`, `test.hepmc3` are pure $\Upsilon(4S) \rightarrow B^+ B^-$, the first two sets each contain 100k events, and the last one contain 10k events. `train_03.hepmc3` and `test_03.hepmc3` contains 200k and 20k events with the actual branching ratios, respectively.

## How to run
1. Use `hepmc_2_root.py` to generate training and testing data in root file format, for details see [Python code usages](README.md#python-code-usages).

2. The `python/train.sh` file depends on weaver-core to function.
```
PREFIX='particlenet'
SUFFIX='complete'
ROOT_DIR='/home/B-Reconstruction-2-FPGA/python/'
SUB_DIR='pure_B_plus_B_minus/gen_level_1B/with_partial_vertex/'
MODEL_CONFIG='config/particlenet_pf.py'
DATA_CONFIG='config/data_config.yaml'
```

The shell script takes the following flags
- -t | --train: to train
- -c | --continue: continue with existing model
- -p | --predict: load model and output prediction in root files
- -r | --regression: train or predict regression, classification is defaulted if this is missing
- -s | --save: save model to ONNX
- -g | --graph: plot calls `plot_mass.py` to plot all prediction outputs in the training_log directory

For example, to train regression, run 
```
./train.sh -tr
```

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