# B-Reconstruction to FPGA
Aiming to configure FPGAs to run graph neural networks (GNNs) for B-meson reconstruction. Now at the NN design phase, this repo relies on the weaver-core framework to train and test. The C++ part is currently empty except for loading data.

## Dependencies
- python: pyhepmc, uproot, torch 
- C++: yaml-cpp (for loading test data)

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

## Training 
The ```python/train.sh``` file depends on weaver-core to function. The directory of ```train.py``` has to be changed appropriately. If weaver-core is installed via pip, change the ```python ${directory}/train.py "${args[@]}"``` to just ```weaver "${args[@]}"```. Then modify the following part of the script to appropriate files and names
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
- -p | --predict: load model and predict
- -r | --regression: train or predict regression, classification is defaulted if this is missing
- -s | --save: save model to ONNX
- -g | --graph: plot calls ```plot_mass.py``` to plot all prediction outputs in the training_log directory

For example, to train regression, run 
```
./train.sh -tr
```

## Python code usages
- hepmc_2_root.py \
The original data are in hepmc3 format, this piece of code converts hepmc3 file ```./data/${filename}.hepmc3``` to ready to train root file ```./${subdirectory}/data/${filename}.root```. The selections or any preprocessing are also done in this file.
```
python hepmc_2_root.py ${subdirectory} ${filename}
```

- add_mask_2_data.py

