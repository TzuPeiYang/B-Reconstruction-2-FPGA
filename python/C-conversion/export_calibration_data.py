#!/usr/bin/env python3
"""
export_calibration_data.py
===========================
Exports a small calibration .npz from a weaver dataset, ready for use
with find_range_requirements.py.

Usage
------
    python export_calibration_data.py \
        --data-config  data/config.yaml        \
        --data-files   data/test/*.root         \
        --output       calibration.npz          \
        --n-events     500
"""

import argparse
import glob
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-config',  required=True)
    p.add_argument('--data-files',   required=True, nargs='+')
    p.add_argument('--output',       default='calibration.npz')
    p.add_argument('--n-events',     type=int, default=500)
    p.add_argument('--input-groups', nargs='*', default=None)
    p.add_argument('--num-workers',  type=int, default=1)
    p.add_argument('--fetch-step',   type=float, default=0.01)
    return p.parse_args()


def expand_files(patterns):
    files = []
    for pat in patterns:
        matched = glob.glob(pat)
        if not matched:
            print(f'  Warning: no files matched "{pat}"')
        files.extend(matched)
    if not files:
        print('Error: no input files found.')
        sys.exit(1)
    return files


def build_dataloader(data_config_path, file_list, num_workers, fetch_step):
    from torch.utils.data import DataLoader
    from weaver.utils.dataset import SimpleIterDataset

    file_dict = {'_calib': file_list}

    dataset = SimpleIterDataset(
        file_dict,
        data_config_path,
        for_training=False,
        fetch_step=fetch_step,
        infinity_mode=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
    )
    return loader, dataset._data_config


def get_all_groups(data_config):
    """
    Try several attribute names that different weaver versions use
    for the ordered list of input group names.
    """
    for attr in ('input_names', 'input_dicts', 'inputs'):
        val = getattr(data_config, attr, None)
        if val is not None:
            return list(val.keys()) if isinstance(val, dict) else list(val)
    # Last resort: look at the data_config object directly
    print('  Warning: could not find input group names automatically.')
    print(f'  data_config attributes: {[a for a in dir(data_config) if not a.startswith("_")]}')
    return []


def inspect_batch(batch):
    """
    Work out the structure of a weaver batch and return a getter function.

    Weaver batches are (X, y, weight).
    X can be:
      - a dict  keyed by input group name   (newer weaver versions)
      - a tuple/list ordered by YAML group  (older weaver versions)
    """
    print('  Batch structure inspection:')
    print(f'    type(batch)    = {type(batch)}')
    print(f'    len(batch)     = {len(batch)}')

    X = batch[0]
    print(f'    type(batch[0]) = {type(X)}')

    if isinstance(X, dict):
        print(f'    X is a dict.  Keys:')
        for k, v in X.items():
            print(f'      "{k}"  dtype={v.dtype}  shape={tuple(v.shape)}')
        # getter: use name as key
        def getter(X, name, idx):
            return X[name]
        return getter

    if isinstance(X, (list, tuple)):
        print(f'    X is a {type(X).__name__} of length {len(X)}.')
        for i, v in enumerate(X):
            shape = tuple(v.shape) if hasattr(v, 'shape') else '?'
            print(f'      [{i}]  dtype={v.dtype}  shape={shape}')
        # getter: use positional index
        def getter(X, name, idx):
            return X[idx]
        return getter

    # Single tensor — unusual, print and fail gracefully
    shape = tuple(X.shape) if hasattr(X, 'shape') else '?'
    print(f'    X is a single tensor  shape={shape}')
    print()
    print('  ERROR: batch[0] is a single tensor, not a dict or sequence.')
    print('  This suggests all input groups are concatenated.')
    print('  Check your data config.')
    raise RuntimeError('Cannot determine batch structure — see output above.')


def collect(loader, data_config, input_groups_arg, n_events):
    all_groups = get_all_groups(data_config)
    print(f'  Input groups from YAML: {all_groups}')

    if input_groups_arg is None:
        groups_to_save = all_groups
    else:
        unknown = [g for g in input_groups_arg if g not in all_groups]
        if unknown:
            print(f'  Warning: requested groups not in YAML: {unknown}')
        groups_to_save = [g for g in input_groups_arg if g in all_groups]
        if not groups_to_save:
            print('  No valid groups to save.  Saving all groups instead.')
            groups_to_save = all_groups

    print(f'  Groups to export: {groups_to_save}')
    group_indices = {g: all_groups.index(g) for g in groups_to_save}
    print()

    # Peek at the first batch before starting the main loop
    batch_iter = iter(loader)
    try:
        first_batch = next(batch_iter)
    except StopIteration:
        print('Error: loader yielded no batches.  Check your data files.')
        sys.exit(1)

    getter = inspect_batch(first_batch)
    print()

    # Collect — include the first batch we already consumed
    collected = defaultdict(list)
    n_collected = 0

    def all_batches():
        yield first_batch
        for b in batch_iter:
            yield b

    for batch in all_batches():
        if n_collected >= n_events:
            break
        X = batch[0]
        for name, idx in group_indices.items():
            try:
                tensor = getter(X, name, idx)
            except (KeyError, IndexError) as e:
                print(f'  Error accessing group "{name}" (idx={idx}): {e}')
                print(f'  Batch[0] type={type(X)}, '
                      f'keys/len={list(X.keys()) if isinstance(X, dict) else len(X)}')
                raise
            collected[name].append(tensor.numpy())

        n_collected += 1
        if n_collected % 50 == 0:
            print(f'  Collected {n_collected} / {n_events} ...')

    print(f'  Done. Collected {n_collected} events.')

    result = {}
    for name, arrays in collected.items():
        stacked = np.concatenate(arrays, axis=0)
        result[name] = stacked
        print(f'  {name:<25}  shape={stacked.shape}  '
              f'range=[{stacked.min():.4f}, {stacked.max():.4f}]')

    return result


def main():
    args = parse_args()
    print('=== export_calibration_data.py ===\n')

    file_list = expand_files(args.data_files)
    print(f'Input files ({len(file_list)}):')
    for f in file_list[:5]:
        print(f'  {f}')
    if len(file_list) > 5:
        print(f'  ... and {len(file_list)-5} more')

    print(f'\nBuilding weaver dataloader...')
    loader, data_config = build_dataloader(
        args.data_config, file_list,
        args.num_workers, args.fetch_step,
    )

    print(f'\nCollecting up to {args.n_events} events...')
    arrays = collect(loader, data_config, args.input_groups, args.n_events)

    if not arrays:
        print('Error: nothing collected.')
        sys.exit(1)

    np.savez(args.output, **arrays)
    n = next(iter(arrays.values())).shape[0]
    print(f'\nSaved {n} events → {args.output}')

    groups = list(arrays.keys())
    print(f'\nNext step:')
    print(f'  python find_range_requirements.py \\')
    print(f'      model.onnx \\')
    print(f'      --data {args.output} \\')
    print(f'      --input-names {" ".join(groups)} \\')
    print(f'      --n-samples {n} \\')
    print(f'      --hotspot-threshold 8 \\')
    print(f'      --total-bits 16 \\')
    print(f'      --csv ranges.csv')


if __name__ == '__main__':
    main()