import numpy as np
import sys
import uproot
import awkward as ak


if __name__ == "__main__":
    sub_dir = sys.argv[1]
    file_name = sys.argv[2]

    with np.load("score.npz") as file:
        score = file["score"]
        print(score.shape)

    with np.load(sub_dir + "data/part_mask_truth_" + file_name + ".npz") as file:
        mask_truth = file["mask"]
        print(mask_truth.shape)

    mask = np.argmax(score, axis=-1)
    print(mask.shape)
    print(np.sum(np.abs(mask - mask_truth)) / len(mask))

    # Open existing file
    file = uproot.open(sub_dir + "data/" + file_name + ".root")
    tree = file["Events"]
    data = tree.arrays(library="np")
    data["pred_mask"] = mask

    # Write out to a new ROOT file
    with uproot.recreate(sub_dir + "data/" + file_name + ".root") as f:
        f["Events"] = data