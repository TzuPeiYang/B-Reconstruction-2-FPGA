import numpy as np
import sys
import uproot
import awkward as ak


if __name__ == "__main__":
    sub_dir = sys.argv[1]
    file_name = sys.argv[2]

    with np.load(sub_dir + "data/b_type_truth_" + file_name + ".npz") as file:
        b_type_truth = file["btype"]
        print(b_type_truth.shape)

    with uproot.open(sub_dir + "with_partial_vertex/training_log/particlenet_predict_complete_add_linear.root") as file:
        tree = file["Events"]
        data = tree.arrays(library="np")
        b_type_pred = data["score_B_type"]
        
    for i in range(len(b_type_pred)):
        if b_type_pred[i] < 0.5:
            b_type_pred[i] = 1
        else:
            b_type_pred[i] = 0

    print(1 - np.abs(b_type_pred - b_type_truth).sum() / len(b_type_pred))

    