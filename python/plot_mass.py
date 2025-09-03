import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob


def calculate_invariant_mass(E, px, py, pz):
    """Calculate invariant mass from 4-momentum components."""
    p_squared = px**2 + py**2 + pz**2
    mass_squared = E**2 - p_squared
    # Ensure mass_squared is non-negative to avoid sqrt errors
    mass_squared = np.where(mass_squared > 0, mass_squared, 0)
    return np.sqrt(mass_squared)


if __name__ == "__main__":
    # Path to the ROOT file
    sub_dir = sys.argv[1]
    root_files = glob.glob(sub_dir + "training_log/particlenet*.root")
    tree_name = "Events"

    plt.figure(figsize=(8, 6), dpi=200)
    for file in root_files:
    # Load the ROOT 
        name = file.replace(sub_dir + "training_log/particlenet_predict_", "")
        if name == file:
            name = file.replace(sub_dir + "training_log/particlenet_predict", "")
        name = name.replace(".root", "")
        print(name)
        with uproot.open(f"{file}:{tree_name}") as tree:
            # Read 4-momentum branches
            data = tree.arrays(["score_B_E", "score_B_px", "score_B_py", "score_B_pz"])

        # Extract 4-momentum components
        B_E = data["score_B_E"]
        B_px = data["score_B_px"]
        B_py = data["score_B_py"]
        B_pz = data["score_B_pz"]

        # Calculate invariant mass
        masses = calculate_invariant_mass(B_E, B_px, B_py, B_pz)

        # Convert to numpy for plotting
        masses_np = ak.to_numpy(masses)
        avg = np.mean(masses_np)
        stdev = np.std(masses_np)

        # Plot the mass distribution
        plt.hist(masses_np, bins=50, range=(5.26, 5.3), histtype='step', label='%s: $m_B = %.3f \pm %.3f$ GeV' % (name, avg, stdev))
    
    plt.xlabel('Reconstructed Mass [GeV]')
    plt.ylabel('Events')
    plt.title('Reconstructed B Mass')
    plt.grid(True)
    plt.legend()
    plt.xlim(5.26, 5.3)
    plt.savefig(sub_dir + 'B_mass_distribution.png', transparent=True)
    plt.show()