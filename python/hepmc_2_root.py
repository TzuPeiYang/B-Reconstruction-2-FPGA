import pyhepmc
import numpy as np
import awkward as ak
import uproot
import sys


def get_b_ancestor(particle):
    vtx = particle.production_vertex
    while vtx:
        for parent in vtx.particles_in:
            if abs(parent.pid) in [511, 521] or parent.pid == 300553:
                return parent.pid
            vtx = parent.production_vertex
    return None


if __name__ == "__main__":
    sub_dir = sys.argv[1]
    filename = sys.argv[2]
    # Input/output
    input_file = "data/" + filename + ".hepmc3"
    output_file = sub_dir + "data/" + filename + ".root"

    # Set up HepMC3 reader
    reader = pyhepmc.io.ReaderAscii(input_file)
    event = pyhepmc.GenEvent()

    # Lists to hold all events
    part_px, part_py, part_pz, part_E = [], [], [], []
    part_vx, part_vy, part_vz = [], [], []
    part_eta, part_phi = [], []
    part_pdgid = []
    part_mask = []
    
    B_type = []

    n_nu = []

    B_px, B_py, B_pz, B_E = [], [], [], []

    decay_products = []

    event_count = 0
    max_part = 0
    b_found = 0
    while not reader.failed():
        reader.read_event(event)
        if reader.failed():
            continue

        # Per-event lists
        ev_px, ev_py, ev_pz, ev_E = [], [], [], []
        ev_vx, ev_vy, ev_vz = [], [], []
        ev_eta, ev_phi = [], []
        ev_pdgid = []
        ev_mask = []
        ev_n_nu = 0

        bpx = bpy = bpz = be = None

        total_px = 0
        total_py = 0
        total_pz = 0
        total_E = 0

        # total_px, total_py, total_pz, total_E = 0, 0, 0, 0

        for p in event.particles:
            if p.status == 1:  # Final-state particle

                if p.pid not in decay_products:
                    decay_products.append(p.pid)                
                
                ''' ===========        complete B+B-          =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select =  b_ancestor != 300553

                ''' ===========       incomplete B+B-         =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select =  b_ancestor != 300553 and (np.random.rand() > 0.1 and abs(p.pid) not in [12, 14, 16])

                ''' =========== complete B+ and incomplete B- =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select =  b_ancestor != 300553 and ((b_ancestor == -521 and np.random.rand() > 0.1 and abs(p.pid) not in [12, 14, 16]) or (b_ancestor == 521))

                ''' ===========         complete B+ (B0)           =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select = b_ancestor not in [300553, -511, -521]

                ''' ===========        incomplete B+          =========== '''
                b_ancestor = get_b_ancestor(p)
                select = b_ancestor not in [-511, -521, 300553] and np.random.rand() > 0.1 and abs(p.pid) not in [12, 14, 16]

                if abs(p.pid) in [12, 14, 16]:
                    ev_n_nu += 1
                if select:
                    mom = p.momentum
                    ev_px.append(mom.px)
                    ev_py.append(mom.py)
                    ev_pz.append(mom.pz)
                    ev_E.append(mom.e)
                    ev_pdgid.append(p.pid)
                    
                    p_abs = np.sqrt(mom.px ** 2 + mom.py ** 2 + mom.pz ** 2)
                    denom = np.clip(p_abs - np.abs(mom.pz), 1e-15, None)
                    numer = p_abs + np.abs(mom.pz)

                    ev_phi.append(np.arctan2(mom.py, mom.px))
                    ev_eta.append(np.log(numer / denom) / 2)

                    if b_ancestor in [-511, -521]:
                        ev_mask.append(0)
                    else:
                        ev_mask.append(1)

                    total_px += mom.px
                    total_py += mom.py
                    total_pz += mom.pz
                    total_E += mom.e

                    vtx = p.production_vertex
                    charged = abs(p.pid) in [211, 11, 321, 13, 321, 2212]
                    if vtx and charged:
                        ev_vx.append(vtx.position.x)
                        ev_vy.append(vtx.position.y)
                        ev_vz.append(vtx.position.z)
                    else:
                        ev_vx.append(0.0)
                        ev_vy.append(0.0)
                        ev_vz.append(0.0)

            if p.pid in [511, 521] and bpx is None:
                if p.pid == 511:
                    ev_b_type = 0
                else:
                    ev_b_type = 1
                mom = p.momentum
                bpx = mom.px
                bpy = mom.py
                bpz = mom.pz
                be = mom.e
                b_found += 1

        # print(total_px - bpx, total_py - bpy, total_pz - bpz, total_E - be)

        # Add this event's info
        if bpx is not None:
            part_px.append(ev_px)
            part_py.append(ev_py)
            part_pz.append(ev_pz)
            part_E.append(ev_E)
            part_vx.append(ev_vx)
            part_vy.append(ev_vy)
            part_vz.append(ev_vz)
            part_eta.append(ev_eta)
            part_phi.append(ev_phi)
            part_pdgid.append(ev_pdgid)
            part_mask.append(ev_mask)

            B_type.append(ev_b_type)

            n_nu.append(ev_n_nu)

            B_px.append(bpx if bpx is not None else 0)
            B_py.append(bpy if bpy is not None else 0)
            B_pz.append(bpz if bpz is not None else 0)
            B_E.append(be if be is not None else 0)

            event_count += 1
            if len(ev_px) > max_part:
                max_part = len(ev_px)

    # Convert to awkward arrays
    print(b_found)
    print("Decay products: ", decay_products)
    length = 52
    part_mask = np.array([xi + [0] * (length - len(xi)) for xi in part_mask], dtype=np.float32)
    np.savez(sub_dir + "data/part_mask_truth_" + filename + ".npz", mask=part_mask)
    np.savez(sub_dir + "data/b_type_truth_" + filename + ".npz", btype=B_type)
    output = {
        "Part_px": ak.Array(part_px),
        "Part_py": ak.Array(part_py),
        "Part_pz": ak.Array(part_pz),
        "Part_E":  ak.Array(part_E),
        "Part_vx": ak.Array(part_vx),
        "Part_vy": ak.Array(part_vy),
        "Part_vz": ak.Array(part_vz),
        "Part_eta": ak.Array(part_eta),
        "Part_phi": ak.Array(part_phi),
        "pdg_id":  ak.Array(part_pdgid),
        "Part_mask": part_mask,
        "B_px": np.array(B_px, dtype=np.float32),
        "B_py": np.array(B_py, dtype=np.float32),
        "B_pz": np.array(B_pz, dtype=np.float32),
        "B_E":  np.array(B_E,  dtype=np.float32),
        "n_nu": np.array(n_nu, dtype=np.int32),
        "B_type": np.array(B_type, dtype=np.int32),
    }

    # Write with uproot
    with uproot.recreate(output_file) as f:
        f["Events"] = output

    print(f"Wrote {event_count} events to {output_file}")
    print("Maximum number of particles in one event: %d" % max_part)
