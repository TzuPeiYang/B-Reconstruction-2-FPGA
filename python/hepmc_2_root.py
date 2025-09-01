import pyhepmc
import numpy as np
import awkward as ak
import uproot
import sys


def get_b_ancestor(particle):
    vtx = particle.production_vertex
    while vtx:
        for parent in vtx.particles_in:
            if abs(parent.pid) == 521 or parent.pid == 300553:
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
    part_pdgid = []
    part_mask = []

    B_px, B_py, B_pz, B_E = [], [], [], []
    Bbar_px, Bbar_py, Bbar_pz, Bbar_E = [], [], [], []

    event_count = 0
    max_part = 0
    while not reader.failed():
        reader.read_event(event)
        if reader.failed():
            continue

        # Per-event lists
        ev_px, ev_py, ev_pz, ev_E = [], [], [], []
        ev_vx, ev_vy, ev_vz = [], [], []
        ev_pdgid = []
        ev_mask = []

        bpx = bpy = bpz = be = None
        bbarpx = bbarpy = bbarpz = bbare = None

        # total_px, total_py, total_pz, total_E = 0, 0, 0, 0

        for p in event.particles:
            if p.status == 1:  # Final-state particle
                ''' ===========        complete B-B-          =========== '''
                b_ancestor = get_b_ancestor(p)
                select =  b_ancestor != 300553

                ''' =========== complete B+ and incomplete B- =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select =  b_ancestor != 300553 and ((b_ancestor == -521 and np.random.rand() > 0.1 and abs(p.pid) not in [12, 14, 16]) or (b_ancestor == 521))

                ''' ===========         complete B+           =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select = p.pid != -521 and b_ancestor != 300553

                ''' ===========        incomplete B+          =========== '''
                # b_ancestor = get_b_ancestor(p)
                # select = b_ancestor not in [-521, 300553] and np.random.rand() > 0.1 and abs(p.pid) not in [12, 14, 16]

                if select:
                    mom = p.momentum
                    ev_px.append(mom.px)
                    ev_py.append(mom.py)
                    ev_pz.append(mom.pz)
                    ev_E.append(mom.e)
                    ev_pdgid.append(p.pid)

                    if b_ancestor == -521:
                        ev_mask.append(0)
                    else:
                        ev_mask.append(1)

                    # total_px += mom.px
                    # total_py += mom.py
                    # total_pz += mom.pz
                    # total_E += mom.e

                    vtx = p.production_vertex
                    if vtx:
                        ev_vx.append(vtx.position.x)
                        ev_vy.append(vtx.position.y)
                        ev_vz.append(vtx.position.z)
                    else:
                        ev_vx.append(0.0)
                        ev_vy.append(0.0)
                        ev_vz.append(0.0)

            if p.pid == 521 and bpx is None:
                mom = p.momentum
                bpx = mom.px
                bpy = mom.py
                bpz = mom.pz
                be = mom.e

            if p.pid == -521 and bbarpx is None:
                mom = p.momentum
                bbarpx = mom.px
                bbarpy = mom.py
                bbarpz = mom.pz
                bbare = mom.e

        # print(total_px, total_py, total_pz, total_E)

        # Add this event's info
        part_px.append(ev_px)
        part_py.append(ev_py)
        part_pz.append(ev_pz)
        part_E.append(ev_E)
        part_vx.append(ev_vx)
        part_vy.append(ev_vy)
        part_vz.append(ev_vz)
        part_pdgid.append(ev_pdgid)
        part_mask.append(ev_mask)

        B_px.append(bpx if bpx is not None else 0)
        B_py.append(bpy if bpy is not None else 0)
        B_pz.append(bpz if bpz is not None else 0)
        B_E.append(be if be is not None else 0)

        Bbar_px.append(bbarpx if bbarpx is not None else 0)
        Bbar_py.append(bbarpy if bbarpy is not None else 0)
        Bbar_pz.append(bbarpz if bbarpz is not None else 0)
        Bbar_E.append(bbare if bbare is not None else 0)

        event_count += 1
        if len(ev_px) > max_part:
            max_part = len(ev_px)

    # Convert to awkward arrays
    length = 52
    part_mask = np.array([xi + [0] * (length - len(xi)) for xi in part_mask], dtype=np.float32)
    np.savez(sub_dir + "data/part_mask_truth_" + filename + ".npz", mask=part_mask)
    output = {
        "Part_px": ak.Array(part_px),
        "Part_py": ak.Array(part_py),
        "Part_pz": ak.Array(part_pz),
        "Part_E":  ak.Array(part_E),
        "Part_vx": ak.Array(part_vx),
        "Part_vy": ak.Array(part_vy),
        "Part_vz": ak.Array(part_vz),
        "pdg_id":  ak.Array(part_pdgid),
        "Part_mask": part_mask,
        "B_px": np.array(B_px, dtype=np.float32),
        "B_py": np.array(B_py, dtype=np.float32),
        "B_pz": np.array(B_pz, dtype=np.float32),
        "B_E":  np.array(B_E,  dtype=np.float32),
        "Bbar_px": np.array(Bbar_px, dtype=np.float32),
        "Bbar_py": np.array(Bbar_py, dtype=np.float32),
        "Bbar_pz": np.array(Bbar_pz, dtype=np.float32),
        "Bbar_E":  np.array(Bbar_E,  dtype=np.float32),
    }

    # Write with uproot
    with uproot.recreate(output_file) as f:
        f["Events"] = output

    print(f"Wrote {event_count} events to {output_file}")
    print("Maximum number of particles in one event: %d" % max_part)
