"""
Export pixel-hit cluster features from an edm4hep ROOT file to HDF5.

Each cluster (TrackerHitPlane) produces one row.  Fixed-size cluster-level
features are stored as regular 1-D datasets; per-pixel arrays (energy,
time, x, y, z) are variable-length so every cluster keeps *all* its raw
hits without truncation.

Reads the ROOT TTree directly (no podio Python reader) so that files with
broken or unresolvable Link references (common in BIB overlay samples)
can be processed without segfaults.  Relation branches
(_<RelCol>_from/to) are used to build the hit→sim-hit mapping, and the
SimTrackerHit quality bit 31 (isOverlay) provides truth labels.

Usage (inside the MuC software container):
    python create_hdf5_pixelhits.py -i input1.edm4hep.root input2.edm4hep.root \
        -o output.h5 [--max-r 32] [--collections VXDBarrelHits]
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import ROOT
import h5py
import numpy as np


## helpers (mirrors PixelHitBDTTestAlg.cpp) ##########################

OVERLAY_BIT = 1 << 31


def get_rms(cluster_pos, positions, energies):
    """Energy-weighted variance of *positions* around *cluster_pos*."""
    if len(positions) == 0:
        return 0.0
    diffs = positions - cluster_pos
    num = np.sum(energies * diffs**2)
    den = np.sum(energies)
    return float(num / den) if den > 0 else 0.0


def get_cov(cx, cy, px, py, energies):
    """Energy-weighted covariance between x and y deviations."""
    if len(px) == 0:
        return 0.0
    num = np.sum(energies * (cx - px) * (cy - py))
    den = np.sum(energies)
    return float(num / den) if den > 0 else 0.0


def get_skew(cluster_pos, positions, energies):
    """Energy-weighted skewness around *cluster_pos*."""
    rms = get_rms(cluster_pos, positions, energies)
    if rms <= 0:
        return 0.0
    rms_term = math.sqrt(rms) ** 3
    if rms_term <= 0:
        return 0.0
    diffs = positions - cluster_pos
    num = np.sum(energies * diffs**3)
    den = np.sum(energies) * rms_term
    return float(num / den) if den > 0 else 0.0


def cluster_size(positions_x, positions_y):
    if len(positions_x) == 0:
        return 0.0, 0.0
    sx = float(positions_x.max() - positions_x.min()) + 1.0
    sy = float(positions_y.max() - positions_y.min()) + 1.0
    return sx, sy


def eigen_shape(rms_x, rms_y, cov_xy):
    """Aspect ratio and eccentricity from the 2×2 covariance eigenvalues."""
    trace = rms_x + rms_y
    det = rms_x * rms_y - cov_xy**2
    disc = max(0.0, trace**2 - 4 * det)
    sqrt_disc = math.sqrt(disc)
    l1 = 0.5 * (trace + sqrt_disc)
    l2 = 0.5 * (trace - sqrt_disc)

    if l1 >= l2:
        aspect = math.sqrt(l1 / l2) if l2 > 0 else 0.0
        ecc = math.sqrt(max(0.0, 1 - l2 / l1)) if l1 > 0 else 0.0
    else:
        aspect = math.sqrt(l2 / l1) if l1 > 0 else 0.0
        ecc = math.sqrt(max(0.0, 1 - l1 / l2)) if l2 > 0 else 0.0
    return aspect, ecc


## main ##############################################################

def options():
    p = argparse.ArgumentParser(
        description="Export pixel-hit features from edm4hep to HDF5 with variable-length raw hits."
    )
    p.add_argument("-i", required=True, nargs="+", type=Path,
                    help="One or more input edm4hep ROOT files")
    p.add_argument("-o", required=True, type=Path, help="Output HDF5 file")
    p.add_argument(
        "--max-r", default=32.0, type=float,
        help="Maximum transverse radius to select hits (mm). Set 0 to disable.",
    )
    p.add_argument(
        "--collections", nargs="+", default=["VXDBarrelHits"],
        help="TrackerHitPlane collection name(s) to process",
    )
    p.add_argument(
        "--relation-collection", default="VXDBarrelRawHitsRelations",
        help="Link collection name (raw hit relations). "
             "The TTree branches _{name}_from are read directly.",
    )
    p.add_argument(
        "--simhit-collection", default="VertexBarrelHits_Passed",
        help="SimTrackerHit collection containing the raw pixel hits",
    )
    p.add_argument("--no-truth", action="store_true", help="Skip truth label extraction")
    return p.parse_args()


def process_file(input_path, args):
    rfile = ROOT.TFile.Open(str(input_path))
    if not rfile or rfile.IsZombie():
        raise RuntimeError(f"Cannot open {input_path}")
    tree = rfile.Get("events")
    if not tree:
        raise RuntimeError(f"No 'events' TTree in {input_path}")

    n_events = tree.GetEntries()
    rel_from_branch = f"_{args.relation_collection}_from"

    # Verify required branches exist
    for bname in [args.simhit_collection, rel_from_branch]:
        if not tree.GetBranch(bname):
            raise RuntimeError(f"Branch '{bname}' not found in TTree")

    acc = defaultdict(list)
    ragged_raw_energy = []
    ragged_raw_time = []
    ragged_raw_x = []
    ragged_raw_y = []
    ragged_raw_z = []

    has_truth = not args.no_truth
    truth_labels = []

    for i_event in range(n_events):
        tree.GetEntry(i_event)

        for col_name in args.collections:
            hits = getattr(tree, col_name, None)
            if hits is None:
                continue

            sim_hits = getattr(tree, args.simhit_collection)
            rel_from = getattr(tree, rel_from_branch)

            # Build hit_index → [relation-positions] (= sim-hit indices)
            # _from[i].index is the VXDBarrelHits entry; position i is the
            # matching entry in VertexBarrelHits_Passed (1:1 positional).
            hit_to_sim = defaultdict(list)
            for i_rel in range(rel_from.size()):
                hit_to_sim[rel_from[i_rel].index].append(i_rel)

            for i_hit in range(hits.size()):
                hit = hits[i_hit]
                x = hit.position.x
                y = hit.position.y
                z = hit.position.z
                r = math.sqrt(x * x + y * y)

                if args.max_r > 0 and r > args.max_r:
                    continue

                sim_indices = hit_to_sim.get(i_hit, [])

                raw_e = np.array(
                    [sim_hits[j].eDep for j in sim_indices], dtype=np.float32
                )
                raw_t = np.array(
                    [sim_hits[j].time for j in sim_indices], dtype=np.float32
                )
                raw_px = np.array(
                    [sim_hits[j].position.x for j in sim_indices], dtype=np.float32
                )
                raw_py = np.array(
                    [sim_hits[j].position.y for j in sim_indices], dtype=np.float32
                )
                raw_pz = np.array(
                    [sim_hits[j].position.z for j in sim_indices], dtype=np.float32
                )

                npix = len(sim_indices)
                sx, sy = cluster_size(raw_px, raw_py)
                rms_x = get_rms(x, raw_px, raw_e)
                rms_y = get_rms(y, raw_py, raw_e)
                skew_x = get_skew(x, raw_px, raw_e)
                skew_y = get_skew(y, raw_py, raw_e)
                cov_xy = get_cov(x, y, raw_px, raw_py, raw_e)
                aspect, ecc = eigen_shape(rms_x, rms_y, cov_xy)

                acc["cluster_energy"].append(hit.eDep)
                acc["cluster_time"].append(hit.time)
                acc["cluster_x"].append(x)
                acc["cluster_y"].append(y)
                acc["cluster_z"].append(z)
                acc["cluster_r"].append(r)
                acc["incident_angle"].append(math.atan2(r, z))
                acc["cluster_size_tot"].append(npix)
                acc["cluster_size_x"].append(sx)
                acc["cluster_size_y"].append(sy)
                acc["cluster_rms_x"].append(rms_x)
                acc["cluster_rms_y"].append(rms_y)
                acc["cluster_skew_x"].append(skew_x)
                acc["cluster_skew_y"].append(skew_y)
                acc["cluster_aspect"].append(aspect)
                acc["cluster_ecc"].append(ecc)
                acc["event_number"].append(i_event)

                ragged_raw_energy.append(raw_e)
                ragged_raw_time.append(raw_t)
                ragged_raw_x.append(raw_px)
                ragged_raw_y.append(raw_py)
                ragged_raw_z.append(raw_pz)

                # Truth label via SimTrackerHit quality bit 31 (isOverlay).
                # Matches PixelHitBDTTestAlg logic: label = 1 (signal) if
                # ANY contributing raw hit is non-overlay, else 0 (BIB).
                if has_truth:
                    any_signal = any(
                        not (sim_hits[j].quality & OVERLAY_BIT)
                        for j in sim_indices
                    )
                    truth_labels.append(1 if any_signal else 0)

        if (i_event + 1) % 10 == 0:
            n = len(acc["cluster_energy"])
            print(f"  Processed {i_event + 1}/{n_events} events, {n} clusters so far")

    rfile.Close()

    n_clusters = len(acc["cluster_energy"])
    print(f"Total clusters extracted: {n_clusters}")
    return acc, ragged_raw_energy, ragged_raw_time, ragged_raw_x, ragged_raw_y, ragged_raw_z, truth_labels, has_truth


def write_hdf5(outpath, acc, raw_e, raw_t, raw_x, raw_y, raw_z, truth_labels, has_truth):
    n = len(acc["cluster_energy"])
    if n == 0:
        print("No clusters to write.")
        return

    vlen_f32 = h5py.special_dtype(vlen=np.float32)

    with h5py.File(str(outpath), "w") as f:
        ## cluster-level (fixed size) ##
        grp = f.create_group("clusters")
        for key, vals in acc.items():
            dtype = np.int32 if key in ("cluster_size_tot", "event_number") else np.float32
            grp.create_dataset(key, data=np.array(vals, dtype=dtype))

        ## raw-hit level (variable length) ##
        raw_grp = f.create_group("raw_hits")
        for name, data in [
            ("energy", raw_e),
            ("time", raw_t),
            ("x", raw_x),
            ("y", raw_y),
            ("z", raw_z),
        ]:
            ds = raw_grp.create_dataset(name, shape=(n,), dtype=vlen_f32)
            for i, arr in enumerate(data):
                ds[i] = arr

        ## truth label ##
        if has_truth and len(truth_labels) == n:
            f.create_dataset("truth_label", data=np.array(truth_labels, dtype=np.int8))

    print(f"Wrote {n} clusters to {outpath}")


def main():
    args = options()

    print(f"Input:  {args.i}  ({len(args.i)} file(s))")
    print(f"Output: {args.o}")
    print(f"Collections: {args.collections}")
    print(f"Max R cut: {args.max_r} mm {'(disabled)' if args.max_r <= 0 else ''}")

    # Accumulate results across all input files
    merged_acc = defaultdict(list)
    merged_raw_e, merged_raw_t, merged_raw_x, merged_raw_y, merged_raw_z = [], [], [], [], []
    merged_truth = []
    has_truth = not args.no_truth

    for i_file, input_path in enumerate(args.i):
        print(f"\n--- File {i_file + 1}/{len(args.i)}: {input_path} ---")
        acc, raw_e, raw_t, raw_x, raw_y, raw_z, truth_labels, _ = process_file(input_path, args)
        for k, v in acc.items():
            merged_acc[k].extend(v)
        merged_raw_e.extend(raw_e)
        merged_raw_t.extend(raw_t)
        merged_raw_x.extend(raw_x)
        merged_raw_y.extend(raw_y)
        merged_raw_z.extend(raw_z)
        merged_truth.extend(truth_labels)

    print(f"\nTotal clusters from all files: {len(merged_acc['cluster_energy'])}")
    write_hdf5(args.o, merged_acc, merged_raw_e, merged_raw_t,
               merged_raw_x, merged_raw_y, merged_raw_z, merged_truth, has_truth)


if __name__ == "__main__":
    main()
