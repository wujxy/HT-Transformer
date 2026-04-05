"""
H5 Schema Inspection Tool for JUNO CDWP Muon Track Reconstruction.

Inspects an h5 file and produces a comprehensive schema report:
- Top-level keys, shapes, dtypes
- Sample event data
- CD/WP hit distribution
- Time/charge statistics
- Truth endpoint geometry validation
"""

import sys
import numpy as np
import h5py
from loguru import logger


PMT_COPYNO_OFFSET = 50000
PMT_RADIUS = 19433.975  # mm, approximate sphere radius


def inspect_h5(h5_path: str, max_sample_events: int = 3):
    """Run full h5 schema inspection and print report."""
    logger.info(f"Opening h5 file: {h5_path}")
    f = h5py.File(h5_path, 'r')

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("H5 SCHEMA INSPECTION REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"File: {h5_path}")
    report_lines.append("")

    # --- Top-level keys ---
    report_lines.append("--- TOP-LEVEL KEYS ---")
    keys = list(f.keys())
    report_lines.append(f"Number of datasets: {len(keys)}")
    report_lines.append("")

    key_info = {}
    for key in keys:
        ds = f[key]
        info = {
            'shape': ds.shape,
            'dtype': str(ds.dtype),
            'ndim': ds.ndim,
        }
        key_info[key] = info
        report_lines.append(f"  {key:25s}  shape={str(ds.shape):20s}  dtype={ds.dtype}")

    # Determine number of events
    event_keys = [k for k in keys if f[k].ndim >= 1]
    n_events = f[event_keys[0]].shape[0] if event_keys else 0
    report_lines.append(f"\nTotal events: {n_events}")

    # --- Event-level fields ---
    scalar_keys = [k for k in keys if f[k].ndim == 1]
    array_keys = [k for k in keys if f[k].ndim == 2]

    report_lines.append("\n--- EVENT-LEVEL SCALAR FIELDS (shape: (N_events,)) ---")
    for k in scalar_keys:
        report_lines.append(f"  {k:25s}  range=[{f[k][()].min()}, {f[k][()].max()}]")

    report_lines.append("\n--- HIT-LEVEL ARRAY FIELDS (shape: (N_events, max_hits)) ---")
    for k in array_keys:
        report_lines.append(f"  {k:25s}  shape={f[k].shape}")

    # --- Sample events ---
    sample_indices = np.random.choice(n_events, size=min(max_sample_events, n_events), replace=False)
    sample_indices.sort()
    report_lines.append(f"\n--- SAMPLE EVENTS (indices: {list(sample_indices)}) ---")

    for idx in sample_indices:
        report_lines.append(f"\n  Event {idx}:")
        if 'nhits' in f:
            nhits = int(f['nhits'][idx])
            report_lines.append(f"    nhits = {nhits}")
        else:
            nhits = f[array_keys[0]].shape[1]
            report_lines.append(f"    max_hits = {nhits} (no nhits field, assuming all valid)")

        # Copyno distribution
        if 'copyno' in f:
            copyno = f['copyno'][idx]
            valid_copyno = copyno[:nhits]
            cd_count = np.sum(valid_copyno < PMT_COPYNO_OFFSET)
            wp_count = np.sum(valid_copyno >= PMT_COPYNO_OFFSET)
            report_lines.append(f"    CD hits: {cd_count},  WP hits: {wp_count}")
            report_lines.append(f"    copyno range: [{valid_copyno.min()}, {valid_copyno.max()}]")

        # Charge distribution
        if 'charge' in f:
            charge = f['charge'][idx][:nhits]
            report_lines.append(f"    charge: min={charge.min():.1f}, max={charge.max():.1f}, "
                              f"mean={charge.mean():.1f}, median={np.median(charge):.1f}")

        # Time distribution
        if 'time' in f:
            time = f['time'][idx][:nhits]
            report_lines.append(f"    time: min={time.min():.1f}, max={time.max():.1f}, "
                              f"mean={time.mean():.1f} ns")

        # Truth endpoints
        enter = np.array([f['tt_enter_x'][idx], f['tt_enter_y'][idx], f['tt_enter_z'][idx]])
        exit_pt = np.array([f['tt_exit_x'][idx], f['tt_exit_y'][idx], f['tt_exit_z'][idx]])
        enter_r = np.linalg.norm(enter)
        exit_r = np.linalg.norm(exit_pt)
        chord = np.linalg.norm(exit_pt - enter)
        ang_sep = np.degrees(np.arccos(np.clip(
            np.dot(enter, exit_pt) / (enter_r * exit_r + 1e-10), -1, 1)))

        report_lines.append(f"    enter_point: ({enter[0]:.1f}, {enter[1]:.1f}, {enter[2]:.1f})  "
                          f"||r||={enter_r:.1f} mm")
        report_lines.append(f"    exit_point:  ({exit_pt[0]:.1f}, {exit_pt[1]:.1f}, {exit_pt[2]:.1f})  "
                          f"||r||={exit_r:.1f} mm")
        report_lines.append(f"    chord length: {chord:.1f} mm")
        report_lines.append(f"    angular separation: {ang_sep:.2f} deg")

        # Validate on sphere
        if abs(enter_r - PMT_RADIUS) / PMT_RADIUS > 0.05:
            report_lines.append(f"    WARNING: enter point radius {enter_r:.1f} differs from "
                              f"PMT_RADIUS {PMT_RADIUS:.1f} by >5%")
        if abs(exit_r - PMT_RADIUS) / PMT_RADIUS > 0.05:
            report_lines.append(f"    WARNING: exit point radius {exit_r:.1f} differs from "
                              f"PMT_RADIUS {PMT_RADIUS:.1f} by >5%")

    # --- Global statistics ---
    report_lines.append("\n--- GLOBAL STATISTICS ---")
    if 'nhits' in f:
        all_nhits = f['nhits'][()]
        report_lines.append(f"  nhits: min={all_nhits.min()}, max={all_nhits.max()}, "
                          f"mean={all_nhits.mean():.0f}")

    if 'copyno' in f and 'nhits' in f:
        total_cd = 0
        total_wp = 0
        all_times = []
        all_charges = []
        for i in range(n_events):
            nh = int(f['nhits'][i])
            cn = f['copyno'][i][:nh]
            total_cd += np.sum(cn < PMT_COPYNO_OFFSET)
            total_wp += np.sum(cn >= PMT_COPYNO_OFFSET)
            all_times.extend(f['time'][i][:nh].tolist())
            all_charges.extend(f['charge'][i][:nh].tolist())

        all_times = np.array(all_times)
        all_charges = np.array(all_charges)
        report_lines.append(f"  Total CD hits: {total_cd},  WP hits: {total_wp}")
        report_lines.append(f"  Time range (all): [{all_times.min():.1f}, {all_times.max():.1f}] ns")
        report_lines.append(f"  Time p99: {np.percentile(all_times, 99):.1f} ns")
        report_lines.append(f"  Charge range (all): [{all_charges.min():.1f}, {all_charges.max():.1f}] PE")
        report_lines.append(f"  Charge p99: {np.percentile(all_charges, 99):.1f} PE")

    # --- H5 key mapping suggestion ---
    report_lines.append("\n--- SUGGESTED h5_key_map FOR CONFIG ---")
    report_lines.append("h5_key_map:")
    key_map = {
        'copyno': 'copyno' if 'copyno' in f else None,
        'charge': 'charge' if 'charge' in f else None,
        'time': 'time' if 'time' in f else None,
        'nhits': 'nhits' if 'nhits' in f else None,
        'enter_x': 'tt_enter_x' if 'tt_enter_x' in f else None,
        'enter_y': 'tt_enter_y' if 'tt_enter_y' in f else None,
        'enter_z': 'tt_enter_z' if 'tt_enter_z' in f else None,
        'exit_x': 'tt_exit_x' if 'tt_exit_x' in f else None,
        'exit_y': 'tt_exit_y' if 'tt_exit_y' in f else None,
        'exit_z': 'tt_exit_z' if 'tt_exit_z' in f else None,
        'track_time': 'track_time' if 'track_time' in f else None,
    }
    for logical, actual in key_map.items():
        if actual is not None:
            report_lines.append(f"  {logical}: \"{actual}\"")
        else:
            report_lines.append(f"  {logical}: NOT FOUND")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    f.close()

    report_text = "\n".join(report_lines)
    print(report_text)
    return report_text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python InspectH5.py <h5_path> [max_sample_events]")
        sys.exit(1)
    h5_path = sys.argv[1]
    max_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    inspect_h5(h5_path, max_sample)
