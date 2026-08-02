#!/usr/bin/env python3
"""
LX521 Polar Analysis Pipeline

This script orchestrates the entire analysis workflow:
1. Connects to REW API (launches if needed).
2. Loads measurement files (.mdat).
3. Applies Time Gating (remove room reflections).
4. Applies optional smoothing (via config.DEFAULT_SMOOTHING).
5. Saves processed data to HDF5.
6. Generates Visualization Reports.

Author: Andres Torrubia
Date: 2025-11-23
"""

import sys
import argparse
import os
from pathlib import Path

import config
from polar_data_loader import PolarDataLoader
from generate_visualizations import PolarResponseVisualizer


def apply_measurement_metadata_overrides(data, overrides):
    """Apply configured measurement metadata to loaded driver angle records."""

    for driver_name, driver_override in (overrides or {}).items():
        driver_data = data.get(driver_name)
        if not driver_data:
            print(f"Warning: metadata override skipped; driver '{driver_name}' was not loaded")
            continue

        note = str(driver_override.get("notes", "")).strip()
        extra_attrs = {
            key: value
            for key, value in driver_override.items()
            if key != "notes"
        }
        for side_key in ("angles", "rear_angles"):
            for angle_data in driver_data.get(side_key, {}).values():
                metadata = angle_data.setdefault("metadata", {})
                if note:
                    existing = str(metadata.get("notes", "")).strip()
                    if note not in existing:
                        metadata["notes"] = f"{existing}\n{note}" if existing else note
                metadata.update(extra_attrs)


def run_pipeline(args):
    # Get measurement set configuration
    mset_name = args.measurement_set
    if mset_name not in config.MEASUREMENT_SETS:
        print(f"Error: Unknown measurement set '{mset_name}'")
        print(f"Available sets: {', '.join(config.MEASUREMENT_SETS.keys())}")
        sys.exit(1)

    mset = config.MEASUREMENT_SETS[mset_name]
    has_rear = mset["has_rear"]
    hdf5_override = args.hdf5_output or os.environ.get("HDF5_PATH")
    hdf5_path = Path(hdf5_override) if hdf5_override else config.DATA_DIR / mset["hdf5_file"]
    if (
        mset_name == "andres"
        and not args.skip_loading
        and hdf5_path.name == "polar_data_andres_early_peak_legacy.h5"
        and not args.allow_published_parity_hdf5_overwrite
    ):
        raise SystemExit(
            "Refusing to regenerate the Andres published-parity HDF5 from the current loader. "
            "Use --skip-loading to visualize the canonical published data, write a diagnostic HDF5 "
            "with --hdf5-output, or pass --allow-published-parity-hdf5-overwrite only when the "
            "published explorer is regenerated from that same HDF5 in the same commit."
        )
    output_dir = mset["output_dir"]
    static_plots_dir = output_dir / "static_plots"
    interactive_plots_dir = output_dir / "interactive"
    peak_policy = args.peak_policy or mset.get("direct_ir_peak_policy") or config.DIRECT_IR_PEAK_POLICY

    # Check if this is a multi-source measurement set
    sources = mset.get("sources")
    data_dir = mset.get("path")
    pattern_type = mset.get("pattern_type")
    angles = mset.get("angles")
    gate_left_ms = mset.get("gate_left_ms", config.GATE_LEFT_MS)
    gate_right_ms = mset.get("gate_right_ms", config.GATE_RIGHT_MS)
    driver_list = None
    if args.drivers:
        driver_list = [driver.strip() for driver in args.drivers.split(",") if driver.strip()]

    print("=" * 60)
    print("LX521 POLAR ANALYSIS PIPELINE")
    print("=" * 60)
    print(f"Measurement set: {mset_name}")
    if sources:
        print(f"Sources:         {len(sources)} directories")
        for src in sources:
            print(f"                 - {src['path']} ({src['pattern_type']})")
    else:
        print(f"Data directory:  {data_dir}")
    print(f"Output file:     {hdf5_path}")
    print(f"Output plots:    {output_dir}")
    print(f"Peak policy:     {peak_policy}")
    if args.allow_unsafe_strongest_peak_policy:
        config.ALLOW_UNSAFE_STRONGEST_IR_PEAK_POLICY = True
        print("Peak policy note: strongest allowed for legacy diagnostic output only")

    # 1. Load & Process Data
    if not args.skip_loading:
        print("\n[STEP 1] Loading and Processing Data...")
        smoothing_val = 0 if args.no_smoothing else mset.get("smoothing", config.DEFAULT_SMOOTHING)

        try:
            if sources:
                # Multi-source measurement set: load from each source and merge
                all_data = {}
                for src in sources:
                    src_path = src["path"]
                    src_pattern = src["pattern_type"]
                    print(f"\n  Loading from {src_path} (pattern: {src_pattern})...")

                    loader = PolarDataLoader(
                        data_directory=str(src_path),
                        pattern_type=src_pattern,
                        direct_ir_peak_policy=peak_policy,
                        driver_name_aliases=src.get("driver_name_aliases"),
                    )
                    src_data = loader.load_all_drivers(
                        driver_list=driver_list,
                        angles=angles,
                        smoothing=smoothing_val,
                        gate_left_ms=gate_left_ms,
                        gate_right_ms=gate_right_ms,
                        include_rear=has_rear
                    )
                    # Check for collisions before merging
                    collisions = set(all_data.keys()) & set(src_data.keys())
                    if collisions:
                        raise ValueError(
                            f"Driver name collision detected: {collisions}\n"
                            f"Source '{src_path}' contains drivers that already exist.\n"
                            f"Rename drivers or use separate measurement sets."
                        )
                    # Merge into all_data
                    all_data.update(src_data)
                    print(f"  Loaded drivers: {list(src_data.keys())}")

                data = all_data
                # Use last loader instance for saving (just need HDF5 methods)
            else:
                # Single-source measurement set
                loader = PolarDataLoader(
                    data_directory=str(data_dir),
                    pattern_type=pattern_type,
                    direct_ir_peak_policy=peak_policy,
                    driver_name_aliases=mset.get("driver_name_aliases"),
                )
                data = loader.load_all_drivers(
                    driver_list=driver_list,
                    angles=angles,
                    smoothing=smoothing_val,
                    gate_left_ms=gate_left_ms,
                    gate_right_ms=gate_right_ms,
                    include_rear=has_rear
                )

            apply_measurement_metadata_overrides(
                data,
                mset.get("measurement_metadata_overrides"),
            )

            # Save
            print(f"\n[STEP 2] Saving to {hdf5_path}...")
            loader.save_to_hdf5(data, str(hdf5_path),
                               gate_left_ms=gate_left_ms,
                               gate_right_ms=gate_right_ms,
                               smoothing=smoothing_val)

        except RuntimeError as e:
            print(f"\nFATAL ERROR: {e}")
            sys.exit(1)

    else:
        print("\n[STEP 1 & 2] Skipped loading/processing (using existing HDF5).")

    # 2. Generate Visualizations
    if not args.skip_viz:
        print("\n[STEP 3] Generating Visualizations...")
        if not hdf5_path.exists():
            print(f"Error: Data file {hdf5_path} not found. Run without --skip-loading first.")
            sys.exit(1)

        single_angle = bool(mset.get("single_angle"))
        viz = PolarResponseVisualizer(
            str(hdf5_path),
            static_plots_dir=static_plots_dir,
            interactive_plots_dir=interactive_plots_dir,
            require_directivity=not single_angle,
        )
        if single_angle:
            # On-axis-only sets have no polar coverage, so directivity,
            # beamwidth, contour and polar plots would be meaningless.
            print("Single-angle set: generating frequency response outputs only.")
            viz.plot_frequency_response_explorer()
            viz.generate_measurement_summary_html()
        else:
            viz.generate_all_plots()
    else:
        print("\n[STEP 3] Skipped visualizations.")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Outputs located in: {static_plots_dir}")
    print(f"Interactive plots:  {interactive_plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LX521 Polar Analysis Pipeline")
    parser.add_argument("-m", "--measurement-set", default=config.DEFAULT_MEASUREMENT_SET,
                        choices=list(config.MEASUREMENT_SETS.keys()),
                        help=f"Which measurement set to process (default: {config.DEFAULT_MEASUREMENT_SET})")
    parser.add_argument("--skip-loading", action="store_true", help="Skip REW loading/processing, use existing HDF5")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization generation")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable smoothing (raw data)")
    parser.add_argument("--hdf5-output", type=Path, default=None, help="Override the processed HDF5 output path")
    parser.add_argument(
        "--allow-published-parity-hdf5-overwrite",
        action="store_true",
        help=(
            "Permit overwriting polar_data_andres_early_peak_legacy.h5. Use only when regenerating "
            "the published polar explorer from the same HDF5 in the same commit."
        ),
    )
    parser.add_argument("--drivers", default=None, help="Comma-separated driver names to load")
    parser.add_argument(
        "--peak-policy",
        default=None,
        choices=["strongest", "first-strong", "ir-start"],
        help=(
            "IR timing selector used before gating. Defaults to the measurement-set "
            "policy, then config.DIRECT_IR_PEAK_POLICY. 'ir-start' uses REW's stored "
            "timeOfIRStartSeconds/onset as the window reference. 'strongest' and "
            "'first-strong' are legacy diagnostics for no-timing-ref captures because "
            "high-angle reflections can be stronger than the direct arrival."
        ),
    )
    parser.add_argument(
        "--allow-unsafe-strongest-peak-policy",
        action="store_true",
        help="Permit legacy strongest-peak HDF5 regeneration; outputs are marked diagnostic-only.",
    )

    args = parser.parse_args()
    run_pipeline(args)
