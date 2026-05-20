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
from pathlib import Path

import config
from polar_data_loader import PolarDataLoader
from generate_visualizations import PolarResponseVisualizer

def run_pipeline(args):
    # Get measurement set configuration
    mset_name = args.measurement_set
    if mset_name not in config.MEASUREMENT_SETS:
        print(f"Error: Unknown measurement set '{mset_name}'")
        print(f"Available sets: {', '.join(config.MEASUREMENT_SETS.keys())}")
        sys.exit(1)

    mset = config.MEASUREMENT_SETS[mset_name]
    has_rear = mset["has_rear"]
    hdf5_path = config.DATA_DIR / mset["hdf5_file"]
    output_dir = mset["output_dir"]
    static_plots_dir = output_dir / "static_plots"
    interactive_plots_dir = output_dir / "interactive"

    # Check if this is a multi-source measurement set
    sources = mset.get("sources")
    data_dir = mset.get("path")
    pattern_type = mset.get("pattern_type")
    angles = mset.get("angles")

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

    # 1. Load & Process Data
    if not args.skip_loading:
        print("\n[STEP 1] Loading and Processing Data...")
        smoothing_val = 0 if args.no_smoothing else config.DEFAULT_SMOOTHING

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
                        pattern_type=src_pattern
                    )
                    src_data = loader.load_all_drivers(
                        angles=angles,
                        smoothing=smoothing_val,
                        gate_left_ms=config.GATE_LEFT_MS,
                        gate_right_ms=config.GATE_RIGHT_MS,
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
                    pattern_type=pattern_type
                )
                data = loader.load_all_drivers(
                    angles=angles,
                    smoothing=smoothing_val,
                    gate_left_ms=config.GATE_LEFT_MS,
                    gate_right_ms=config.GATE_RIGHT_MS,
                    include_rear=has_rear
                )

            # Save
            print(f"\n[STEP 2] Saving to {hdf5_path}...")
            loader.save_to_hdf5(data, str(hdf5_path),
                               gate_left_ms=config.GATE_LEFT_MS,
                               gate_right_ms=config.GATE_RIGHT_MS,
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

        viz = PolarResponseVisualizer(
            str(hdf5_path),
            static_plots_dir=static_plots_dir,
            interactive_plots_dir=interactive_plots_dir
        )
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

    args = parser.parse_args()
    run_pipeline(args)
