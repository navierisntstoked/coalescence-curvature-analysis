#!/usr/bin/env python3
"""Run the curvature pipeline from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coalescence_curvature import (  # noqa: E402
    AnalysisConfig,
    analyze_frame,
    analyze_tiff,
    plot_frame_diagnostic,
    plot_time_series,
    read_tiff_frame,
    save_measurements_csv,
    summarize_measurements,
    tiff_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract neck radius and curvature from a multi-frame TIFF stack."
    )
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--fps", type=float, default=200.0)
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--midpoint-y", type=float)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--fit-half-width", type=int, default=7)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--diagnostic-frame",
        type=int,
        help="Frame to show in the fit-overlay figure; defaults to the stack midpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = tiff_metadata(args.input_tiff)
    config = AnalysisConfig(
        threshold=args.threshold,
        fps=args.fps,
        midpoint_y=args.midpoint_y,
        polynomial_degree=args.degree,
        fit_half_width_px=args.fit_half_width,
    )

    started = time.perf_counter()
    measurements = analyze_tiff(args.input_tiff, config, max_frames=args.max_frames)
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_measurements_csv(measurements, args.output_dir / "curvature_measurements.csv")
    plot_time_series(measurements, args.output_dir / "curvature_summary.png")

    diagnostic_index = args.diagnostic_frame
    if diagnostic_index is None:
        diagnostic_index = len(measurements) // 2
    diagnostic_index = min(diagnostic_index, len(measurements) - 1)
    diagnostic_frame = read_tiff_frame(args.input_tiff, diagnostic_index)
    diagnostic = analyze_frame(diagnostic_frame, diagnostic_index, config)
    plot_frame_diagnostic(
        diagnostic_frame,
        diagnostic,
        args.output_dir / "frame_diagnostic.png",
    )

    summary = {
        **metadata,
        **summarize_measurements(measurements),
        "fps": config.fps,
        "analysis_runtime_s": elapsed,
        "throughput_frames_per_s": len(measurements) / elapsed,
        "threshold": config.threshold,
        "polynomial_degree": config.polynomial_degree,
        "fit_half_width_px": config.fit_half_width_px,
    }
    with (args.output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
