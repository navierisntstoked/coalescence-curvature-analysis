#!/usr/bin/env python3
"""Benchmark repeated per-point fitting against one fit per boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coalescence_curvature import (  # noqa: E402
    AnalysisConfig,
    extract_boundaries,
    iter_tiff_frames,
    select_local_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_tiff", type=Path)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "benchmark.json")
    return parser.parse_args()


def _prepare_windows(path: Path, config: AnalysisConfig, frames: int):
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for frame in iter_tiff_frames(path, max_frames=frames):
        boundaries = extract_boundaries(frame, config.threshold, config.midpoint_y)
        for x, y, kind in (
            (boundaries.top_x, boundaries.top_y, "maximum"),
            (boundaries.bottom_x, boundaries.bottom_y, "minimum"),
        ):
            local_x, local_y, center_x = select_local_window(
                x,
                y,
                kind,
                config.fit_half_width_px,
                config.min_fit_points,
            )
            windows.append((local_x - center_x, local_y))
    return windows


def _repeated_per_point_fit(windows, degree: int) -> int:
    fit_calls = 0
    for x, y in windows:
        for _ in range(len(x)):
            np.polyfit(x, y, degree)
            fit_calls += 1
    return fit_calls


def _single_pass_fit(windows, degree: int) -> int:
    for x, y in windows:
        np.polyfit(x, y, degree)
    return len(windows)


def _best_time(function, repeats: int) -> tuple[float, int]:
    timings = []
    calls = 0
    for _ in range(repeats):
        started = time.perf_counter()
        calls = function()
        timings.append(time.perf_counter() - started)
    return min(timings), calls


def main() -> None:
    args = parse_args()
    config = AnalysisConfig()
    windows = _prepare_windows(args.input_tiff, config, args.frames)

    repeated_s, repeated_calls = _best_time(
        lambda: _repeated_per_point_fit(windows, config.polynomial_degree), args.repeats
    )
    single_pass_s, single_pass_calls = _best_time(
        lambda: _single_pass_fit(windows, config.polynomial_degree), args.repeats
    )

    result = {
        "frames_benchmarked": args.frames,
        "boundaries_benchmarked": len(windows),
        "repeated_fit_calls": repeated_calls,
        "single_pass_fit_calls": single_pass_calls,
        "fit_call_reduction_x": repeated_calls / single_pass_calls,
        "repeated_fit_runtime_s": repeated_s,
        "single_pass_runtime_s": single_pass_s,
        "fit_stage_speedup_x": repeated_s / single_pass_s,
        "note": "Benchmark isolates the local polynomial-fitting stage.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
