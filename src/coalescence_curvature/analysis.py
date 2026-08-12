"""Curvature extraction for bright interfaces in multi-frame TIFF stacks.

The pipeline is intentionally small and transparent:

1. Stream one TIFF frame at a time.
2. Threshold bright interface pixels.
3. Collapse line thickness to one y-value per x-coordinate.
4. Fit local polynomials around the upper and lower neck extrema.
5. Evaluate curvature from the analytical polynomial derivatives.

The public sample uses synthetic data, but the functions are configurable for
experimental stacks with the same bright-interface/dark-background geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
from typing import Iterator, Literal, Sequence

import numpy as np
from PIL import Image, ImageSequence


ExtremumKind = Literal["maximum", "minimum"]


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration shared by frame and TIFF-stack analysis."""

    threshold: int = 96
    fps: float = 200.0
    midpoint_y: float | None = None
    polynomial_degree: int = 4
    fit_half_width_px: int = 7
    min_fit_points: int = 9

    def validate(self) -> None:
        if not 0 <= self.threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.polynomial_degree < 2:
            raise ValueError("polynomial_degree must be at least 2")
        if self.fit_half_width_px < 2:
            raise ValueError("fit_half_width_px must be at least 2 pixels")
        if self.min_fit_points < self.polynomial_degree + 1:
            raise ValueError(
                "min_fit_points must exceed the number of polynomial coefficients"
            )


@dataclass(frozen=True)
class BoundaryPoints:
    """One-pixel-thick upper and lower interfaces, indexed by x-coordinate."""

    top_x: np.ndarray
    top_y: np.ndarray
    bottom_x: np.ndarray
    bottom_y: np.ndarray


@dataclass(frozen=True)
class LocalFit:
    """Local polynomial fit and curvature estimate at one neck boundary."""

    x_neck: float
    y_neck: float
    curvature_px_inv: float
    rmse_px: float
    observed_x: np.ndarray
    observed_y: np.ndarray
    fitted_x: np.ndarray
    fitted_y: np.ndarray


@dataclass(frozen=True)
class FrameMeasurement:
    """Serializable measurements for one frame."""

    frame_index: int
    time_s: float
    neck_radius_px: float
    top_curvature_px_inv: float
    bottom_curvature_px_inv: float
    mean_curvature_px_inv: float
    top_neck_x_px: float
    top_neck_y_px: float
    bottom_neck_x_px: float
    bottom_neck_y_px: float
    top_fit_rmse_px: float
    bottom_fit_rmse_px: float

    def to_record(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FrameDiagnostic:
    """Full frame result used for plots and debugging."""

    measurement: FrameMeasurement
    boundaries: BoundaryPoints
    top_fit: LocalFit
    bottom_fit: LocalFit


def tiff_metadata(path: str | Path) -> dict[str, int | str]:
    """Return basic stack metadata without loading the full TIFF into memory."""

    path = Path(path)
    with Image.open(path) as image:
        return {
            "frames": int(getattr(image, "n_frames", 1)),
            "width_px": int(image.width),
            "height_px": int(image.height),
            "mode": image.mode,
        }


def iter_tiff_frames(
    path: str | Path, max_frames: int | None = None
) -> Iterator[np.ndarray]:
    """Yield grayscale frames while keeping only one frame in memory."""

    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive when provided")

    with Image.open(path) as image:
        for frame_index, frame in enumerate(ImageSequence.Iterator(image)):
            if max_frames is not None and frame_index >= max_frames:
                break
            yield np.asarray(frame.convert("L"), dtype=np.float64)


def read_tiff_frame(path: str | Path, frame_index: int) -> np.ndarray:
    """Read one grayscale frame by index."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    with Image.open(path) as image:
        n_frames = int(getattr(image, "n_frames", 1))
        if frame_index >= n_frames:
            raise IndexError(f"frame_index {frame_index} exceeds stack length {n_frames}")
        image.seek(frame_index)
        return np.asarray(image.convert("L"), dtype=np.float64)


def _collapse_line_thickness(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average all detected y-values that share an x-coordinate."""

    unique_x, inverse = np.unique(xs.astype(np.int64), return_inverse=True)
    counts = np.bincount(inverse)
    mean_y = np.bincount(inverse, weights=ys.astype(float)) / counts
    return unique_x.astype(float), mean_y.astype(float)


def extract_boundaries(
    frame: np.ndarray,
    threshold: int = 96,
    midpoint_y: float | None = None,
) -> BoundaryPoints:
    """Extract upper and lower interfaces from one grayscale or RGB frame."""

    array = np.asarray(frame)
    if array.ndim == 3:
        if array.shape[-1] < 3:
            raise ValueError("RGB input must have at least three channels")
        array = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    elif array.ndim != 2:
        raise ValueError("frame must be a 2D grayscale or 3D RGB array")

    midpoint = float(array.shape[0] / 2 if midpoint_y is None else midpoint_y)
    ys, xs = np.nonzero(array >= threshold)
    if xs.size == 0:
        raise ValueError("no interface pixels detected; lower the threshold")

    top_mask = ys < midpoint
    bottom_mask = ys > midpoint
    if top_mask.sum() < 5 or bottom_mask.sum() < 5:
        raise ValueError("both interfaces need at least five detected pixels")

    top_x, top_y = _collapse_line_thickness(xs[top_mask], ys[top_mask])
    bottom_x, bottom_y = _collapse_line_thickness(xs[bottom_mask], ys[bottom_mask])
    return BoundaryPoints(top_x, top_y, bottom_x, bottom_y)


def select_local_window(
    x: np.ndarray,
    y: np.ndarray,
    kind: ExtremumKind,
    half_width_px: int,
    min_points: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select points around the discrete boundary extremum."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < min_points:
        raise ValueError("boundary arrays must have equal length and enough points")

    index = int(np.argmax(y) if kind == "maximum" else np.argmin(y))
    center_x = float(x[index])
    mask = np.abs(x - center_x) <= half_width_px

    if int(mask.sum()) < min_points:
        nearest = np.argsort(np.abs(x - center_x))[:min_points]
        nearest.sort()
        return x[nearest], y[nearest], center_x

    return x[mask], y[mask], center_x


def fit_local_boundary(
    x: np.ndarray,
    y: np.ndarray,
    kind: ExtremumKind,
    degree: int = 4,
    half_width_px: int = 7,
    min_points: int = 9,
) -> LocalFit:
    """Fit a local polynomial and evaluate curvature at its neck extremum."""

    local_x, local_y, center_x = select_local_window(
        x, y, kind, half_width_px, min_points
    )
    local_u = local_x - center_x
    coefficients = np.polyfit(local_u, local_y, degree)
    first_derivative = np.polyder(coefficients, 1)
    second_derivative = np.polyder(coefficients, 2)

    roots = np.roots(first_derivative)
    real_roots = np.array(
        [root.real for root in roots if abs(root.imag) < 1e-8], dtype=float
    )
    within_window = real_roots[
        (real_roots >= local_u.min()) & (real_roots <= local_u.max())
    ]

    desired_sign = -1 if kind == "maximum" else 1
    matching_roots = np.array(
        [
            root
            for root in within_window
            if desired_sign * np.polyval(second_derivative, root) > 0
        ],
        dtype=float,
    )
    candidates = matching_roots if matching_roots.size else within_window
    extremum_u = float(candidates[np.argmin(np.abs(candidates))]) if candidates.size else 0.0

    y_neck = float(np.polyval(coefficients, extremum_u))
    slope = float(np.polyval(first_derivative, extremum_u))
    second = float(np.polyval(second_derivative, extremum_u))
    curvature = abs(second) / (1.0 + slope**2) ** 1.5

    predicted_y = np.polyval(coefficients, local_u)
    rmse = float(np.sqrt(np.mean((predicted_y - local_y) ** 2)))
    fitted_u = np.linspace(local_u.min(), local_u.max(), 200)

    return LocalFit(
        x_neck=center_x + extremum_u,
        y_neck=y_neck,
        curvature_px_inv=float(curvature),
        rmse_px=rmse,
        observed_x=local_x,
        observed_y=local_y,
        fitted_x=center_x + fitted_u,
        fitted_y=np.polyval(coefficients, fitted_u),
    )


def analyze_frame(
    frame: np.ndarray,
    frame_index: int,
    config: AnalysisConfig,
) -> FrameDiagnostic:
    """Extract bridge radius and upper/lower curvature from one frame."""

    config.validate()
    boundaries = extract_boundaries(frame, config.threshold, config.midpoint_y)
    top_fit = fit_local_boundary(
        boundaries.top_x,
        boundaries.top_y,
        kind="maximum",
        degree=config.polynomial_degree,
        half_width_px=config.fit_half_width_px,
        min_points=config.min_fit_points,
    )
    bottom_fit = fit_local_boundary(
        boundaries.bottom_x,
        boundaries.bottom_y,
        kind="minimum",
        degree=config.polynomial_degree,
        half_width_px=config.fit_half_width_px,
        min_points=config.min_fit_points,
    )

    neck_radius = max(0.0, (bottom_fit.y_neck - top_fit.y_neck) / 2.0)
    mean_curvature = (top_fit.curvature_px_inv + bottom_fit.curvature_px_inv) / 2.0
    measurement = FrameMeasurement(
        frame_index=frame_index,
        time_s=frame_index / config.fps,
        neck_radius_px=float(neck_radius),
        top_curvature_px_inv=top_fit.curvature_px_inv,
        bottom_curvature_px_inv=bottom_fit.curvature_px_inv,
        mean_curvature_px_inv=float(mean_curvature),
        top_neck_x_px=top_fit.x_neck,
        top_neck_y_px=top_fit.y_neck,
        bottom_neck_x_px=bottom_fit.x_neck,
        bottom_neck_y_px=bottom_fit.y_neck,
        top_fit_rmse_px=top_fit.rmse_px,
        bottom_fit_rmse_px=bottom_fit.rmse_px,
    )
    return FrameDiagnostic(measurement, boundaries, top_fit, bottom_fit)


def analyze_tiff(
    path: str | Path,
    config: AnalysisConfig,
    max_frames: int | None = None,
) -> list[FrameMeasurement]:
    """Analyze a TIFF stack sequentially without loading it all into memory."""

    config.validate()
    measurements: list[FrameMeasurement] = []
    for frame_index, frame in enumerate(iter_tiff_frames(path, max_frames=max_frames)):
        measurements.append(analyze_frame(frame, frame_index, config).measurement)
    if not measurements:
        raise ValueError("the TIFF stack did not contain any frames")
    return measurements


def save_measurements_csv(
    measurements: Sequence[FrameMeasurement], path: str | Path
) -> None:
    """Write frame measurements to a CSV file."""

    if not measurements:
        raise ValueError("measurements cannot be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [measurement.to_record() for measurement in measurements]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def summarize_measurements(
    measurements: Sequence[FrameMeasurement],
) -> dict[str, float | int]:
    """Return concise portfolio-friendly analysis metrics."""

    if not measurements:
        raise ValueError("measurements cannot be empty")
    neck = np.array([m.neck_radius_px for m in measurements], dtype=float)
    curvature = np.array([m.mean_curvature_px_inv for m in measurements], dtype=float)
    rmse = np.array(
        [(m.top_fit_rmse_px + m.bottom_fit_rmse_px) / 2 for m in measurements],
        dtype=float,
    )
    return {
        "frames_analyzed": len(measurements),
        "initial_neck_radius_px": float(neck[0]),
        "final_neck_radius_px": float(neck[-1]),
        "minimum_mean_curvature_px_inv": float(np.nanmin(curvature)),
        "maximum_mean_curvature_px_inv": float(np.nanmax(curvature)),
        "median_fit_rmse_px": float(np.nanmedian(rmse)),
    }


def plot_time_series(
    measurements: Sequence[FrameMeasurement], path: str | Path | None = None
):
    """Plot bridge growth and curvature evolution."""

    import matplotlib.pyplot as plt

    time = np.array([m.time_s for m in measurements])
    neck = np.array([m.neck_radius_px for m in measurements])
    curvature = np.array([m.mean_curvature_px_inv for m in measurements])

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    axes[0].plot(time, neck, color="#D1495B", linewidth=2)
    axes[0].set(xlabel="Time (s)", ylabel="Neck radius (px)")

    axes[1].plot(time, curvature, color="#2C7DA0", linewidth=2)
    axes[1].set(xlabel="Time (s)", ylabel="Mean curvature (px$^{-1}$)")

    positive = (neck > 0) & (curvature > 0)
    axes[2].loglog(neck[positive], curvature[positive], color="#6A4C93", linewidth=2)
    axes[2].set(xlabel="Neck radius (px)", ylabel="Mean curvature (px$^{-1}$)")

    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure


def plot_frame_diagnostic(
    frame: np.ndarray,
    diagnostic: FrameDiagnostic,
    path: str | Path | None = None,
):
    """Overlay extracted boundaries, local fits, and neck endpoints."""

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.2, 8.0))
    axis.imshow(frame, cmap="gray", origin="upper")
    boundaries = diagnostic.boundaries
    axis.scatter(boundaries.top_x, boundaries.top_y, s=2, color="#4CC9F0", label="Top boundary")
    axis.scatter(
        boundaries.bottom_x,
        boundaries.bottom_y,
        s=2,
        color="#F72585",
        label="Bottom boundary",
    )
    axis.plot(
        diagnostic.top_fit.fitted_x,
        diagnostic.top_fit.fitted_y,
        color="#FFD166",
        linewidth=2,
    )
    axis.plot(
        diagnostic.bottom_fit.fitted_x,
        diagnostic.bottom_fit.fitted_y,
        color="#FFD166",
        linewidth=2,
        label="Local polynomial fit",
    )
    axis.plot(
        [diagnostic.top_fit.x_neck, diagnostic.bottom_fit.x_neck],
        [diagnostic.top_fit.y_neck, diagnostic.bottom_fit.y_neck],
        color="white",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="Measured neck",
    )
    axis.set(
        title=f"Frame {diagnostic.measurement.frame_index}",
        xlabel="x (px)",
        ylabel="y (px)",
    )
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200, bbox_inches="tight")
    return figure
