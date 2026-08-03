"""Animates one episode of the multi-vehicle bearing-only tracking experiment.

The episode is re-simulated deterministically from its seed, so the animation shows exactly the
episode of a run (``--run-dir``) including the sigma points, which are not stored in the run.

One panel per filter shows the true vehicles (black, with trails), the landmarks, the bearing
measurements (dashed lines from the observer along the *measured* bearing, with the length of the
true distance), the estimated positions with their 95% ellipses (red when the true position lies
outside) and optionally the sigma points. Observations for which the filter's sigma points
"flipped" (a predicted bearing more than 90 degrees from the predicted mean) are drawn solid red.

Frames are 1280x720 px like ``paper/visualization/cel4o-dq-video.mp4``; the output is a GIF (or an
mp4 if ffmpeg is installed).

Run from the repository root, for example::

    python -m paper.experiments.filtering.animate --run-dir paper/data/filter-<timestamp> \\
        --vehicles 11 --rank
    python -m paper.experiments.filtering.animate --run-dir paper/data/filter-<timestamp> \\
        --vehicles 11 --episode 3 --sigma-points
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.patheffects import withStroke
from scipy.stats import chi2

from ... import plot_style as ps
from ...filters import CONSTRUCTORS
from ...paths import DATA_DIR
from .metrics import COVERAGE_PROBABILITY, FLIP_THRESHOLD, position_coverage
from .report import SHORT_NAMES
from .scenario import ScenarioConfig
from .simulate import episode_seed, run_episode

TRUE_COLOR = "#0b0b0b"
MUTED_COLOR = "#8a8984"
MISS_COLOR = "#e34948"
TRAIL_LENGTH = 30  # [steps]
FRAME_DPI = 100
FRAME_SIZE = (12.8, 7.2)  # [in] at FRAME_DPI -> 1280x720 px, like cel4o-dq-video.mp4
TOP_MARGIN = 0.85  # [in] suptitle and two-line panel titles
BOTTOM_MARGIN = 0.45  # [in] legend
SIGMA_POINT_SIZE = 3.5  # [pt]


def _darker(color: str, factor: float = 0.6) -> tuple[float, float, float]:
    r, g, b = to_rgb(color)
    return (factor * r, factor * g, factor * b)


def rank_episodes(run_dir: Path, num_of_vehicles: int, filter_keys: list[str], count: int = 10):
    """Prints the episodes with the largest coverage gap between the second and first filter."""
    with np.load(run_dir / f"v-{num_of_vehicles}.npz") as archive:
        data = {name: archive[name] for name in archive.files}
    coverage = {key: position_coverage(data, key) for key in filter_keys}
    gap = coverage[filter_keys[1]] - coverage[filter_keys[0]]
    print(f"episodes by position coverage gap {filter_keys[1]} - {filter_keys[0]}:")
    for episode in np.argsort(-gap)[:count]:
        values = "  ".join(f"{key}={coverage[key][episode]:.3f}" for key in filter_keys)
        print(f"  episode {episode:4d}: gap={gap[episode]:+.3f}  {values}")


class EpisodeAnimation:
    def __init__(
        self,
        config: ScenarioConfig,
        num_of_vehicles: int,
        filter_keys: list[str],
        episode: dict[str, np.ndarray],
        *,
        show_sigma_points: bool,
    ):
        self._config = config
        self._num_of_vehicles = num_of_vehicles
        self._filter_keys = filter_keys
        self._episode = episode
        self._show_sigma_points = show_sigma_points
        self._landmarks = config.landmarks()
        self._true_positions = np.reshape(
            episode["true"], (config.num_of_steps, num_of_vehicles, 4)
        )[..., :2]
        self._ellipse_scale = np.sqrt(chi2.ppf(COVERAGE_PROBABILITY, df=2))
        self._sensor_noise_stddev = config.noise_scale * config.sensor_noise_stddev

        points = np.concatenate([np.reshape(self._true_positions, (-1, 2)), self._landmarks])
        low, high = np.min(points, axis=0) - 2, np.max(points, axis=0) + 2

        self.figure, axes = plt.subplots(1, len(filter_keys), figsize=FRAME_SIZE, squeeze=False)
        self._axes = axes[0]
        for axis in self._axes:
            axis.set_xlim(low[0], high[0])
            axis.set_ylim(low[1], high[1])
            axis.set_aspect("equal")
        self.figure.subplots_adjust(
            left=0.02,
            right=0.98,
            bottom=BOTTOM_MARGIN / FRAME_SIZE[1],
            top=1 - TOP_MARGIN / FRAME_SIZE[1],
            wspace=0.05,
        )
        self.figure.legend(
            handles=self._legend_handles(),
            loc="lower center",
            ncol=4 if show_sigma_points else 3,
            frameon=False,
            fontsize=9,
        )

    def _legend_handles(self) -> list:
        handles = [
            Line2D([], [], linestyle="none", marker="o", color=TRUE_COLOR, label="true vehicle"),
            Line2D([], [], linestyle="none", marker="^", color=MUTED_COLOR, label="landmark"),
            Line2D([], [], color=MUTED_COLOR, linestyle="--", label="bearing measurement"),
            Line2D([], [], color=MISS_COLOR, linewidth=1.6, label="sigma points flipped (>90°)"),
            Line2D(
                [],
                [],
                color=TRUE_COLOR,
                linestyle="none",
                marker="x",
                markeredgewidth=1.6,
                label="estimate, 95% ellipse",
            ),
            Line2D(
                [], [], color=MISS_COLOR, linestyle=(0, (3, 1.5)), label="truth outside ellipse"
            ),
        ]
        if self._show_sigma_points:
            handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=SIGMA_POINT_SIZE,
                    color=MUTED_COLOR,
                    markeredgecolor=_darker(MUTED_COLOR),
                    markeredgewidth=0.5,
                    label="sigma points",
                )
            )
        return handles

    def draw(self, step: int):
        for axis, key in zip(self._axes, self._filter_keys, strict=True):
            self._draw_panel(axis, key, step)
        self.figure.suptitle(
            f"t = {(step + 1) * self._config.dt:5.1f} s   N = {self._num_of_vehicles} vehicles "
            f"(n = {4 * self._num_of_vehicles})   "
            f"bearing noise {np.rad2deg(self._sensor_noise_stddev):.1f}°",
            fontsize=12,
        )

    def _draw_panel(self, axis, key: str, step: int):
        xlim, ylim = axis.get_xlim(), axis.get_ylim()
        axis.cla()
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xticks([])
        axis.set_yticks([])
        color = ps.IDENTITIES[ps.FILTER_IDENTITY[key]].color
        episode = self._episode
        truth = self._true_positions[step]

        axis.plot(
            *self._landmarks.T, linestyle="none", marker="^", markersize=10, color=MUTED_COLOR
        )

        # Measurements: from the observer along the measured bearing with the true distance.
        residual = episode[f"{key}/max_sigma_residual"][step]
        for vehicle, target in enumerate(episode["observation"][step]):
            target_position = (
                truth[target]
                if target < self._num_of_vehicles
                else self._landmarks[target - self._num_of_vehicles]
            )
            distance = np.linalg.norm(target_position - truth[vehicle])
            bearing = episode["measurement"][step, vehicle]
            end = truth[vehicle] + distance * np.array([np.cos(bearing), np.sin(bearing)])
            flipped = residual[vehicle] > FLIP_THRESHOLD
            axis.plot(
                [truth[vehicle, 0], end[0]],
                [truth[vehicle, 1], end[1]],
                linestyle="-" if flipped else "--",
                linewidth=1.6 if flipped else 0.8,
                color=MISS_COLOR if flipped else MUTED_COLOR,
                zorder=1,
            )

        if self._show_sigma_points and f"{key}/sigma_points" in episode:
            points = np.reshape(
                episode[f"{key}/sigma_points"][step], (-1, self._num_of_vehicles, 4)
            )
            axis.plot(
                points[..., 0].ravel(),
                points[..., 1].ravel(),
                linestyle="none",
                marker="o",
                markersize=SIGMA_POINT_SIZE,
                color=color,
                markeredgecolor=_darker(color),
                markeredgewidth=0.5,
                zorder=2,
            )

        # Truth with trails.
        first = max(0, step - TRAIL_LENGTH)
        for vehicle in range(self._num_of_vehicles):
            axis.plot(
                *self._true_positions[first : step + 1, vehicle].T,
                color=TRUE_COLOR,
                alpha=0.25,
                linewidth=1,
            )
        axis.plot(*truth.T, linestyle="none", marker="o", markersize=5, color=TRUE_COLOR, zorder=4)

        # Estimates with 95% ellipses.
        diverged_at = int(episode[f"{key}/diverged_at"])
        position_nees = episode[f"{key}/position_nees"]
        if step < diverged_at:
            mean = np.reshape(episode[f"{key}/mean"][step], (self._num_of_vehicles, 4))[:, :2]
            covariances = episode[f"{key}/position_covariance"][step]
            for vehicle in range(self._num_of_vehicles):
                eigenvalues, eigenvectors = np.linalg.eigh(covariances[vehicle])
                covered = position_nees[step, vehicle] <= self._ellipse_scale**2
                axis.add_patch(
                    Ellipse(
                        mean[vehicle],
                        width=2 * self._ellipse_scale * np.sqrt(max(eigenvalues[1], 0)),
                        height=2 * self._ellipse_scale * np.sqrt(max(eigenvalues[0], 0)),
                        angle=np.rad2deg(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])),
                        fill=False,
                        linewidth=1.5 if covered else 2.0,
                        linestyle="-" if covered else (0, (3, 1.5)),
                        edgecolor=color if covered else MISS_COLOR,
                        zorder=3,
                    )
                )
            # Black with a white halo, so that it stays visible on top of the sigma points.
            axis.plot(
                *mean.T,
                linestyle="none",
                marker="x",
                markersize=7,
                markeredgewidth=1.6,
                color=TRUE_COLOR,
                path_effects=[withStroke(linewidth=3, foreground="white")],
                zorder=6,
            )

        with np.errstate(invalid="ignore"):
            covered_so_far = position_nees[: step + 1] <= self._ellipse_scale**2
        status = "  DIVERGED" if step >= diverged_at else ""
        axis.set_title(
            f"{CONSTRUCTORS[key][1]}{status}\n"
            f"position coverage so far {100 * np.mean(covered_so_far):.1f}%   "
            f"now {int(np.sum(~covered_so_far[-1]))} of {self._num_of_vehicles} outside",
            fontsize=11,
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, help="Run to take the scenario and seed from")
    parser.add_argument("--noise-scale", type=float, help="Overrides the noise scale of the run")
    parser.add_argument("--seed", type=int, help="Overrides the seed of the run")
    parser.add_argument("--vehicles", type=int, default=11)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--filters", nargs="+", default=["ukf_chol", "ukf_d"], choices=list(CONSTRUCTORS)
    )
    parser.add_argument("--sigma-points", action="store_true")
    parser.add_argument("--rank", action="store_true", help="Only print episodes by coverage gap")
    parser.add_argument("--snapshot", type=int, nargs="*", help="Only save PNGs of these steps")
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="10 = real time (one frame per step); a 30 fps video shows each frame 3 times",
    )
    parser.add_argument("--output", type=Path, help="Defaults to an mp4 (with ffmpeg) or gif")
    args = parser.parse_args()

    meta = {}
    if args.run_dir is not None:
        with open(args.run_dir / "config.json") as f:
            meta = json.load(f)
    if args.rank:
        rank_episodes(args.run_dir, args.vehicles, args.filters)
        return
    config = ScenarioConfig.from_dict(meta["scenario"]) if meta else ScenarioConfig()
    if args.noise_scale is not None:
        config = config.replace(noise_scale=args.noise_scale)
    seed = args.seed if args.seed is not None else meta.get("seed", 0)

    episode = run_episode(
        config,
        args.vehicles,
        args.filters,
        episode_seed(seed, args.vehicles, args.episode),
        record_sigma_points=args.sigma_points,
    )
    animation = EpisodeAnimation(
        config, args.vehicles, args.filters, episode, show_sigma_points=args.sigma_points
    )
    directory = args.run_dir or DATA_DIR
    stem = f"animation-v{args.vehicles}-e{args.episode}-" + "-".join(
        SHORT_NAMES.get(_, _) for _ in args.filters
    )

    if args.snapshot is not None:
        for step in args.snapshot:
            animation.draw(step)
            path = directory / f"{stem}-step{step}.png"
            animation.figure.savefig(path, dpi=FRAME_DPI)
            print(f"Wrote {path}")
        return

    has_ffmpeg = shutil.which("ffmpeg") is not None
    output = args.output or directory / f"{stem}.{'mp4' if has_ffmpeg else 'gif'}"
    writer = (
        FFMpegWriter(fps=args.fps, bitrate=4000)
        if output.suffix == ".mp4"
        else PillowWriter(fps=args.fps)
    )
    FuncAnimation(animation.figure, animation.draw, frames=config.num_of_steps).save(
        output, writer=writer, dpi=FRAME_DPI
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
