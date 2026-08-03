"""Optimize rectangular Q matrices for a range of dimensions and save them as NPZ.

Example::

    python paper/scripts/optimize_q_matrices.py --min-n 2 --max-n 50 --iterations 100000 \
        --seed 12 --output paper/data/optimized_q_matrices.npz

By default, ``n x n`` and ``n x 2n`` matrices are generated for every ``n``.
Use ``--shapes``, for example ``--shapes 1x1 1x2``, to select ratios.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from compact_matrix_sqrt.algorithms import (
    ceol4_d_bound2,
    ceol4_d_bound2_second_branch_active,
    ceol4_d_bound_metrics,
    optimize_ceol4_d_bound2_python,
    random_Q,
)


OptimizationResult = tuple[
    dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray
]


class PartialOptimizationError(RuntimeError):
    """An optimization error carrying all matrices completed before it occurred."""

    def __init__(self, message: str, result: OptimizationResult) -> None:
        super().__init__(message)
        self.result = result


def optimize_range(min_n: int, max_n: int, *,
    step: int, iterations: int, seed: int, max_bound: float,
    shapes: tuple[tuple[int, int], ...] = ((1, 1)),
) -> OptimizationResult:
    """Return optimized matrices and diagnostic arrays for an inclusive range."""
    if min_n < 1:
        raise ValueError("min_n must be at least 1")
    if max_n < min_n:
        raise ValueError("max_n must be greater than or equal to min_n")
    if step < 1:
        raise ValueError("step must be at least 1")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    if not np.isfinite(max_bound) or max_bound <= 0.0:
        raise ValueError("max_bound must be finite and positive")
    if not shapes or any(rows < 1 or cols < 1 for rows, cols in shapes):
        raise ValueError("shapes must contain positive row and column multipliers")
    if any(rows > cols for rows, cols in shapes):
        raise ValueError("shapes must be landscape, i.e. rows must not exceed columns")
    if len(set(shapes)) != len(shapes):
        raise ValueError("shapes must not contain duplicates")

    ns = np.arange(min_n, max_n + 1, step, dtype=np.int64)
    rng = np.random.default_rng(seed)
    matrices: dict[str, np.ndarray] = {}
    number_of_matrices = len(ns) * len(shapes)
    matrix_shapes = np.empty((number_of_matrices, 2), dtype=np.int64)
    initial_bounds = np.empty(number_of_matrices, dtype=np.float64)
    optimized_bounds = np.empty(number_of_matrices, dtype=np.float64)
    completed_ns: list[int] = []

    index = 0

    def fail(message: str) -> None:
        partial_result: OptimizationResult = (
            matrices.copy(),
            np.asarray(sorted(set(completed_ns)), dtype=np.int64),
            matrix_shapes[:index].copy(),
            initial_bounds[:index].copy(),
            optimized_bounds[:index].copy(),
        )
        raise PartialOptimizationError(message, partial_result)

    for n_value in ns:
        n = int(n_value)
        for rows_per_n, cols_per_n in shapes:
            rows, cols = rows_per_n * n, cols_per_n * n
            matrix_key = f"Q_{rows}x{cols}"
            if matrix_key in matrices:
                raise ValueError(
                    f"Multiple n/shape combinations produce the same matrix {matrix_key}"
                )
            initial = random_Q(rng, (rows, cols))
            initial_bounds[index] = ceol4_d_bound2(initial)

            if rows == 1 or iterations == 0:
                optimized = initial
                used_iterations = 0
            else:
                optimized, history = optimize_ceol4_d_bound2_python(
                    initial,
                    iterations=iterations,
                    max_bound=max_bound,
                    return_history=True,
                )
                used_iterations = len(history)

            optimized_bound = ceol4_d_bound2(optimized)
            if rows >= 2 and not ceol4_d_bound2_second_branch_active(optimized):
                r_max, delta_min = ceol4_d_bound_metrics(optimized)
                threshold = (4.0 / 3.0) * r_max
                if delta_min > threshold + 1e-12:
                    fail(
                        f"Optimized matrix for shape ({rows}, {cols}) is on the wrong "
                        f"side of the bound2 branch condition: delta_min={delta_min:.12g} "
                        f"> 4/3*r_max={threshold:.12g}."
                    )
            print(
                f"shape=({rows}, {cols}): bound {initial_bounds[index]:.12g} -> "
                f"{optimized_bound:.12g}, iterations={used_iterations}/{iterations}"
            )
            if optimized_bound > max_bound:
                fail(
                    f"Optimized bound for shape ({rows}, {cols}) is "
                    f"{optimized_bound:.12g}, "
                    f"above the allowed maximum of {max_bound:.12g}."
                )
            matrices[matrix_key] = optimized
            matrix_shapes[index] = (rows, cols)
            optimized_bounds[index] = optimized_bound
            completed_ns.append(n)
            index += 1

    return matrices, ns, matrix_shapes, initial_bounds, optimized_bounds


def parse_shape(value: str) -> tuple[int, int]:
    """Parse a multiplier such as ``1x2`` for use by argparse."""
    try:
        rows_text, cols_text = value.lower().split("x", maxsplit=1)
        shape = (int(rows_text), int(cols_text))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("shape must have the form ROWSxCOLS") from error
    if shape[0] < 1 or shape[1] < 1:
        raise argparse.ArgumentTypeError("shape multipliers must be positive")
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, required=True, help="Smallest matrix size")
    parser.add_argument("--max-n", type=int, required=True, help="Largest matrix size (inclusive)")
    parser.add_argument("--step", type=int, default=1, help="Step between matrix sizes")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10000,
        help="Maximum optimization iterations per matrix (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(1, 1)],
        metavar="ROWSxCOLS",
        help="Aspect ratios to generate (default: 1x1)",
    )
    parser.add_argument(
        "--max-bound",
        type=float,
        default=3.5,
        help=(
            "Stop optimizing a matrix once ceol4_d_bound2 is at or below this "
            "value; abort if it is not reached (default: 3.5)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "optimized_q_matrices.npz",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Add matrices to an existing output archive instead of replacing it",
    )
    return parser.parse_args()


def _metadata_vector(
    archive: dict[str, np.ndarray], vector_key: str, scalar_key: str, count: int
) -> np.ndarray:
    """Read per-matrix metadata, including archives using the old scalar format."""
    if vector_key in archive:
        values = np.asarray(archive[vector_key])
        if values.shape == (count,):
            return values
    if scalar_key in archive:
        return np.full(count, np.asarray(archive[scalar_key]).item())
    return np.full(count, np.nan)


def main() -> None:
    args = parse_args()
    optimization_error: PartialOptimizationError | None = None
    try:
        result = optimize_range(
            args.min_n,
            args.max_n,
            step=args.step,
            iterations=args.iterations,
            seed=args.seed,
            max_bound=args.max_bound,
            shapes=tuple(args.shapes),
        )
    except PartialOptimizationError as error:
        result = error.result
        optimization_error = error

    matrices, ns, matrix_shapes, initial_bounds, optimized_bounds = result
    matrix_iterations = np.full(len(matrices), args.iterations, dtype=np.int64)
    matrix_seeds = np.full(len(matrices), args.seed, dtype=np.int64)
    matrix_max_bounds = np.full(len(matrices), args.max_bound, dtype=np.float64)

    if args.add and args.output.exists():
        with np.load(args.output) as loaded:
            existing = {key: np.array(loaded[key]) for key in loaded.files}
        existing_matrices = {
            key: value for key, value in existing.items() if key.startswith("Q_")
        }
        duplicates = sorted(existing_matrices.keys() & matrices.keys())
        if duplicates:
            raise RuntimeError(
                "Refusing to overwrite existing matrices in --add mode: "
                + ", ".join(duplicates)
            )

        existing_count = len(existing_matrices)
        existing_shapes = existing.get(
            "matrix_shapes",
            np.asarray([matrix.shape for matrix in existing_matrices.values()], dtype=np.int64),
        )
        matrix_shapes = np.concatenate([existing_shapes, matrix_shapes])
        initial_bounds = np.concatenate(
            [
                _metadata_vector(existing, "initial_bounds", "initial_bound", existing_count),
                initial_bounds,
            ]
        )
        optimized_bounds = np.concatenate(
            [
                _metadata_vector(existing, "optimized_bounds", "optimized_bound", existing_count),
                optimized_bounds,
            ]
        )
        matrix_iterations = np.concatenate(
            [
                _metadata_vector(existing, "matrix_iterations", "iterations", existing_count),
                matrix_iterations,
            ]
        ).astype(np.int64)
        matrix_seeds = np.concatenate(
            [_metadata_vector(existing, "matrix_seeds", "seed", existing_count), matrix_seeds]
        ).astype(np.int64)
        matrix_max_bounds = np.concatenate(
            [
                _metadata_vector(existing, "matrix_max_bounds", "max_bound", existing_count),
                matrix_max_bounds,
            ]
        )
        ns = np.unique(np.concatenate([np.asarray(existing.get("ns", [])), ns])).astype(np.int64)
        matrices = existing_matrices | matrices

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        ns=ns,
        matrix_shapes=matrix_shapes,
        initial_bounds=initial_bounds,
        optimized_bounds=optimized_bounds,
        matrix_iterations=matrix_iterations,
        matrix_seeds=matrix_seeds,
        matrix_max_bounds=matrix_max_bounds,
        **matrices,
    )
    print(f"Wrote {len(matrices)} optimized matrices to {args.output}")
    if optimization_error is not None:
        raise optimization_error


if __name__ == "__main__":
    main()
