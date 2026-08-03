"""Runs the multi-vehicle bearing-only tracking experiment for several numbers of vehicles.

Run from the repository root, for example::

    python -m paper.experiments.filtering.run --vehicles 2 3 4 6 8 11 16 25 --episodes 200

Results are written to ``paper/data/filter-<timestamp>/`` (one ``v-<N>.npz`` per number of
vehicles and a ``config.json``) and can be evaluated with ``paper.experiments.filtering.report``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...filters import CONSTRUCTORS
from ...paths import DATA_DIR
from .scenario import ScenarioConfig
from .simulate import episode_seed, run_episode

DEFAULT_FILTERS = ["ukf_chol", "ukf_sqrtm", "ukf_d", "ukf_q", "ekf"]


def _run_task(args: tuple) -> tuple[int, int, dict]:
    config_dict, num_of_vehicles, filter_keys, base_seed, episode = args
    result = run_episode(
        ScenarioConfig.from_dict(config_dict),
        num_of_vehicles,
        filter_keys,
        episode_seed(base_seed, num_of_vehicles, episode),
    )
    return num_of_vehicles, episode, result


def run_episodes(
    config: ScenarioConfig,
    nums_of_vehicles: list[int],
    filter_keys: list[str],
    num_of_episodes: int,
    *,
    base_seed: int = 0,
    processes: int | None = None,
    progress: bool = True,
) -> dict[int, dict[str, np.ndarray]]:
    """Runs all episodes in parallel and stacks the results per number of vehicles (leading axis =
    episode, ordered by episode index)."""
    # The largest problems first to balance the load.
    tasks = [
        (config.to_dict(), num_of_vehicles, filter_keys, base_seed, episode)
        for num_of_vehicles in sorted(nums_of_vehicles, reverse=True)
        for episode in range(num_of_episodes)
    ]
    collected: dict[int, list[dict | None]] = {
        num_of_vehicles: [None] * num_of_episodes for num_of_vehicles in nums_of_vehicles
    }
    with multiprocessing.Pool(processes or os.cpu_count()) as pool:
        iterator = pool.imap_unordered(_run_task, tasks)
        if progress:
            iterator = tqdm(iterator, total=len(tasks), unit="episode")
        for num_of_vehicles, episode, result in iterator:
            collected[num_of_vehicles][episode] = result
    return {
        num_of_vehicles: {
            name: np.stack([episode[name] for episode in episodes])  # type: ignore[index]
            for name in episodes[0]  # type: ignore[union-attr]
        }
        for num_of_vehicles, episodes in collected.items()
    }


def save_run(
    directory: Path,
    config: ScenarioConfig,
    filter_keys: list[str],
    base_seed: int,
    results: dict[int, dict[str, np.ndarray]],
):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "config.json", "w") as f:
        json.dump(
            {
                "scenario": config.to_dict(),
                "filters": filter_keys,
                "seed": base_seed,
                "vehicles": sorted(results),
            },
            f,
            indent=2,
        )
    for num_of_vehicles, data in results.items():
        np.savez_compressed(directory / f"v-{num_of_vehicles}.npz", allow_pickle=True, **data)


def load_run(directory: Path) -> tuple[dict, dict[int, dict[str, np.ndarray]]]:
    """Loads the meta data (``config.json``) and the results per number of vehicles."""
    with open(directory / "config.json") as f:
        meta = json.load(f)
    results = {}
    for num_of_vehicles in meta["vehicles"]:
        with np.load(directory / f"v-{num_of_vehicles}.npz") as archive:
            results[num_of_vehicles] = {name: archive[name] for name in archive.files}
    return meta, results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--vehicles", type=int, nargs="+", default=[2, 3, 4, 6, 8, 11, 16, 25])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--filters", nargs="+", default=DEFAULT_FILTERS, choices=list(CONSTRUCTORS))
    parser.add_argument("--noise-scale", type=float, default=ScenarioConfig.noise_scale)
    parser.add_argument("--steps", type=int, default=ScenarioConfig.num_of_steps)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = ScenarioConfig(noise_scale=args.noise_scale, num_of_steps=args.steps)
    stamp = datetime.now().replace(microsecond=0).isoformat()
    directory = args.output or DATA_DIR / f"filter-{stamp}"
    results = run_episodes(
        config,
        args.vehicles,
        args.filters,
        args.episodes,
        base_seed=args.seed,
        processes=args.processes,
    )
    save_run(directory, config, args.filters, args.seed, results)
    print(f"Wrote {directory}")


if __name__ == "__main__":
    main()
