"""Calibrates the noise scale of the experiment.

For a range of noise scales, a few episodes are run for a small and a large number of vehicles.
The table reports the distribution of rho (relative position standard deviation / distance of the
observed object, see ``simulate.observation_rho``) and the per-filter metrics.

Selection rule (fixed before looking at the results): the largest noise scale for which *all*
filters reach at least ``--min-coverage`` position coverage with the small number of vehicles,
i.e. the hardest setting in which a UKF is still adequate in low dimensions.

Run from the repository root, for example::

    python -m paper.experiments.filtering.calibrate
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ...filters import CONSTRUCTORS
from ...paths import DATA_DIR
from .metrics import METRICS, metric_value, rho_quantiles, rho_regime_fractions
from .run import DEFAULT_FILTERS, run_episodes
from .scenario import ScenarioConfig

RHO_EDGES = (0.1, 0.15, 0.3, 0.5)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--noise-scales", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 3, 4, 6, 8]
    )
    parser.add_argument("--vehicles", type=int, nargs=2, default=[3, 11])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--filters", nargs="+", default=DEFAULT_FILTERS, choices=list(CONSTRUCTORS))
    parser.add_argument("--min-coverage", type=float, default=0.93)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base = ScenarioConfig()
    rows = []
    for noise_scale in args.noise_scales:
        config = base.replace(noise_scale=noise_scale)
        results = run_episodes(
            config, args.vehicles, args.filters, args.episodes, base_seed=args.seed, progress=False
        )
        for num_of_vehicles, data in sorted(results.items()):
            row = {
                "noise_scale": noise_scale,
                "sensor_noise_deg": float(np.rad2deg(noise_scale * base.sensor_noise_stddev)),
                "num_of_vehicles": num_of_vehicles,
                "rho_quantiles": rho_quantiles(data).tolist(),
                "rho_fractions": rho_regime_fractions(data, RHO_EDGES).tolist(),
                "metrics": {
                    key: {name: metric_value(data, key, name) for name in METRICS}
                    for key in args.filters
                },
            }
            rows.append(row)
            print_row(row, args.filters)

    small = min(args.vehicles)
    admissible = [
        row["noise_scale"]
        for row in rows
        if row["num_of_vehicles"] == small
        and all(m["position_coverage"] >= args.min_coverage for m in row["metrics"].values())
    ]
    chosen = max(admissible) if admissible else None
    print(
        f"\nLargest noise scale with all filters >= {args.min_coverage} coverage at "
        f"N={small}: {chosen}"
    )

    stamp = datetime.now().replace(microsecond=0).isoformat()
    output = args.output or DATA_DIR / f"calibration-{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "base": base.to_dict(),
                "args": {k: str(v) for k, v in vars(args).items()},
                "rho_edges": RHO_EDGES,
                "rows": rows,
                "chosen_noise_scale": chosen,
            },
            f,
            indent=2,
        )
    print(f"Wrote {output}")


def print_row(row: dict, filter_keys: list[str]):
    q = row["rho_quantiles"]
    fractions = " ".join(f"{100 * _:4.1f}" for _ in row["rho_fractions"])
    print(
        f"\ns={row['noise_scale']:<4g} sigma_theta={row['sensor_noise_deg']:5.2f}deg "
        f"N={row['num_of_vehicles']:<3d} rho q50/q90/q99={q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}  "
        f"rho% <.1|.1-.15|.15-.3|.3-.5|>.5: {fractions}"
    )
    print(
        f"  {'filter':<16} {'pos cov':>8} {'joint cov':>9} {'pos RMSE':>9} "
        f"{'diverged':>8} {'flip':>6}"
    )
    for key in filter_keys:
        m = row["metrics"][key]
        print(
            f"  {CONSTRUCTORS[key][1]:<16} {m['position_coverage']:8.3f} "
            f"{m['joint_coverage']:9.3f} "
            f"{m['position_rmse']:9.3f} {m['diverged']:8.3f} {m['flip_rate']:6.3f}"
        )


if __name__ == "__main__":
    main()
