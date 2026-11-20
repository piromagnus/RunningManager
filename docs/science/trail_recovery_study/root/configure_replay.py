"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Point PR #5 at the portable PR #4 data, and the results at a chosen folder."""

from argparse import ArgumentParser
from pathlib import Path


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("repo", type=Path)
    p.add_argument("portable", type=Path)
    p.add_argument("config_output", type=Path)
    p.add_argument("results_dir", type=Path)
    args = p.parse_args()
    source = args.repo / "configs/trail_digital_twin_claim_validation.yaml"
    config = source.read_text()
    for path in ("data/timeseries", "data/metrics_ts", "data/raw/strava"):
        assert (args.portable / path).is_dir()
        config = config.replace(f": {path}\n", f': "{(args.portable / path).resolve()}"\n')
    config = config.replace(
        ": data/exp_perf_predictions/trail_digital_twin_claim_validation\n",
        f': "{args.results_dir.resolve()}"\n',
    )
    args.config_output.write_text(config)


if __name__ == "__main__":
    main()
