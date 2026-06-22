"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Run the configurable trail digital-twin extension pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.trail_digital_twin_pipeline import load_config, run_pipeline, write_outputs  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trail digital-twin extension fitting and reporting.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/trail_digital_twin_extensions.yaml"),
        help="YAML config file.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config paths.output_dir.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Parallel Stage 0-3 fitting workers. Defaults to execution.jobs from the YAML config.",
    )
    parser.add_argument(
        "--html", dest="write_html", action="store_true", default=None, help="Force HTML report output."
    )
    parser.add_argument("--no-html", dest="write_html", action="store_false", help="Disable HTML report output.")
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate the config and exit without running the fitting pipeline.",
    )
    parser.add_argument("--quiet", action="store_true", help="Print only final output paths.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    if args.output_dir is not None:
        config["paths"]["output_dir"] = str(args.output_dir)
    if args.jobs is not None:
        if args.jobs < 1:
            raise SystemExit("--jobs must be a positive integer.")
        config.setdefault("execution", {})["jobs"] = int(args.jobs)
    if args.write_html is not None:
        config["outputs"]["write_html"] = bool(args.write_html)
    output_dir_value = config["paths"].get("output_dir")
    if not output_dir_value:
        raise SystemExit("Refusing to write without paths.output_dir or --output-dir.")
    output_dir = Path(str(output_dir_value))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    if args.validate_config:
        if not args.quiet:
            print(f"Config OK: {config_path}")
            print(f"Output directory: {output_dir}")
        return

    if not args.quiet:
        objectives = ", ".join(config["fitting"]["enabled_objectives"])
        print(f"Running trail digital-twin pipeline with objectives: {objectives}")
        print(f"Config: {config_path}")
        print(f"Output: {output_dir}")
        print(f"Jobs: {config['execution']['jobs']}")
    result = run_pipeline(config, project_root=REPO_ROOT, config_path=config_path)
    written = write_outputs(result, output_dir)
    if args.quiet:
        for path in written.values():
            print(path)
    else:
        print("\nWritten outputs:")
        for name, path in sorted(written.items()):
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
