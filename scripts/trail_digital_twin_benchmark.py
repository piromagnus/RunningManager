"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Run a benchmark sweep over trail digital-twin extension configs.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from services.trail_digital_twin_benchmark import (  # noqa: E402
    BenchmarkRunResult,
    benchmark_plan_frame,
    combine_benchmark_tables,
    execute_benchmark_run,
    expand_benchmark_runs,
    load_benchmark_config,
    read_benchmark_output_tables,
    write_benchmark_html,
    write_benchmark_outputs,
)
from services.trail_digital_twin_pipeline import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run trail digital-twin benchmark sweeps.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/trail_digital_twin_extensions.yaml"),
        help="Base YAML pipeline config.",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=Path("configs/trail_digital_twin_benchmark.yaml"),
        help="Benchmark sweep YAML config.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override benchmark execution.output_dir.")
    parser.add_argument("--max-runs", type=int, default=None, help="Run only the first N expanded experiments.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Number of benchmark runs to execute in parallel. Defaults to execution.jobs or 1.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the expanded plan without fitting models.")
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Rebuild benchmark HTML from existing aggregate CSVs without fitting models.",
    )
    parser.add_argument("--validate-config", action="store_true", help="Validate configs and print run count.")
    parser.add_argument("--fail-fast", action="store_true", default=None, help="Stop at the first failed run.")
    parser.add_argument("--html", dest="write_html", action="store_true", default=None, help="Write benchmark HTML.")
    parser.add_argument("--no-html", dest="write_html", action="store_false", help="Skip benchmark HTML.")
    parser.add_argument("--quiet", action="store_true", help="Print only final output paths.")
    return parser


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _benchmark_output_dir(benchmark_config: dict[str, object], cli_output_dir: Path | None) -> Path:
    if cli_output_dir is not None:
        return _repo_path(cli_output_dir)
    execution = benchmark_config.get("execution", {})
    output_dir_value = execution.get("output_dir") if isinstance(execution, dict) else None
    if not output_dir_value:
        raise SystemExit("Refusing to write without benchmark execution.output_dir or --output-dir.")
    return _repo_path(Path(str(output_dir_value)))


def _empty_benchmark_tables(plan: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return combine_benchmark_tables(
        per_run_tables=[],
        manifest=pd.DataFrame(columns=["runId", "experimentGroup", "experimentName", "status"]),
        plan=plan,
    )


def _benchmark_jobs(benchmark_config: dict[str, object], jobs_override: int | None) -> int:
    execution = benchmark_config["execution"]
    jobs_value = jobs_override if jobs_override is not None else execution.get("jobs", 1)
    try:
        jobs = int(jobs_value)
    except (TypeError, ValueError) as exc:
        raise SystemExit("Benchmark jobs must be a positive integer.") from exc
    if jobs < 1:
        raise SystemExit("Benchmark jobs must be a positive integer.")
    execution["jobs"] = jobs
    return jobs


def _failed_worker_result(run_id: str, group: str, name: str, output_dir: Path, error: str) -> BenchmarkRunResult:
    return BenchmarkRunResult(
        run_id=run_id,
        tables={},
        manifest_row={
            "runId": run_id,
            "experimentGroup": group,
            "experimentName": name,
            "status": "failed",
            "elapsedSec": 0.0,
            "runOutputDir": str(output_dir),
            "htmlReport": "",
            "error": error,
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    base_config_path = _repo_path(args.base_config)
    benchmark_config_path = _repo_path(args.benchmark_config)
    base_config = load_config(base_config_path)
    benchmark_config = load_benchmark_config(benchmark_config_path)
    if args.fail_fast is not None:
        benchmark_config["execution"]["fail_fast"] = bool(args.fail_fast)
    if args.write_html is not None:
        benchmark_config["execution"]["write_html"] = bool(args.write_html)
    jobs = _benchmark_jobs(benchmark_config, args.jobs)

    output_dir = _benchmark_output_dir(benchmark_config, args.output_dir)
    runs = expand_benchmark_runs(base_config, benchmark_config, output_dir, max_runs=args.max_runs)
    plan = benchmark_plan_frame(runs)
    metadata = {
        "baseConfigPath": str(base_config_path),
        "benchmarkConfigPath": str(benchmark_config_path),
        "outputDir": str(output_dir),
        "plannedRunCount": len(runs),
        "maxRuns": args.max_runs,
        "dryRun": args.dry_run,
        "jobs": jobs,
        "selection": benchmark_config["selection"],
        "execution": benchmark_config["execution"],
    }

    if args.validate_config:
        if not args.quiet:
            print(f"Base config OK: {base_config_path}")
            print(f"Benchmark config OK: {benchmark_config_path}")
            print(f"Expanded runs: {len(runs)}")
            print(f"Output directory: {output_dir}")
            print(f"Jobs: {jobs}")
        return

    if args.html_only:
        tables = read_benchmark_output_tables(output_dir)
        html_path = write_benchmark_html(tables, output_dir, metadata)
        if args.quiet:
            print(html_path)
        else:
            print(f"Rebuilt benchmark HTML from existing CSV outputs: {html_path}")
        return

    if args.dry_run:
        tables = _empty_benchmark_tables(plan)
        written = write_benchmark_outputs(
            tables,
            output_dir,
            metadata,
            write_html=bool(benchmark_config["execution"].get("write_html", True)),
        )
        for path in written.values():
            print(path)
        return

    if not args.quiet:
        print(f"Running {len(runs)} trail digital-twin benchmark experiments")
        print(f"Base config: {base_config_path}")
        print(f"Benchmark config: {benchmark_config_path}")
        print(f"Output: {output_dir}")
        print(f"Jobs: {jobs}")

    fail_fast = bool(benchmark_config["execution"].get("fail_fast", False))
    run_results: list[BenchmarkRunResult] = []
    if jobs == 1:
        for run in runs:
            if not args.quiet:
                print(f"[{run.run_id}] {run.group} / {run.name}")
            result = execute_benchmark_run(
                run,
                benchmark_config["selection"],
                project_root=REPO_ROOT,
                config_path=base_config_path,
            )
            run_results.append(result)
            if result.manifest_row["status"] == "failed":
                if not args.quiet:
                    print(f"[{run.run_id}] failed: {result.manifest_row['error']}")
                if fail_fast:
                    raise SystemExit(f"Benchmark run failed: {run.run_id}: {result.manifest_row['error']}")
    else:
        executor = ProcessPoolExecutor(max_workers=jobs)
        shutdown_done = False
        try:
            futures = {
                executor.submit(
                    execute_benchmark_run,
                    run,
                    benchmark_config["selection"],
                    project_root=REPO_ROOT,
                    config_path=base_config_path,
                ): run
                for run in runs
            }
            for future in as_completed(futures):
                run = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - record process-pool failures in the manifest.
                    result = _failed_worker_result(run.run_id, run.group, run.name, run.output_dir, str(exc))
                run_results.append(result)
                if not args.quiet:
                    status = result.manifest_row["status"]
                    print(f"[{run.run_id}] {status}")
                    if status == "failed":
                        print(f"[{run.run_id}] failed: {result.manifest_row['error']}")
                if result.manifest_row["status"] == "failed" and fail_fast:
                    for pending in futures:
                        pending.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    shutdown_done = True
                    raise SystemExit(f"Benchmark run failed: {run.run_id}: {result.manifest_row['error']}")
        finally:
            if not shutdown_done:
                executor.shutdown()

    results_by_id = {result.run_id: result for result in run_results}
    ordered_results = [results_by_id[run.run_id] for run in runs if run.run_id in results_by_id]
    per_run_tables = [result.tables for result in ordered_results if result.tables]
    manifest_rows = [result.manifest_row for result in ordered_results]

    manifest = pd.DataFrame(manifest_rows)
    tables = combine_benchmark_tables(per_run_tables, manifest, plan)
    written = write_benchmark_outputs(
        tables,
        output_dir,
        metadata,
        write_html=bool(benchmark_config["execution"].get("write_html", True)),
    )
    if args.quiet:
        for path in written.values():
            print(path)
    else:
        print("\nWritten benchmark outputs:")
        for name, path in sorted(written.items()):
            print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
