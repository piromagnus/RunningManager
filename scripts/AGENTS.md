# Scripts

Runnable analysis and maintenance tools.

## Files

| File | Purpose |
|------|---------|
| `estimate_prerace_hrr.py` | GPX pre-race estimate by constant HRR sweep using the trail digital-twin model |
| `trail_digital_twin_extensions.py` | YAML-configured trail digital-twin fitting, CSV export, and HTML report |
| `trail_digital_twin_benchmark.py` | Large benchmark sweep runner for trail digital-twin config variants |

## Conventions

- Keep scripts runnable from the repository root with `uv run python`.
- Reuse domain helpers from `services/` instead of duplicating model logic.
- Write outputs only when an explicit `--output-dir` is provided.
- Do not print secrets or raw platform tokens.

## Trail Digital Twin Extensions

```bash
uv run python scripts/trail_digital_twin_extensions.py \
  --config configs/trail_digital_twin_extensions.yaml
```

- Config owns paths, cohorts, physiology/readiness weights, grids, fitting objectives, robustness, and output toggles
- Default output directory is configured in `configs/trail_digital_twin_extensions.yaml`
- Use `--output-dir /path/to/run` for ad hoc runs that should not update science assets
- Use `--jobs N` to run independent cohort/objective Stage 0-3 fits in parallel; benchmark runs force this inner setting back to 1
- Use `--validate-config` to check YAML without fitting models

## Trail Digital Twin Benchmark

```bash
uv run python scripts/trail_digital_twin_benchmark.py \
  --base-config configs/trail_digital_twin_extensions.yaml \
  --benchmark-config configs/trail_digital_twin_benchmark.yaml
```

- Benchmark config expands the compact best-setup fatigue-floor and decay-lambda sweeps by default
- Default benchmark disables per-run HTML; tune `disable_inner_robustness` and `disable_inner_segment_grid` for faster screening runs
- Use `--dry-run` to write the expanded plan without fitting models
- Use `--html-only` to rebuild the aggregate benchmark HTML report from saved CSVs without fitting models
- Use `--max-runs N` for smoke tests before the full sweep
- Use `--jobs N` to execute independent benchmark runs in parallel; the default is `execution.jobs` from the benchmark YAML
- The benchmark includes both `hardTrailRun` and mixed `hardRunOrTrailRun` cohort coverage
