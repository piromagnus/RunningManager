# Paper assets — HR digital twin for trail-running performance

Publication-oriented figures and tables derived from the section-7 experiments.
Tone and labelling follow the working manuscript `docs/science/trail_digital_twin_hr_performance_paper_draft.md`.

## Regenerating

```bash
uv run python scripts/prepare_paper_assets.py
```

## Figures

PNG exports (2× scale) are stored under `figures/`. Scientific captions: `figures/figure_captions.md`.

- `fig_bland_altman_hardRunOrTrailRun.png`
- `fig_bland_altman_hardTrailRun.png`
- `fig_bland_altman_runTrailOver20Min.png`
- `fig_bland_altman_selectedDateRaces.png`
- `fig_component_ablation_delta_mae.png`
- `fig_incremental_model_mae.png`
- `fig_pred_vs_actual_hardRunOrTrailRun.png`
- `fig_pred_vs_actual_hardTrailRun.png`
- `fig_pred_vs_actual_runTrailOver20Min.png`
- `fig_pred_vs_actual_selectedDateRaces.png`
- `fig_segment_gap_optimisation_bias.png`
- `fig_segment_gap_optimisation_mae.png`
- `fig_segment_rejection_policies.png`
- `fig_segment_vs_race_objective.png`
- `fig_speed_vs_hrr.png`
- `fig_speed_vs_hrr_climb10pct.png`
- `fig_speed_vs_hrr_descent10pct.png`
- `fig_speed_vs_hrr_flat.png`
- `legacy_fig_hrr_speed_residual.png`
- `legacy_fig_loo_residuals_drivers.png`
- `legacy_fig_model_stage_flow.png`
- `legacy_fig_regression_coefficients.png`
- `legacy_fig_robustness_heatmaps.png`
- `legacy_fig_stage3_ablation.png`
- `legacy_fig_stage3_fatigue_state_comparison.png`
- `legacy_fig_stage_predicted_vs_actual.png`

## Tables

CSV + Markdown under `tables/`.

- `table01_cohort_descriptives.csv` / `table01_cohort_descriptives.md`
- `table02_incremental_model_loo.csv` / `table02_incremental_model_loo.md`
- `table03_component_ablation.csv` / `table03_component_ablation.md`
- `table04_bootstrap_uncertainty.csv` / `table04_bootstrap_uncertainty.md`
- `table05_prospective_predictions.csv` / `table05_prospective_predictions.md`
- `table06_speed_vs_hrr_flat.csv` / `table06_speed_vs_hrr_flat.md`
- `table07_segment_rejection_summary.csv` / `table07_segment_rejection_summary.md`
- `table07b_segment_rejection_examples.csv` / `table07b_segment_rejection_examples.md`
- `table08_segment_gap_optimisation.csv` / `table08_segment_gap_optimisation.md`
- `table09_segment_vs_race_objective.csv` / `table09_segment_vs_race_objective.md`

## Model ladder (for Table 2)

| ID | Addition relative to previous |
|----|-------------------------------|
| M0 | Baseline physics twin (GAP, altitude, CTL, progress fatigue) |
| M1 | + acute TRIMP fatigue (replaces progress decay) |
| M2 | + REDI readiness (replaces CTL) |
| M3 | + continuous HRR effort term |
| Full | M3 + asymmetric trail GAP soft-ramp scales |

