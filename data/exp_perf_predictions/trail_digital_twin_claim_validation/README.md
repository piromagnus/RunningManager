# Intensity-conditioned trail twin claim validation

Artifacts from `uv run python scripts/validate_intensity_conditioned_claim.py`.

Protocol: `docs/plan/intensity_conditioned_trail_twin_claim_validation.md`

## Key outputs

| File | Experiment |
|------|------------|
| `freeze_manifest.json` | Input hashes, HR rest/max, physiology freeze |
| `table_data_support.csv` | E0 support / event labels / HR sensitivity |
| `table_intensity_premise_e1.csv` | E1 intensity residual after course/duration |
| `table_prescribed_hrr_model_comparison.csv` | E2 B0–B3 + U1 summary |
| `table_prescribed_hrr_folds_hardTrailRun.csv` | E2 per-activity folds |
| `table_route_pair_validation.csv` | E3 coarse repeated-route pairs |
| `table_rolling_origin_hrr_selection.csv` | E4 history-only HRR rules |
| `table_causal_fatigue_ablation.csv` | E5 predicted-time fatigue variants |
| `claim_gates.csv` | pass / fail / not_testable per claim |

## Decision table (latest run)

See `claim_gates.csv` row `paper_story`.
