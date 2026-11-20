# Reproduire les expériences RunningManager

Voir le [rapport de résultats](REPORT.md). `root/PLAN.md` contient le plan initial et son journal de mise à jour. Les répertoires `archive_pr5`, `rejeu_pr5` et `rejeu_corrige_pr5` contiennent les résultats agrégés. `root/pr5_audit_corrections.patch` est le patch local d'audit au commit de tête de la PR #5. Il a passé les 54 tests ciblés. Les scripts régénèrent les prédictions et données dérivées localement. L'archive contient des tableaux agrégés et du code, aucun enregistrement individuel de trajectoire.

## Clones et rejeux de la PR

Copier ce répertoire dans un dossier de travail, puis y exécuter les commandes avec Git, Python 3.12, uv et une connexion à GitHub.

```bash
git clone --single-branch --branch cursor/segment-outlier-optimization-7aaf https://github.com/piromagnus/RunningManager.git repo
git -C repo checkout --detach 5cc439ffbdbacfc66c86a733ec4ace942b4647f6
git clone --single-branch --branch cursor/add-portable-data-8df1 https://github.com/piromagnus/RunningManager.git portable
git -C portable checkout --detach 830ab045367fc91bd77fb40ccdb1316b2b2db88a
git -C repo worktree add --detach ../review_fix HEAD
git -C review_fix apply ../root/pr5_audit_corrections.patch
(cd repo && uv sync)
python root/configure_replay.py repo portable root/pr5_replay.yaml root/pr5_replay
python root/configure_replay.py review_fix portable root/pr5_review_fix.yaml root/pr5_review_fix
(cd repo && .venv/bin/python scripts/validate_intensity_conditioned_claim.py --config ../root/pr5_replay.yaml)
(cd review_fix && ../repo/.venv/bin/python scripts/validate_intensity_conditioned_claim.py --config ../root/pr5_review_fix.yaml)
(cd repo && .venv/bin/python -m pytest -q tests/test_validate_intensity_conditioned_claim.py tests/test_trail_performance_model.py)
```

La suite complète a été exécutée avec pandas 3.1.1. Le rejeu des expériences PR a utilisé pandas 2.2.3, numpy 2.3.5, plotly 7.1.0 et PyYAML. Si le rejeu diverge dans l'environnement du dépôt, utiliser un environnement séparé avec ces versions. L'ancienne graine bootstrap E4 dépend de `hash()` dans le code original. Ses intervalles peuvent varier légèrement d'un processus à l'autre. Les mesures ponctuelles sont déterministes. Les sorties des rejeux incluent localement des identifiants du dépôt et peuvent rester dans un répertoire de travail privé.

## Nouvelles expériences

Depuis ce même dossier de travail, avec numpy, pandas, scipy et, pour l'export, plotly et PyYAML. Les scripts du benchmark supposent la disposition de répertoires de l'archive.

```bash
python root/export_replay_segments.py repo root/pr5_replay.yaml root/replay_segments.csv
python fatigue/replay_benchmark.py
python fatigue/replay_segment_sum_benchmark.py
python residuals/forward_diagnostic.py
python residuals/sensitivity_diagnostic.py
python root/eta_to_finish.py fatigue/replay root/eta_to_finish.csv
python events/audit_events.py > events/event_support_replay.json
python synthetic/check_synthetic.py

curl -fsSL https://raw.githubusercontent.com/piromagnus/RunningManager/4c5cc8be3a10c2bc037f551932e0d07cf2103f7c/docs/science/paper_assets/anonymized_segment_features.csv -o fatigue/segments.csv
cp fatigue/segments.csv residuals/segments.csv
python fatigue/benchmark.py
python residuals/diagnostic.py
```

Le SHA-256 attendu pour l'export source `anonymized_segment_features.csv` est `1331ba84dd6c246000431cf40efc92aa506d49beae5f74b813dee0f56f757eac`. Voir les README d'expérience pour les hypothèses et les cibles. Les résultats de fatigue fondés sur la durée totale et ceux fondés sur la somme des temps des segments utilisent des cibles différentes.
