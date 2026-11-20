# Diagnostic des résidus de segments

Données sources du dépôt `piromagnus/RunningManager`, commit `4c5cc8be3a10c2bc037f551932e0d07cf2103f7c`, chemin `docs/science/paper_assets/anonymized_segment_features.csv`. Les octets de `segments.csv` correspondent au fichier de ce commit. SHA-256 `1331ba84dd6c246000431cf40efc92aa506d49beae5f74b813dee0f56f757eac`. L'export utilise des identifiants de sortie pseudonymisés. Il contient des données de performance par segment et peut permettre une réidentification par recoupement.

Dans ce répertoire, `summary.json` est agrégé par modèle sur les sorties et segments. `diagnostic.py` contient du code sans données de sortie. `segments.csv`, `pair_predictions.csv`, `per_activity.csv` et `fold_parameters.csv` conservent un identifiant de sortie pseudonymisé et des valeurs individuelles dérivées. `pair_predictions.csv` contient les temps et prédictions par segment. La pseudonymisation ne garantit pas l'anonymat. Ne diffuser ces tableaux qu'après examen de la granularité et des règles du projet.

Reproduction depuis une copie Git du dépôt disposant du commit et un environnement Python avec NumPy, pandas et SciPy. Adapter uniquement les deux chemins.

```bash
REPO=/chemin/vers/RunningManager
OUT=/chemin/vers/residuals
mkdir -p "$OUT"
git -C "$REPO" show 4c5cc8be3a10c2bc037f551932e0d07cf2103f7c:docs/science/paper_assets/anonymized_segment_features.csv > "$OUT/segments.csv"
cp diagnostic.py "$OUT/diagnostic.py"
(cd "$OUT" && sha256sum segments.csv diagnostic.py > source_manifest.sha256)
python "$OUT/diagnostic.py"
(cd "$OUT" && sha256sum summary.json per_activity.csv fold_parameters.csv pair_predictions.csv > output_manifest.sha256)
```

Le code produit un diagnostic rétrospectif par activité tenue à part. Les prédictions de base ont été ajustées sur les données sources et utilisent la HRR observée. Elles incluent donc des informations de la sortie évaluée. La correction d'un segment utilise les résidus passés, et ne constitue pas une prévision pré-course. Les intervalles empiriques sont conditionnés au segment précédent et calibrés sur les autres sorties. Leur couverture est calculée au niveau des segments, avec des échantillons dépendants.
