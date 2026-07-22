# Idées détaillées des expériences du dépôt (RQ, E, M, R, H, B, D)

Document de travail — **travail courant de ce repo**, pas la bibliographie externe.  
Companion : `idees_travaux_recents_detaillees_fr.md` (littérature), `journal_segment_hrr_adaptive.md`, `robustness_experiments_report.md`, `journal_steep.md`.  
Date : 2026-07-21.

**Attention aux homonymes :** deux séries **H1–H5** existent.

| Préfixe | Domaine | Journal / rapport |
|---------|---------|-------------------|
| **GAP-H*** | Soft-ramp GAP montée/descente | `journal_steep.md` |
| **ADAPT-H*** | HRR adaptatif par famille de terrain (prospectif) | `journal_segment_hrr_adaptive.md` |

Ci-dessous : **GAP-H** et **ADAPT-H** pour éviter toute confusion. Les IDs code (`H1_mean_mod`, …) restent ceux des scripts.

---

## Carte globale

```text
RQ1–RQ3          questions de recherche du papier
E1–E6            protocoles d’évaluation (§ méthodes)
M0–M3            échelle de modèles (ladder Stage 3)
R1–R11           robustesse / cohérence publication
GAP-H1–H5        hypothèses GAP trail (journal_steep)
ADAPT-H1–H20     hypothèses HRR prospectif adaptatif
B1–B5            bloqués (données)
D1–D4            différés (modélisation optionnelle)
```

Chiffres LOO cités : pipeline §7 avec `hrr_max_factor=1.20` → hard run/trail M3 **MAE 8,25 min**, **MAPE 7,3 %** (sauf si une section note explicitement une ancienne valeur 9,09).

---

# 1. Questions de recherche (RQ1–RQ3)

### RQ1 — Le twin HRR+TRIMP bat-il la physique seule ?

| | |
|---|---|
| **Idée** | Après correction pente/altitude (et readiness), la trajectoire HRR + la fatigue TRIMP aiguë réduisent l’erreur LOO de temps final. |
| **Test** | Ladder M0→M3, LOO par cohortes (hard trail, hard run/trail, >20 min, race dates). |
| **Résultat** | Oui sur hard run/trail : M0 ≈ 25,7 min → M3 ≈ 8,25 min. |
| **Limite** | Reconstruction avec HR observée ≠ prédiction pré-course. |

### RQ2 — Où se concentrent les erreurs (terrain) ?

| | |
|---|---|
| **Idée** | Les résidus ne sont pas homogènes : montées/descentes raides vs plat. |
| **Test** | Métriques par `terrainFamily` ; optimisation GAP (E6 / GAP-H*). |
| **Résultat** | Soft-ramp 0,85 / 1,60 → MAE steep combiné ≈ −46 %. |
| **Limite** | Outliers alpins nommés (R8) ; technicité non mesurée (Nicot). |

### RQ3 — Même twin en mode prospectif (HRR prescrit) ?

| | |
|---|---|
| **Idée** | Sans HR de la course cible, une stratégie HRR (enveloppe durée ± adaptatif) produit des temps utilisables. |
| **Test** | Hold-outs LUT, Grésivaudan, Échappée, Passerelles (+ Rome négatif) ; Table 5 ; ADAPT-H*. |
| **Résultat** | Constant PL faisable : MAPE trail ≈ 7,5 % ; **ADAPT-H13b** ≈ 2,1 %. |
| **Limite** | Rome hors scope ; optimisme ultra si HRR trop haut. |

---

# 2. Protocoles d’évaluation (E1–E6)

### E1 — LOO sur cohortes d’activités

| | |
|---|---|
| **Idée** | Généralisation inter-activités (même athlète) : laisser une activité de côté, prédire son temps mobile. |
| **Métriques** | MAE, MAPE, biais, R² (minutes). |
| **Sortie** | Table 2, figures pred-vs-actual / Bland–Altman. |

### E2 — Résidus segmentaires par classe de terrain

| | |
|---|---|
| **Idée** | Diagnostiquer la physique locale (pas seulement le finish). |
| **Métriques** | MAE / biais (min) par flat, climb, steep_*, descent. |
| **Sortie** | Table segment type ; motive GAP-H*. |

### E3 — Prédictions prospectives constant-HRR

| | |
|---|---|
| **Idée** | Hold-out courses ; balayer HRR ; enveloppe durée ; comparer aussi HRR_ref et (éval-only) mean HRR observé. |
| **Métriques** | Δ min, bandes P05/P50/P95 (R9). |
| **Sortie** | Table 5. |

### E4 — Ablations de composants avec re-optimisation (α, κ)

| | |
|---|---|
| **Idée** | Mesurer la contribution *récupérable* d’un terme en re-fittant après retrait (pas un freeze naïf). |
| **Alignement** | Ligne « full » = MAE ladder M3 (R1). |
| **Résultat typique** | −HRR et −TRIMP dominent ; GAP fort ; REDI/altitude secondaires. |

### E5 — Rejet léger de segments (A/B)

| | |
|---|---|
| **Idée** | Exclure des segments quasi-immobiles / plats parasites du fit, surtout avec scrubbing du temps mobile. |
| **Résultat** | Avec moving-time : peu de rejects restants (~0,17 %) ; headline LOO stable. |

### E6 — Optimisation segmentaire (GAP trail ; objectif segment vs race)

| | |
|---|---|
| **Idée** | Calibrer `gap_climb_scale` / `gap_descent_scale` sur résidus terrain ; comparer fit objectif segment vs activité. |
| **Résultat** | Scales 0,85 / 1,60 ; objectif activité préféré pour le prospectif (R2). |

---

# 3. Échelle de modèles (M0–M3)

Loi de vitesse (schéma) :

\[
v \propto v_{\mathrm{VMA}}\cdot\alpha\cdot f_{\mathrm{alt}}\cdot f_{\mathrm{load}}\cdot E(\mathrm{HRR})\cdot F / f_{\mathrm{GAP}}
\]

### M0 — Baseline physique

| | |
|---|---|
| **Idée** | Jumeau « Jaén-like » sans HR continue : GAP/Minetti, altitude, readiness CTL, fatigue de progression. |
| **Rôle** | Référence pour ΔMAE ; montre l’erreur sans canal interne. |
| **Chiffre** | Hard run/trail MAE ≈ **25,7 min**. |

### M1 — + TRIMP aigu (à la place de la fatigue de progression)

| | |
|---|---|
| **Idée** | Cumul de charge type Banister *dans* l’activité comme état de fatigue. |
| **Résultat** | Améliore le mixte ; peut **empirer** le hard-trail seul sans HRR. |
| **Chiffre** | Hard run/trail ≈ **21,6 min**. |

### M2 — + readiness REDI (remplace CTL)

| | |
|---|---|
| **Idée** | Autre proxy de fraîcheur entre séances. |
| **Résultat** | Gain marginal vs M1 ici. |
| **Chiffre** | ≈ **21,8 min**. |

### M3 — + effort HRR continu (twin complet Stage 3)

| | |
|---|---|
| **Idée** | \(E=\mathrm{clip}(\mathrm{HRR}/\mathrm{HRR}_{\mathrm{ref}}, h_{\min}, h_{\max})\) avec \(\mathrm{HRR}_{\mathrm{ref}}=0{,}88\), \(h_{\max}=1{,}20\). |
| **Résultat** | Gain dominant ; HRR et TRIMP tous deux nécessaires (ablations). |
| **Chiffre** | Hard run/trail **8,25 min** ; hard trail **17,4** ; >20 min **5,91** ; race dates **11,8**. |

---

# 4. Robustesse (R1–R11)

### R1 — Cohérence Table 2 / Table 3 (« full »)

| | |
|---|---|
| **Idée** | La ligne « full » des ablations doit réutiliser le MAE LOO du ladder M3, sinon les ΔMAE sont illisibles. |
| **Statut** | PASS — MAE identiques (désormais 8,25 min hard run/trail @ max_factor 1,20). |

### R2 — Objectif de fit figé sans « peeking » hold-outs

| | |
|---|---|
| **Idée** | Choisir activity vs segment sur LOO hard mixte **sans** utiliser les courses prospectives. |
| **Résultat** | Objectif **activity** (meilleur pour finish prospectif ; segment meilleur sur certaines dates in-sample). |

### R3 — Inventaire des hold-outs prospectifs

| | |
|---|---|
| **Idée** | Liste explicite : 4 trails (LUT, Grésivaudan, Échappée, Passerelles) + Rome ; profils GPX ou timeseries. |
| **Statut** | PASS — pas de fuite dans le fit / enveloppe. |

### R4 — Budget d’aide post-hoc (non fitté)

| | |
|---|---|
| **Idée** | Une partie de l’optimisme (temps mobile) = ravitaillements / pauses non modélisés. |
| **Résultat** | Grésivaudan Δ @ HRR_ref : −17 → ≈ −5 min avec ~12 min d’aide estimée. |
| **Limite** | Budget coach, pas log CSV ; n’explique pas Échappée à HRR trop haut. |

### R5 — Route hors scope

| | |
|---|---|
| **Idée** | Les scales GAP trail rendent un marathon route catastrophique (Rome Δ ≈ −79 min) ; ce n’est pas un échec du twin trail, c’est hors domaine. |
| **Statut** | PASS — Rome = contrôle négatif / out of scope. |

### R6 — LOO bloqué par date de course

| | |
|---|---|
| **Idée** | Agréger l’erreur par jour de course (éviter de surpondérer des doubles sorties). |
| **Résultat** | MAE ≈ 11,8 min (IC 90 % ≈ 6,5–18,1). |

### R7 — Sensibilité QC HR

| | |
|---|---|
| **Idée** | Si beaucoup d’activités ont un faible `hrValidShare`, les résultats HRR sont fragiles. |
| **Résultat** | Toutes à 1,0 → pas de levier QC ici. |

### R8 — Outliers steep-climb nommés

| | |
|---|---|
| **Idée** | Accepter le scatter steep en documentant les activités dominantes (7 Laux, KV reco, …). |
| **Statut** | PASS — transparence, pas de suppression silencieuse. |

### R9 — Bandes d’incertitude de finish non dégénérées

| | |
|---|---|
| **Idée** | P05 < P50 < P95 avec spread ≥ ~3 min (jitter α/κ + bruit résiduel LOO). |
| **Usage** | Enveloppes coaching Table 5. |

### R10 — Alignement headline §7

| | |
|---|---|
| **Idée** | Une seule config « source of truth » (`trail_digital_twin_paper_section7.yaml`) pour chiffres papier. |
| **Actuel** | MAE ≈ **8,25 min** avec GAP 0,85/1,60 et `hrr_max_factor=1,20`. |

### R11 — Plancher de κ (fatigue)

| | |
|---|---|
| **Idée** | Sur hard trail, la grille pousse κ au plancher ; baisser le plancher empirait le LOO. |
| **Résultat** | κ=0,20 **contraignant mais protecteur** (17,4 vs 19,7 si on autorise κ=0,10). |

*(Pas de R12 dans le rapport actuel ; la suite est B*/D*.)*

---

# 5. Hypothèses GAP trail (GAP-H1–H5) — `journal_steep.md`

### GAP-H1 — Minetti trop optimiste en descente trail

| | |
|---|---|
| **Idée** | Sur descente technique, le coût Minetti sous-estime le temps → multiplier GAP par `gap_descent_scale > 1` (soft ramp 4 %→15 %). |
| **Verdict** | **CONFIRMÉ** — scale ≈ **1,60**. |

### GAP-H2 — Minetti trop cher en montée raide trail

| | |
|---|---|
| **Idée** | En montée raide, le coureur « marche / s’adapte » mieux que le coût tapis → `gap_climb_scale < 1`. |
| **Verdict** | **CONFIRMÉ** (biais moyen) — scale ≈ **0,85**. |

### GAP-H3 — Asymétrie montée/descente nécessaire

| | |
|---|---|
| **Idée** | Un seul facteur ne corrige pas les deux biais opposés. |
| **Verdict** | **CONFIRMÉ** — paire 0,85 / 1,60. |

### GAP-H4 — Interaction grade × HRR sur steep climbs

| | |
|---|---|
| **Idée** | Effort linéaire en HRR insuffisant quand la pente est extrême (hétéroscédasticité). |
| **Verdict** | **DIFFÉRÉ** (= D1) — biais moyen déjà ~0 après GAP-H2. |

### GAP-H5 — Technicité (au-delà de la pente)

| | |
|---|---|
| **Idée** | Index de technicité GPS / surface (Nicot) comme facteur additionnel. |
| **Verdict** | **DIFFÉRÉ** — secondaire tant que GAP soft-ramp suffit. |

---

# 6. Hypothèses HRR adaptatif prospectif (ADAPT-H1–H20)

Contexte commun :

- 5 familles : `flat`, `climb`, `steep_climb`, `descent`, `steep_descent`.
- Cible moyenne HRR★ (souvent enveloppe durée) × modulateur \(m_f\) (données hors hold-outs).
- Simulation TRIMP séquentielle à HRR par segment.
- **Métrique primaire actuelle : MAPE** ; prédiction plus rapide que le réel = acceptable.
- Baseline constant PL faisable : MAPE trail ≈ **7,52 %**.

## 6.1 Modulation seule (ADAPT-H1–H5)

### ADAPT-H1 — \(m_f\) = moyenne famille × HRR★ PL

| | |
|---|---|
| **Idée** | Redistribuer l’effort comme en entraînement (montée un peu plus haute HRR, descente raide plus basse), en gardant la moyenne = faisable PL. |
| **MAPE** | ≈ 7,43 % | **Verdict** | Échec — \(m_f\) trop doux (~±6 %). |

### ADAPT-H2 — \(m_f\) = p75 famille

| | |
|---|---|
| **Idée** | Même logique avec statistiques « effort dur » (p75). |
| **MAPE** | ≈ 7,25 % | **Verdict** | Marginal ≈ constant. |

### ADAPT-H3 — Plafonds par famille via power-law sur \(T_f\)

| | |
|---|---|
| **Idée** | Capper HRR\(_f\) par la durée passée sur la famille \(f\). |
| **MAPE** | ≈ 7,42 % | **Verdict** | Caps presque jamais actifs. |

### ADAPT-H4 / H4b — Cohort des modulateurs

| | |
|---|---|
| **Idée** | hardTrailRun seul vs tout run/trail utilisable. |
| **Verdict** | Préférer **hardTrail** (H4b pire, MAPE ≈ 7,83 %). |

### ADAPT-H5 — Supra-VMA (\(E>1\)) uniquement à plat

| | |
|---|---|
| **Idée** | Autoriser HRR > HRR_ref seulement sur flats. |
| **Verdict** | Sans effet aux cibles ≤ 0,88. |

## 6.2 Décalages de cible ratés (ADAPT-H6–H9, H11)

### ADAPT-H6 — Mods amplifiés + blend vers moyenne historique

| | |
|---|---|
| **Idée** | Forcer le contraste familles et tirer HRR★ vers l’historique « moyen ». |
| **MAPE** | ≈ 11 % | **Verdict** | Trop mou sur courses courtes (trop lent). |

### ADAPT-H7 — Bande historique + mods amplifiés

| | |
|---|---|
| **Idée** | Cible dans une bande historique plutôt que l’enveloppe max. |
| **MAPE** | ≈ 17 % | **Verdict** | Échec. |

### ADAPT-H8 / H8b — Correction résiduelle (obs − enveloppe) ~ log(durée) + D+/km

| | |
|---|---|
| **Idée** | Ajuster HRR★ par un modèle de résidu entraîné hors hold-outs. |
| **Problème** | Les sorties faciles ont un résidu très négatif → cible trop basse. |
| **MAPE** | ≈ 32 % | **Verdict** | Échec bruyant. |

### ADAPT-H9 / H9b / H9c — min(PL, fenêtre empirique) ou faisabilité empirique pure

| | |
|---|---|
| **Idée** | Les fenêtres empiriques baissent fort à 6–8 h (HRR≈0,62) vs PL≈0,75 à 10 h. |
| **H9** | min → Échappée trop lente (+55 min), MAPE ≈ 8,1 %. |
| **H9b** | Sweep avec \(T_{\max}\) empirique → sélection instable / trop basse sur courts. |
| **Verdict** | Direction bonne pour l’ultra, trop brutale seule. |

### ADAPT-H11 — Switch dur vers empirique si \(T≥6\) h

| | |
|---|---|
| **Idée** | PL court ; empirique pur en ultra. |
| **MAPE** | ≈ 6,5 % | **Verdict** | Même sur-correction ultra que H9. |

## 6.3 Blend durée (ADAPT-H10, H12) — premier vrai gain MAPE

### ADAPT-H10 — Blend power-law / empirique

\[
w=\mathrm{clip}\big((T_h-4)/(10-4),0,1\big)\cdot w_{\max},\quad
\mathrm{HRR}^\star=(1-w)\,\mathrm{HRR}_{\mathrm{PL}}+w\,\mathrm{HRR}_{\mathrm{emp}}(T)
\]

| Variante | MAPE | Note |
|----------|------|------|
| w50 + p75 | ≈ 5,15 % | Ultra encore un peu optimiste |
| **w70 + p75** | **≈ 4,92 %** | Bon compromis |
| w70 + mean | ≈ 4,97 % | Quasi égal |
| w85 / w100 | 5,7–6,5 % | Trop d’empirique → lent |

**Idée centrale :** le PL est trop plat pour les ultras ; les fenêtres empiriques portent l’info longue durée ; \(w_{\max}=0{,}70\) évite le sur-correctif H9.

### ADAPT-H12 — H10 + une réévaluation empirique

| | |
|---|---|
| **Idée** | Recalculer \(\mathrm{HRR}_{\mathrm{emp}}\) au temps prédit après un premier blend. |
| **Résultat** | Ici ≈ H10 w70 p75 (pas de second ordre utile). |

## 6.4 Poussée courses courtes (ADAPT-H13–H20) — MAPE primaire

Problème restant après H10 : courses ≤ ~3–4 h encore **trop lentes** (+6–9 % MAPE). Sous « optimistic OK », on **monte** HRR★ court vers l’effort VMA.

### ADAPT-H13 — Court = HRR_ref ; long = H10

| | |
|---|---|
| **Idée** | Tout le temps à effort VMA plat si \(T<5\) h. |
| **MAPE** | ≈ 4,10 % | Ultra OK ; courts parfois trop optimistes (Grésivaudan). |

### ADAPT-H13b — Court = mid(PL, HRR_ref) ; long = H10  ★ préféré

| | |
|---|---|
| **Idée** | Poussée de course à mi-chemin entre plancher faisable et effort VMA (inspiration VT2 / Jaén-Carrillo), sans paramètre libre type +0,06. |
| **MAPE** | **≈ 2,07 %** |
| **Détail** | LUT +3,0 % ; Grésivaudan −3,2 % (OK) ; Passerelles +0,6 % ; Échappée +1,5 %. |
| **Verdict** | **Meilleure recette principée.** |

### ADAPT-H14 — Court = max(PL, cible fenêtre 60 min)

| | |
|---|---|
| **Idée** | Ancre sur le max historique ≥ 60 min (~0,80). |
| **MAPE** | ≈ 2,99 %. |

### ADAPT-H15 — Court 0,82 ; long mix Fornasiero / empirique

| | |
|---|---|
| **Idée** | Ancre ultra type ~77 % HRmax (Fornasiero) + cible courte fixe. |
| **MAPE** | ≈ 2,56 %. |

### ADAPT-H16 — Boost « style CP » vers HRR_ref puis H10

| | |
|---|---|
| **Idée** | \(\mathrm{HRR}=\mathrm{PL}+\frac{\max(0,4-T_h)}{4}(\mathrm{HRR}_{\mathrm{ref}}-\mathrm{PL})\). |
| **MAPE** | ≈ 3,66 %. |

### ADAPT-H17 — Court = 0,85 ; long = H10

| | |
|---|---|
| **Idée** | Cible courte haute fixe. |
| **MAPE** | ≈ 2,26 % | Plus optimiste (2 courses). |

### ADAPT-H18 — Constant HRR_ref + mods

| | |
|---|---|
| **Idée** | Scénario « tout à E=1 ». |
| **MAPE** | ≈ 9,8 % | Ultra trop rapide pour le MAPE. |

### ADAPT-H19 / H20 — H10 + offset court (+0,04 / +0,06)

| | |
|---|---|
| **Idée** | Compenser le biais lent du modèle sur hold-outs courts. |
| **MAPE** | H19 ≈ 2,37 % ; **H20 ≈ 2,00 %** (meilleur chiffre brut). |
| **Verdict** | H20 ≈ tied avec H13b mais **offset tenu sur 4 hold-outs** ; calibration training 2–5 h ne demande pas ce boost (souvent déjà optimiste). → **sensibilité seulement**, préférer H13b. |

---

# 7. Bloqués (B1–B5)

| ID | Idée | Bloqueur |
|----|------|----------|
| **B1** | Réplication multi-athlètes | Un seul athlète dans le repo |
| **B2** | Strates sexe / âge / niveau | Idem |
| **B3** | Altitude corrigée DEM | Pas de pipeline DEM |
| **B4** | Multiplicateur chaleur / météo | Couverture température sparse |
| **B5** | Logs d’aide / nutrition structurés | Pas dans les CSV (R4 = proxy coach) |

---

# 8. Différés (D1–D4)

| ID | Idée | Lien |
|----|------|------|
| **D1** | Interaction grade × HRR | = GAP-H4 |
| **D2** | Banister long terme fitness–fatigue inter-séances | Au-delà REDI/CTL |
| **D3** | Coût excentrique descente dédié | Au-delà `gap_descent_scale` |
| **D4** | Fusion chaleur / RPE si HR sature | Besoin météo + RPE |

---

# 9. Recettes « état de l’art » actuel du repo

### Reconstruction LOO (HR observée)

**M3** + GAP 0,85/1,60 + `hrr_max_factor=1,20` + objectif activity + κ≥0,20.

### Prospectif constant (Table 5)

Enveloppe **duration-feasible** power-law ; compagnon HRR_ref ; éval-only mean HRR observé.

### Prospectif adaptatif (meilleur MAPE principé)

**ADAPT-H13b** : si \(T_{\mathrm{PL}}<5\) h → mid(PL, 0,88) ; sinon blend H10 \(w_{\max}=0{,}70\) ; puis \(m_f\) hardTrail ; simuler TRIMP.

---

# 10. Index rapide ID → une phrase

| ID | Une phrase |
|----|------------|
| RQ1 | HRR+TRIMP bat la physique seule en LOO. |
| RQ2 | Erreurs terrain-dépendantes. |
| RQ3 | Prospectif par HRR prescrit possible. |
| E1–E6 | Protocoles LOO / terrain / prospectif / ablation / rejet / GAP. |
| M0–M3 | Ladder physique → +TRIMP → +REDI → +HRR. |
| R1 | Ablation full = ladder MAE. |
| R2 | Objectif activity sans peeking. |
| R3 | Hold-outs inventoriés. |
| R4 | Aide post-hoc explique une partie de l’optimisme. |
| R5 | Rome hors scope. |
| R6 | LOO par date. |
| R7 | QC HR non informatif ici. |
| R8 | Outliers steep nommés. |
| R9 | Bandes finish non dégénérées. |
| R10 | Config §7 = vérité. |
| R11 | Plancher κ protecteur. |
| GAP-H1–H3 | Descent×1,60, climb×0,85, asymétrie. |
| GAP-H4–H5 | Différés (interaction / technicité). |
| ADAPT-H1–H5 | Mods famille trop faibles seuls. |
| ADAPT-H6–H9,H11 | Mauvaises cibles / trop d’empirique brut. |
| ADAPT-H10 | Blend PL/emp → MAPE ~5 %. |
| ADAPT-H13b | Mid(PL,ref) court + H10 → MAPE ~2,1 % ★ |
| ADAPT-H20 | +0,06 court → MAPE ~2,0 % (sensibilité). |
| B* / D* | Multi-athlète / météo / modèles futurs. |

---

## Fichiers associés

| Fichier | Contenu |
|---------|---------|
| `trail_digital_twin_hr_performance_paper_draft.md` | RQ, E, M, R checklist |
| `robustness_experiments_report.md` | R1–R11 détaillé |
| `journal_steep.md` | GAP-H1–H5 |
| `journal_segment_hrr_adaptive.md` | ADAPT-H1–H20 + MAPE |
| `remaining_experiments.md` | B*, D* |
| `scripts/predict_race_segment_hrr_adaptive.py` | ADAPT-H exécutable |
| `scripts/run_robustness_experiments.py` | Suite R |
| `configs/trail_digital_twin_paper_section7.yaml` | Physiologie §7 |
