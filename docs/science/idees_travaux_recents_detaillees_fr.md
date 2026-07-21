# Idées détaillées des travaux récents et de la bibliographie enrichie

Document de travail en français.  
Objectif : expliciter **chaque idée** issue des travaux *nouveaux* ou récemment intégrés dans la bibliographie / *related work* (surtout 2020–2026), au-delà d’une simple fiche article.  
Sources : `bibliography_hr_digital_twin.md`, `literature_guide_private_notes_fr.md`, `related_work_paper_ready.md`.  
Date : 2026-07-21.

---

## Comment lire ce document

Pour chaque idée :

1. **Énoncé** — formulation claire de l’idée.
2. **Origine** — article(s) et contexte.
3. **Mécanisme / intuition** — pourquoi cela marche (ou limite).
4. **Implication pour notre jumeau** — ce que cela change dans le modèle Running Manager.
5. **Piège / formulation à éviter** — claim trop fort.

Les travaux « nouveaux » au sens du manuscrit sont surtout : **Jaén-Carrillo & Pattis (2026)**, **Holler & Jaén-Carrillo (2025)**, **Boillet (2024)**, **Gutiérrez (2025)**, **Feist (2026)**, **Genitrini (2024)**, plus les ajouts critiques **Nicot, Born, Lemire, Fogliato/TRAP, Emig, Fornasiero** qui structurent le *related work* enrichi.

---

## A. Digital twins et prédiction de course

### Idée A1 — Un jumeau numérique trail individualisé est déjà publié

| | |
|---|---|
| **Énoncé** | On peut calibrer, pour un même athlète, un modèle physique (pente, altitude, charge, pacing) et prédire d’autres courses en leave-one-race-out avec une erreur de l’ordre de ~10–20 min. |
| **Origine** | Jaén-Carrillo & Pattis, *Sensors* 2026 (13 courses, 1 athlète ; MAE 18,2 min, MAPE 11,1 %, R² 0,864). |
| **Mécanisme** | Le parcours impose un coût (GAP/Minetti) ; l’altitude réduit la capacité ; une charge type Banister et une décroissance de pacing capturent la dégradation ; deux paramètres individuels (fraction durable type VT2, pente de decay) absorbent l’athlète. |
| **Implication** | Notre papier est une **extension / ablation** de cette famille (HRR continue + TRIMP intra-activité), pas « le premier digital twin trail ». Les métriques LOO de 2026 sont le baseline direct. |
| **Piège** | Claim « first digital twin for trail running ». |

### Idée A2 — Paramètres individuels = fraction durable + decay de pacing

| | |
|---|---|
| **Énoncé** | L’individualisation utile d’un twin trail se concentre sur (i) la fraction d’intensité durable (proche VT2 / α) et (ii) la vitesse à laquelle le pacing se dégrade. |
| **Origine** | Jaén-Carrillo & Pattis (2026). |
| **Mécanisme** | Sur un profil donné, deux athlètes diffèrent surtout par le niveau d’effort qu’ils peuvent tenir et par la pente de fatigue/pacing, pas par la forme du coût de pente Minetti. |
| **Implication** | Chez nous : \(\alpha\) = fraction de VMA à \(E=1\) (HRR_ref) ; \(\kappa\) = sensibilité à la fatigue TRIMP. En prospectif, le choix d’un HRR★ joue le rôle d’« intensité durable prescrite » (cf. H13b : poussée courte vers HRR_ref). |
| **Piège** | Confondre \(\mathrm{HRR}_{\mathrm{ref}}\) (définition VMA) avec la fraction VT2 mesurée en labo. |

### Idée A3 — Un twin physiologique « hydraulique » exige beaucoup de labo et vise le court

| | |
|---|---|
| **Énoncé** | Un digital twin à compartiments métaboliques (Margaria–Morton) peut être individualisé, mais au prix de plusieurs tests et pour des efforts proches du max (~1–20 min), typiquement en cyclisme. |
| **Origine** | Boillet et al., *Sci. Rep.* 2024 (DOI correct : 10.1038/s41598-024-56042-0). |
| **Mécanisme** | Trois réservoirs d’énergie / puissance critique ; paramètres liés à VO2max, CP, efficacité, masse musculaire. |
| **Implication** | Justifie pourquoi notre voie (HRR + physique de parcours) est plus réaliste pour des **heures** de trail sans batterie de labo. CP/W′ reste une **inspiration** pour la capacité courte (boost d’HRR sur courses < 5 h), pas un modèle à importer tel quel. |
| **Piège** | Citer le mauvais DOI (10.1038/s41598-024-71772-x) ; prétendre un twin « aussi physiologique » que Boillet sans les mêmes mesures. |

### Idée A4 — Séparer demande mécanique du parcours et état interne du coureur

| | |
|---|---|
| **Énoncé** | On peut estimer analytiquement la puissance mécanique demandée par le relief (collisions, élastique, force-rate, haut du corps, terrain irrégulier) indépendamment de la fatigue cardiaque. |
| **Origine** | Holler & Jaén-Carrillo, *J. Biomech.* 2025. |
| **Mécanisme** | Le parcours fixe une demande ; l’athlète y répond avec une capacité et un état. Mélanger les deux dans un seul facteur GAP obscurcit les diagnostics. |
| **Implication** | Notre GAP soft-ramp (climb 0,85 / descent 1,60) est une correction empirique de Minetti ; Holler motive une future couche « technicité / puissance » séparée de HRR. |
| **Piège** | Présenter Holler comme un modèle validé de temps final multi-courses. |

---

## B. Physique du parcours et technicité

### Idée B1 — Le coût de pente est non linéaire et asymétrique

| | |
|---|---|
| **Énoncé** | Le coût métabolique de marche/course vs pente (≈ −45 % à +45 %) n’est ni linéaire ni symétrique montée/descente. |
| **Origine** | Minetti et al., *JAP* 2002. |
| **Mécanisme** | En montée le coût croît vite ; en descente il baisse puis remonte (freinage, chocs). |
| **Implication** | Fondation du GAP ; nos familles de terrain (flat / climb / steep_climb / descent / steep_descent) et modulateurs HRR par famille. |
| **Piège** | Appliquer Minetti tel quel au trail technique sans correction. |

### Idée B2 — À pente égale, le vrai trail coûte ~10 % plus cher que le tapis

| | |
|---|---|
| **Énoncé** | Sur montées appariées, le coût O₂ trail > tapis (~+10,5 %), avec plus de variabilité médio-latérale du pied. |
| **Origine** | Nicot et al., *EJSS* 2022. |
| **Mécanisme** | Surface irrégulière, appui, stabilité — pas capturés par la pente moyenne. |
| **Implication** | Un résidu « HR » peut être de la technicité non mesurée. Motiver gap_climb_scale / gap_descent_scale et diagnostics par famille. |
| **Piège** | Attribuer tout biais montée/descente à la fatigue TRIMP. |

### Idée B3 — Montée et descente n’ont pas les mêmes déterminants

| | |
|---|---|
| **Énoncé** | À HR moyenne proche, la montée est surtout cardio-métabolique ; la descente davantage mécanique / neuromusculaire. |
| **Origine** | Lemire et al., *JSAMS* 2021. |
| **Mécanisme** | Deux régimes de production de vitesse sous une même enveloppe HR. |
| **Implication** | Une seule courbe HRR→vitesse peut biaiser la descente ; d’où scales GAP asymétriques et modulateurs par famille. |
| **Piège** | Une seule « zone HR » pour tout le profil. |

---

## C. Fréquence cardiaque, HRR et limites du signal

### Idée C1 — %HRR ≈ %réserve de VO₂ (pas %VO₂max)

| | |
|---|---|
| **Énoncé** | Sur effort incrémental contrôlé, le pourcentage de réserve cardiaque suit la réserve de VO₂. |
| **Origine** | Swain & Leutholtz, *MSSE* 1997. |
| **Mécanisme** | Normalisation individuelle (HR − HRrepos) / (HRmax − HRrepos). |
| **Implication** | Loi d’effort \(E = \mathrm{clip}(\mathrm{HRR}/\mathrm{HRR}_{\mathrm{ref}},\,h_{\min},\,h_{\max})\) avec \(\mathrm{HRR}_{\mathrm{ref}}=0{,}88\). |
| **Piège** | Extraire de Swain que HRR = effort métabolique instantané en descente. |

### Idée C2 — Sur terrain vallonné, HR lisse et retarde ; VO₂ varie plus que HR

| | |
|---|---|
| **Énoncé** | En trail court intense, HR reste haute en descente alors que VO₂ chute nettement ; NIRS suit mieux la pente. |
| **Origine** | Born et al., *IJSPP* 2017. |
| **Mécanisme** | Inertie cardio-vasculaire, régulation centrale, lag. |
| **Implication** | Vocabulaire : *internal-response signal*, pas *ground-truth instantaneous effort*. Lags / lissage possibles en prospective. |
| **Piège** | « HRR measures true instantaneous effort ». |

### Idée C3 — Dérive cardiovasculaire et altitude déplacent la carte HR ↔ capacité

| | |
|---|---|
| **Énoncé** | Chaleur / hydratation : HR monte, VO₂max disponible baisse. Altitude : VO₂max ↓ (~6 %/1000 m), HRmax ↓ plus faiblement. |
| **Origine** | Wingo et al. 2005 ; Wehrlin & Hallén 2006. |
| **Mécanisme** | Même HR externe ≠ même intensité relative si la capacité max change. |
| **Implication** | Facteur altitude côté physique ; ne pas double-compter via HRR et correction VO₂ sans cohérence HRmax. |
| **Piège** | Interpréter toute hausse de HR en fin d’ultra comme « plus d’effort volontaire ». |

### Idée C4 — La qualité du capteur conditionne toute science HRR

| | |
|---|---|
| **Énoncé** | En trail variable, le PPG montre souvent un MAPE HR élevé (~13 %) ; une ceinture ECG ~2 %. |
| **Origine** | Navalta et al., *PLOS ONE* 2020. |
| **Mécanisme** | Artefacts de mouvement, mauvaise perfusion. |
| **Implication** | Exiger / documenter ceinture ; QC HR avant fit. |
| **Piège** | Publier des ablations HRR sans provenance capteur. |

### Idée C5 — On peut calibrer un seuil labo depuis un protocole trail standardisé

| | |
|---|---|
| **Énoncé** | Des secteurs outdoor GNSS–IMU + Polar prédisent la HR au seuil anaérobie labo (RMSE LOOCV ≈ 4,3 bpm). |
| **Origine** | Feist et al., *CISS* 2026 (**book of abstracts**, pas article complet). |
| **Mécanisme** | Lien terrain → seuil individuel sans gazométrie portable. |
| **Implication** | Future calibration de \(\mathrm{HRR}_{\mathrm{ref}}\) / zones ; pas encore un substitut à notre fit α,κ. |
| **Piège** | Citer Feist comme validation journal complète. |

---

## D. Charge, TRIMP et fatigue

### Idée D1 — Banister : dose → fitness & fatigue sur des jours/semaines

| | |
|---|---|
| **Énoncé** | Une impulsion d’entraînement alimente deux réponses antagonistes avec des constantes de temps différentes ; TRIMP = durée × f(HR). |
| **Origine** | Banister / Calvert (1975–1976, 1991). |
| **Mécanisme** | Systèmes dynamiques longitudinaux entre séances. |
| **Implication** | Notre TRIMP *intra-activité* est une **hypothèse nouvelle**, pas une conséquence prouvée de Banister. Formulation : *TRIMP-like cumulative state as predictive proxy*. |
| **Piège** | « Banister showed that TRIMP causes within-race slowing ». |

### Idée D2 — TRIMP individualisé (courbe HR–lactate) bat un coefficient générique

| | |
|---|---|
| **Énoncé** | Pondérer l’intensité par le profil HR–lactate de l’athlète lie mieux charge hebdomadaire et gains de seuil. |
| **Origine** | Manzi et al., *MSSE* 2009. |
| **Mécanisme** | Même %HRR n’a pas le même coût métabolique pour tous. |
| **Implication** | α et κ individuels ; éventuellement TRIMPi futur. |
| **Piège** | Transposer Manzi minute-par-minute sans nouvelles preuves. |

### Idée D3 — Ultra ~65 km : intensité moyenne ~77 % HRmax, surtout sous VT1, TRIMP total énorme

| | |
|---|---|
| **Énoncé** | Un ultra de montagne se court longtemps « facile » en zone, mais avec une charge cumulée très élevée ; VT1 borne l’intensité durable. |
| **Origine** | Fornasiero et al., *JSS* 2018. |
| **Mécanisme** | Durabilité ≠ intensité de seuil court. |
| **Implication** | Enveloppe HRR–durée / blend empirique (H10) : baisser HRR★ sur 8–12 h ; ancrage ~0,65–0,75. |
| **Piège** | Utiliser 77 % HRmax comme cible unique toutes distances. |

### Idée D4 — La fatigue écologique change la cinématique différemment en montée et en descente

| | |
|---|---|
| **Énoncé** | En fin de boucles trail : vitesse ↓, contact ↑, foulée ↓ ; patterns UH ≠ DH. |
| **Origine** | Genitrini et al., *Front. Sports* 2024. |
| **Mécanisme** | Fatigue = changement de mécanisme de production de vitesse, pas seulement baisse d’« effort ». |
| **Implication** | État de fatigue temps-dépendant (TRIMP) + diagnostics par terrain ; modulateurs famille. |
| **Piège** | Traiter Genitrini comme un modèle de temps final. |

---

## E. Déterminants de performance (niveau athlète / course)

### Idée E1 — La triade classique est pertinente mais insuffisante en trail

| | |
|---|---|
| **Énoncé** | VO₂max, fraction durable et économie comptent, mais hétérogénéité des courses et rôle neuromusculaire / pente. |
| **Origine** | de Waal 2021 (revue) ; Ehrström 2018 (27 km). |
| **Implication** | Modèle multicomposant (physique + interne + fatigue), pas régression VO₂max seule. |

### Idée E2 — Les déterminants changent avec la distance

| | |
|---|---|
| **Énoncé** | Capacité aérobie reste centrale ; lipides / autres facteurs deviennent saillants selon la distance (UTMB multi-distances). |
| **Origine** | Pastor et al., *IJSPP* 2022. |
| **Implication** | Ne pas figer un seul HRR★ ou κ pour 3 h et 12 h (d’où H13b court vs H10 ultra). |

### Idée E3 — Historique sur *la même* course est un baseline extrêmement fort

| | |
|---|---|
| **Énoncé** | Ajouter le temps de l’année précédente sur le même parcours fait exploser le R² (labo + historique). |
| **Origine** | Scheer et al., *IJSPP* 2019. |
| **Implication** | Comparer des **tâches** équivalentes ; ne pas comparer notre LOO multi-activités à un modèle « même course N−1 ». |

---

## F. Modèles opérationnels (historique / checkpoints) — autre question scientifique

### Idée F1 — Big data distance–durée donne des indices individuels (~2 % d’erreur route)

| | |
|---|---|
| **Énoncé** | À partir de millions d’activités, on extrait puissance aérobie + indice d’endurance pour prédire 5 km→marathon. |
| **Origine** | Emig & Peltonen, *Nat. Commun.* 2020. |
| **Implication** | Montre la force de l’historique individuel ; **ne traite ni pente ni HR continue** — ne pas le vendre comme « wearable HR → race time ». |

### Idée F2 — TRAP : abandon et temps au prochain checkpoint (UTMB)

| | |
|---|---|
| **Énoncé** | Historique ITRA + terrain + checkpoints → P(atteindre CP) et heure de passage avec incertitude. |
| **Origine** | Fogliato et al., *JQAS* 2021. |
| **Implication** | Baseline *opérationnel* ; rappels d’**intervalles de prédiction**. Tâche ≠ simulation pré-course physiologique. |

### Idée F3 — Après ~1/3 de course, un modèle terrain + pacing explique > 95 % du temps total (un événement)

| | |
|---|---|
| **Énoncé** | Temps pondérés par difficulté, variabilité de pacing, rang au checkpoint → R² ajusté ~0,96. |
| **Origine** | Gutiérrez et al., *Sports* 2025. |
| **Implication** | Puissance de l’**in-race** ; son R² n’est **pas** comparable à une prédiction pré-course. Chez nous : distinguer reconstruction HR / update in-race / scénario HRR prescrit. |

---

## G. Idées méthodologiques transversales (issues du guide FR)

### Idée G1 — Trois couches rarement réunies

Physique du parcours + réponse HRR continue + état de fatigue cumulé, dans un modèle explicite évalué hors échantillon. C’est la lacune centrale revendiquable (*to our knowledge*), avec Jaén-Carrillo 2026 comme intersection déjà non vide.

### Idée G2 — Trois tâches distinctes (ne pas les confondre)

| Tâche | Données de la course cible | Nom recommandé |
|-------|----------------------------|----------------|
| 1 | Profil + **HR observée** | Reconstruction rétrospective informée par HR |
| 2 | HR jusqu’au temps \(t\) | Mise à jour in-race |
| 3 | Profil seul + **stratégie HRR prescrite** | Simulation prospective |

Notre Table 5 / H13b relèvent de la tâche 3. Le LOO M3 avec HR observée relève de la tâche 1.

### Idée G3 — LOO inter-activités ≠ validation inter-athlètes

Généralisation à une nouvelle sortie du **même** athlète. Transfert populationnel = autre hypothèse (multi-athlètes, partial pooling).

### Idée G4 — HRR instantané et TRIMP cumulé partagent le même signal

Ablations statistiques possibles ; séparation physiologique non garantie. Tester contre durée seule, D+, charge mécanique, HR lissée/décalée.

### Idée G5 — MAPE pour comparer des courses de durées très différentes

MAE en minutes surpondère l’ultra. Pour le prospectif adaptatif, **MAPE** comme métrique principale ; biais optimiste (plus rapide que le réel) acceptable pour un enveloppe de course.

### Idée G6 — Enveloppe HRR–durée : power-law vs fenêtres empiriques

| | Power-law \(a\,T^b\) | Fenêtres empiriques monotones |
|--|---------------------|-------------------------------|
| Comportement long | Trop plat (\(b\approx -0{,}04\)) | Baisse nette (~0,62 dès 6–8 h) |
| Usage | Faisabilité « max historique lissé » | Ancre ultra type Fornasiero |
| Recette | H10 blend \(w_{\max}=0{,}70\) | + H13b mid(PL, HRR_ref) si \(T<5\) h |

### Idée G7 — Poussée d’intensité sur courses courtes (inspiration VT2 / CP)

Pour \(T\lesssim 5\) h, viser au-dessus du plancher « faisable » vers \(\mathrm{HRR}_{\mathrm{ref}}\) (effort VMA plat), sans aller jusqu’à tout le temps à 0,88 (qui détruit le MAPE ultra). Recette H13b : \(\mathrm{HRR}^\star=\tfrac12(\mathrm{HRR}_{\mathrm{PL}}+\mathrm{HRR}_{\mathrm{ref}})\).

### Idée G8 — Modulation par type de segment

Distributions empiriques de HRR sur 5 familles → \(m_f\) ; HRR segment = HRR★ · \(m_f\) puis re-centrage. Effet faible seul (~±6 %) ; utile avec une bonne cible moyenne (H10/H13b).

---

## H. Carte « idée → composant du code / papier »

| Idée | Composant Running Manager / papier |
|------|-----------------------------------|
| A1–A2 | Baseline Jaén-Carrillo ; α, κ ; related work |
| A3 | Justification twin « léger » vs labo |
| A4, B1–B3 | GAP Minetti + soft-ramp ; familles terrain |
| C1 | HRR, HRR_ref, loi \(E\) |
| C2–C4 | Limites claims ; QC capteur |
| C5 | Calibration future seuils |
| D1–D2 | TRIMP intra-activité (proxy) |
| D3, G6 | Enveloppe durée ; H10 |
| D4, G8 | Fatigue × terrain ; modulateurs |
| E2, G7 | H13b court vs ultra |
| F1–F3 | Contraste tâches opérationnelles vs twin |
| G2 | Table 5 prospective vs LOO M3 |
| G5 | Journal adaptatif MAPE-primary |

---

## I. Ordre de lecture recommandé (idées d’abord)

1. **A1 + G2** — où se place le papier et quelles tâches mesurer.  
2. **C2** — ce que HRR n’est *pas*.  
3. **B2 + B3** — pourquoi le GAP trail n’est pas Minetti pur.  
4. **D3 + G6 + G7** — comment choisir HRR★ prospectif (court vs ultra).  
5. **A3** — ce qu’on n’essaie pas de reproduire (twin labo lourd).  
6. **F2–F3** — ce qu’on ne doit pas comparer en R².

Fiches article plus compactes : `literature_guide_private_notes_fr.md`.  
Related work anglais prêt à coller : `related_work_paper_ready.md`.  
Expériences HRR adaptatif : `journal_segment_hrr_adaptive.md`.
