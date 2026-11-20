# RunningManager · récupération, ravitos et estimation du temps

20 novembre 2026 · Audit de la PR #5 et résultats expérimentaux sur l’historique d’un athlète

## Décision

**La PR #5 est utile comme base expérimentale. Je la laisse en brouillon.** J’ai reproduit ses résultats archivés et vérifié un correctif local pour trois problèmes de validation. Le correctif change la conclusion du test historique E4. Le code d’E4 continue à construire les parcours depuis les traces réalisées, y compris pour les événements étiquetés « pré-course ». Le passage à un protocole avec GPX réellement disponible avant le départ reste nécessaire pour démontrer la capacité prévisionnelle revendiquée. Le patch et le plan révisé accompagnent ce rapport. [PR #5](https://github.com/piromagnus/RunningManager/pull/5) · [Code d’E4](https://github.com/piromagnus/RunningManager/blob/5cc439ffbdbacfc66c86a733ec4ace942b4647f6/scripts/validate_intensity_conditioned_claim.py#L938)

**Pour le chrono avant départ, j’utiliserais l’oubli par segment comme référence. Pour la mise à jour en course, le signal le plus exploitable est le dernier écart d’allure.** Une correction AR(1) réduit l’erreur sur le prochain segment et sur le temps restant dans nos rejeux. Son gain par rapport au simple report du dernier écart reste incertain à HRR fixe. Les données permettent de repérer des épisodes lents ou immobiles, mais les mesures actuelles ne suffisent pas pour estimer le gain causal d’une marche ou d’un ravito.

## 1. Audit et rejeu de la PR #5

J’ai vérifié la tête `5cc439f` et la base `4c5cc8b`. Le diff comporte 43 commits et 440 fichiers. Les cinq CSV de données du manifeste sont identiques entre les PR #4 et #5. Les 45 sorties de la cohorte `hardTrailRun` ont leurs séries brutes, métriques de séries et fichiers Strava dans la PR #4. Le manifeste n’enregistre aucune empreinte de ces flux. Le rejeu avec ces fichiers retrouve les effectifs archivés de **248 activités segmentables** et **45 sorties `hardTrailRun`**, ainsi que les valeurs publiées de B0 à B3 et E5. Les 54 tests ciblés passent. La suite complète donne **203 réussites et 7 échecs** sur la PR, contre **175 réussites et les mêmes 7 échecs** sur la base, avec les mêmes dépendances. [Données portables](https://github.com/piromagnus/RunningManager/pull/4) · [Résultats archivés](https://github.com/piromagnus/RunningManager/tree/5cc439ffbdbacfc66c86a733ec4ace942b4647f6/data/exp_perf_predictions/trail_digital_twin_claim_validation)

| Vérification | PR rejouée | Après correction locale |
|---|---|---|
| **E2. B3 contre la meilleure référence fixe** | Le code sélectionne B1, la référence au plus grand gain relatif. B3 donne 9,99 % de MAPE, B1 23,01 %. | B0 est la meilleure référence fixe avec 18,65 %. Le gain de B3 reste de 8,66 points, soit 46,4 % en relatif. IC bootstrap 95 % du gain en points [3,89 ; 13,68] sur le rejeu corrigé. |
| **E3. Concordance des paires ayant un écart de HRR > 0,02** | Le dénominateur inclut les 29 paires. Concordance 65,5 %. | 19 paires concordantes sur 22 admissibles, soit 86,4 %. L’appariement distance/dénivelé reste grossier et une activité peut intervenir dans plusieurs paires. |
| **E4. Choix de HRR sur historique, 18 événements** | La normalisation REDI utilise une médiane comprenant les activités futures. Référence fixe 10,18 %, meilleure règle historique 8,47 %. IC du gain en points incluant zéro. | La médiane de normalisation est calculée séparément sur les sorties antérieures à chaque événement. Référence fixe 11,73 %, loi de puissance historique 9,37 %. Gain 2,36 points, IC [0,37 ; 4,08]. Le test passe sur ce rejeu. |
| **E5. Comparaison des représentations de fatigue** | TRIMP cumulé prédit 9,99 %, progression 13,95 %. Les variantes réutilisent des paramètres calibrés pour Stage 3 avec TRIMP décroissant observé. | Même résultat dans le patch d’audit. Les variantes sont réajustées séparément dans le benchmark indépendant de la section suivante. |

B3 reçoit la **HRR moyenne effectivement réalisée**, disponible après la sortie. Son score mesure la qualité de l’estimation conditionnelle quand cette valeur est fournie. En E4, deux événements portent l’étiquette « pré-course », deux « géométrie réalisée » et quatorze « inconnue », mais le code construit les segments depuis la trace réalisée pour les 18. Le patch corrige aussi le libellé du verdict pour refléter le rejeu rétrospectif et rend la graine du bootstrap E4 stable d’un processus à l’autre. Le correctif est inclus dans cette PR de suivi, au-dessus de la PR #5. [Entrées de B3 et E4](https://github.com/piromagnus/RunningManager/blob/5cc439ffbdbacfc66c86a733ec4ace942b4647f6/scripts/validate_intensity_conditioned_claim.py#L562)

## 2. La récupération continue améliore-t-elle la prédiction de chrono ?

J’ai comparé le TRIMP cumulé à un oubli par segment et à une récupération en temps réel. Le modèle continu résout, à charge constante sur le segment, **dF/dt = q − F/τ**, où q est le débit de TRIMP. À l’inférence, la durée qui alimente l’état est celle **prédite par le modèle**. Les variantes reçoivent les mêmes distances, facteurs de pente et d’altitude, et la même HRR. Chacune a ses propres paramètres ajustés sur le train. Le modèle simplifié utilise un ancrage de 18 km/h, un ratio de HRR borné et un multiplicateur de vitesse exponentiel en F. Il omet la disponibilité liée à la charge chronique. Les résultats ci-dessous portent sur la géométrie des sorties réalisées.

**Test principal.** 45 sorties reconstituées, 836 segments, 44 groupes de dates. Les 30 sorties évaluées chronologiquement ont chacune au moins 15 sorties d’apprentissage à une date strictement antérieure. La cible est le temps en mouvement de l’activité. La HRR fixe de 0,70 sert de sensibilité commune à toutes les variantes.

| Fatigue | HRR réalisée fournie · MAPE | HRR réalisée fournie · MAE | HRR fixe 0,70 · MAPE |
|---|---:|---:|---:|
| Aucune | 11,05 % | 22,42 min | 19,38 % |
| TRIMP cumulé | 11,23 % | 24,84 min | 19,19 % |
| **Oubli par segment** | **10,43 %** | 22,44 min | **16,44 %** |
| Oubli en minutes, τ = 30 min | 10,93 % | 24,36 min | 17,91 % |
| Oubli en minutes, τ = 90 min | 11,07 % | 23,69 min | 17,75 % |
| τ choisi sur le train entre 30 et 90 min | 11,07 % | 23,69 min | 17,75 % |

Pour la MAPE avec HRR réalisée fournie, **oubli par segment moins τ = 30 min = −0,49 point**, IC apparié 95 % **[−1,78 ; +0,76]**. L’écart reste du même ordre en ciblant la somme des durées segmentaires et en réajustant tous les modèles, **7,81 % contre 8,40 %**, différence **−0,58 point [−1,56 ; +0,42]**. Cette sensibilité est importante car la somme des temps en mouvement des segments s’écarte d’au moins 5 % de la cible de l’activité pour 13 sorties sur 45. En leave-one-activity-out, l’oubli par segment donne 10,26 % contre 10,88 % pour τ = 30 min. Sur l’ancien export fixe de `perf_dev` avec cible segmentaire, le classement est le même, avec 8,44 % contre 8,98 % dans une comparaison ajustée sortie par sortie. Les intervalles bootstrap rééchantillonnent les différences observées par sortie sans refaire les ajustements. [Export anonymisé de `perf_dev`](https://github.com/piromagnus/RunningManager/blob/4c5cc8be3a10c2bc037f551932e0d07cf2103f7c/docs/science/paper_assets/anonymized_segment_features.csv)

**Conclusion.** L’oubli en minutes est interprétable pour simuler une baisse d’effort, mais **aucune variante temporelle testée n’améliore de façon établie le chrono global**. À HRR fournie, l’oubli par segment a la meilleure MAPE et un résultat de MAE presque identique au modèle sans fatigue. Le choix de la métrique et du protocole compte. Le benchmark à HRR constante compare les dynamiques de fatigue. Il laisse ouverte la question du bénéfice d’une marche volontaire au sein d’un véritable profil d’effort variable.

## 3. Marche, pauses et effet des ravitos

### Propriétés vérifiées sur scénario synthétique

Avec une charge q réduite, F décroît lorsque **F > τq**. Si la marche reste intense, q peut rester assez haut pour que F continue à monter plus lentement. Avec τ fixé arbitrairement à 30 min, une fatigue de départ de 25,94 unités évolue après 5 min à **22,42 en marche facile**, **26,10 en marche soutenue**, **26,56 en poursuivant à la même charge** et **21,96 à l’arrêt**. Les entrées et les unités sont synthétiques. Les assertions vérifient la demi-vie, la composition à charge constante et la décroissance à l’arrêt.

Une pause synthétique de 1, 3 ou 5 min enlève respectivement **0,85, 2,47 ou 3,98 unités** de fatigue. Un même allongement de 2 min a un effet plus petit quand la pause est déjà plus longue. Après un arrêt de 3 min, à effort identique lors de la reprise, l’écart de fatigue entre les scénarios passe de 2,47 à **0,91 après 30 min** de reprise. C’est une propriété de la dynamique choisie. Les données n’ont fourni aucune estimation d’une demi-vie physiologique ni d’un gain de chrono dû à la pause.

Dans la version classique additive, **ARD₁** efface une fraction de la fatigue acquise depuis le dernier événement, et **ARD∞** une fraction de celle qui reste. Pour deux épisodes qui ajoutent 10 puis 4 unités et une réduction de 50 %, le premier événement laisse 5 unités dans les deux cas. Le second laisse **7 en ARD₁** et **4,5 en ARD∞**. Il faut donc plusieurs événements observés pour distinguer ces règles. Le banc teste ces lois séparément de la récupération continue. Une extension devrait réserver le saut à un effet supplémentaire du ravito, à estimer au-delà de l’effet de sa durée. Le bruit brownien n’a pas été ajusté aux traces. [Leroy et al., modèle ARD₁](https://onlinelibrary.wiley.com/doi/full/10.1002/qre.3618) · [Bautista et al., modèle ARD∞](https://doi.org/10.1177/1748006X251348825) · [Revue des modèles de reconstitution de D′](https://doi.org/10.1007/s40279-026-02410-x)

### Signal observable dans l’historique

Les 45 sorties fournissent **510 870 échantillons**. Le pas médian typique est d’une seconde, mais 21 sorties ont au moins un intervalle de plus de 5 s. En rompant les épisodes aux lacunes et en cherchant des blocs de 60 à 600 s, le seuil sous 1 km/h détecte **140 arrêts candidats dans 19 sorties**. Le seuil sous 4 km/h isole **489 épisodes lents dans 36 sorties**, dont certains peuvent être de la course lente ou de la marche fortement chargée. Les liens entre activité et plan de course couvrent deux des 45 sorties. Les prises alimentaires, l’intention de marcher et les instants exacts de ravitaillement ne sont pas annotés dans ces flux.

Avec deux fenêtres de 5 min, des critères exploratoires de couverture, de distance, de pente nette et de FC ne laissent que **8 arrêts candidats comparables dans 6 sorties**. La vitesse brute change de **−5,3 %** en médiane après l’arrêt, quartiles **−16,0 % et +5,1 %**. Le test ne comporte pas de fenêtres témoins appariées et ne contrôle pas la technicité, le changement de stratégie ou la fatigue préalable. Ce résultat ne permet d’estimer ni l’effet d’une pause, ni sa persistance, ni sa variation en fin de course. Le signal ne permet pas non plus de départager ARD₁ et ARD∞.

## 4. ETA en course à partir des écarts déjà observés

Le rejeu chronologique part du modèle à TRIMP cumulé. Les paramètres de correction utilisent uniquement des sorties antérieures, avec un minimum de cinq sorties ayant déjà des prédictions chronologiques et vingt paires adjacentes. Une fois en course, le résidu du dernier segment terminé est disponible. Le résultat couvre **25 sorties et 462 paires de segments** par hypothèse de HRR. Les MAE ci-dessous donnent le même poids à chaque sortie.

| Correction pour le segment suivant | HRR fixe, MAE | HRR réalisée fournie, MAE |
|---|---:|---:|
| Statique | 133,71 s | 94,03 s |
| Report du dernier résidu | 77,28 s | 76,24 s |
| Moyenne des résidus déjà observés dans la sortie | 90,72 s | 80,28 s |
| **AR(1)** | **75,95 s** | **70,04 s** |
| Ornstein–Uhlenbeck à pas temporel variable | 78,56 s | 71,25 s |

À HRR fixe, l’AR(1) améliore 21 sorties sur 25 face au modèle statique. Le gain moyen est **57,76 s**, IC à 95 % **[33,88 ; 84,98]**. Face au simple report du dernier résidu, le gain moyen est **1,33 s**, et l’intervalle apparié va de **−2,44 à +5,05 s**. Un comparateur aussi simple capte donc déjà l’essentiel du signal dans ce scénario. La conclusion vis-à-vis du statique tient avec un modèle entraîné sur la somme des temps des segments. Dans le diagnostic préliminaire sur l’ancien export, la correction AR(1) rétrécit aussi l’intervalle prédictif du prochain segment avec une couverture proche de la cible. Les prédictions de base de cet ancien export sont ajustées sur les données et utilisent la HRR observée.

J’ai ensuite propagé l’AR(1) jusqu’à la fin **sans lire les segments futurs** dans 24 sorties où les indices des segments et leurs temps en mouvement sont utilisables. À HRR fixe, aux checkpoints de 25 %, 50 % et 75 % des segments, la MAE en temps restant passe respectivement de **28,0 à 22,5 min**, **19,1 à 13,3 min** et **10,6 à 5,8 min** face au statique. À mi-parcours, le gain apparié est de **5,8 min**, IC 95 % **[1,8 ; 11,7]**. Face au simple report du dernier résidu, l’incertitude inclut zéro à ces trois checkpoints. Ce calcul concerne le **temps en mouvement restant sur la trace réalisée**. Il ne valide pas la couverture d’un intervalle avant départ ou les temps d’arrêt à venir.

## 5. Plan mis à jour

1. **Avant départ.** Garder l’oubli par segment comme référence. Traiter la règle historique E4 comme candidate à une confirmation sur événements réservés, avec le vrai GPX pré-course, les références de normalisation calculées au jour du départ et une HRR choisie uniquement à partir des données disponibles à ce moment-là.
2. **Pendant la course.** Démarrer l’évaluation opérationnelle avec le dernier résidu et l’AR(1). Comparer l’erreur sur le prochain segment et l’ETA finale, en distinguant temps en mouvement et chrono écoulé. Utiliser de nouvelles sorties pour décider si l’ajustement de l’AR(1) justifie sa complexité. Calibrer les intervalles pré-course dans une expérience dédiée.
3. **Marche et ravitos.** Instrumenter plusieurs sorties comparables avec blocs de marche volontaire, poursuite et arrêt de durées fixées à l’avance, effort après reprise et apports horodatés. Mesurer les 5 premières minutes puis les fenêtres 5–15 et 15–30 min. Comparer le temps regagné au temps cédé. Varier l’ordre des conditions et répéter tôt et tard pour distinguer persistance, durée de pause et fatigue initiale.
4. **ARD.** Tester la récupération continue seule puis l’ajout d’un saut ARD₁ ou ARD∞ après acquisition de plusieurs ravitos annotés par sortie. Ne retenir un terme additionnel que s’il améliore la prédiction hors sortie et si ses paramètres sont estimables. Les séries actuelles ne permettent pas de régler une dose de ravito ou de prédire un bénéfice physiologique individuel.

Les cinq missions techniques ont été confiées à des sous-agents Terra en effort medium avec des objectifs et règles de lecture, d’inférence et de restitution définis. J’ai vérifié les rejeux, produit le correctif local, réalisé le test du temps restant et pris les décisions du plan.
