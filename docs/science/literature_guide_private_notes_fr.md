# Guide critique de la littérature — Digital twin, fréquence cardiaque et performance en trail

Ce document est un support de compréhension et de décision pour les auteurs. Il n'est pas destiné à être inséré tel quel dans l'article. Il synthétise la [bibliographie GitHub fournie](https://github.com/piromagnus/RunningManager/blob/cursor%2Fsegment-outlier-optimization-7aaf/docs%2Fscience%2Fbibliography_hr_digital_twin.md), la met en regard du [brouillon v0.5](https://github.com/piromagnus/RunningManager/blob/cursor%2Fsegment-outlier-optimization-7aaf/docs%2Fscience%2Ftrail_digital_twin_hr_performance_paper_draft.md), ajoute les travaux les plus proches trouvés dans la littérature, puis explicite les lacunes et leurs conséquences méthodologiques. Recherche mise à jour au 21 juillet 2026.

**Complément :** développement idée-par-idée des travaux récents → [`idees_travaux_recents_detaillees_fr.md`](idees_travaux_recents_detaillees_fr.md).

## 1. Carte rapide du champ

| Famille | Question principale | Forces | Limites par rapport au manuscrit |
|---|---|---|---|
| Déterminants physiologiques | Quels athlètes sont les plus rapides ? | Mesures physiologiques interprétables | Sortie globale, petits échantillons, peu de dynamique segmentaire |
| Physique du parcours | Quel coût impose la pente, l'altitude ou le terrain ? | Mécanisme explicite et transférable au profil de route | Ne connaît pas directement l'état interne ou la fatigue du coureur |
| Digital twins physiologiques | Quelle puissance un individu peut-il produire dans le temps ? | Individualisation et interprétabilité | Tests de laboratoire lourds; souvent efforts courts ou cyclisme |
| HR, HRR et TRIMP | Quelle réponse interne et quelle charge sont associées à l'effort ? | Peu coûteux, continu, individualisable | Retard, dérive cardiaque, dépendance au terrain et au capteur |
| Historique et checkpoints | Que va-t-il se passer ensuite dans la course ? | Très bonne précision opérationnelle possible | Utilise l'historique ou une partie de la course cible; mécanisme physiologique faible |
| Biomécanique et fatigue | Comment la technique se dégrade-t-elle ? | Montre l'asymétrie montée/descente et la fatigue réelle | Rarement relié à une prédiction externe du temps final |

La place exacte du projet est à l'intersection de trois familles : physique du parcours, réponse HRR continue et état cumulatif de fatigue. Cette intersection est peu testée, mais elle n'est pas vide : Jaén-Carrillo et Pattis (2026) constituent le précédent direct.

## 2. Fiches de lecture des articles centraux

### 2.1 Synthèses et déterminants de la performance

#### de Waal et al. (2021) — *Physiological Indicators of Trail Running Performance: A Systematic Review*

- **Question.** Quels indicateurs physiologiques sont effectivement associés à la performance en trail ?
- **Données et méthode.** Revue systématique des études reliant des mesures de laboratoire ou de terrain au résultat d'une course de trail.
- **Résultat principal.** VO2max, vitesse aérobie, seuils métaboliques, économie de course et composition corporelle apparaissent régulièrement, mais les résultats ne sont pas uniformes. Les courses, protocoles et populations sont trop hétérogènes pour produire un modèle universel simple.
- **Ce que cela apporte ici.** Cette revue justifie un modèle multicomposant et spécifique au terrain. Elle soutient aussi l'idée que la triade classique de l'endurance ne suffit pas.
- **Limite à retenir.** La littérature synthétisée prédit surtout une performance globale entre individus; elle ne modélise pas une trajectoire d'état seconde par seconde.
- **Source.** [doi:10.1123/ijspp.2020-0812](https://doi.org/10.1123/ijspp.2020-0812).

#### Ehrström et al. (2018) — *Short Trail Running Race: Beyond the Classic Model for Endurance Running Performance*

- **Question.** VO2max, fraction de VO2max au seuil et économie de course suffisent-ils pour une course de trail courte ?
- **Données et méthode.** Neuf traileurs très entraînés; course officielle de 27 km; VO2max, seuil ventilatoire, économie à 0% et +10%, force des extenseurs du genou, endurance locale et test d'épuisement.
- **Résultat principal.** L'indice de fatigue musculaire locale et VO2max sont les variables les plus corrélées au temps de course; les autres variables classiques sont moins informatives dans ce petit échantillon. Les auteurs recommandent des tests spécifiques à la pente et à l'endurance locale.
- **Ce que cela apporte ici.** Il existe une composante neuromusculaire et spécifique à la pente que HR seul ne représente pas nécessairement. Cela soutient les analyses séparées montée/descente.
- **Limite à retenir.** Échantillon de neuf et modèle explicatif ajusté sur la même course : il ne démontre ni généralisation externe ni prédiction segmentaire.
- **Source.** [doi:10.1249/MSS.0000000000001467](https://doi.org/10.1249/MSS.0000000000001467).

#### Alvero-Cruz et al. (2019) — *Prediction of Performance in a Short Trail Running Race: The Role of Body Composition*

- **Question.** Quelle combinaison simple de physiologie et de composition corporelle explique le temps d'une course courte ?
- **Données et méthode.** Vingt-cinq coureurs; tests physiologiques et anthropométriques avant une course courte; régression du temps final.
- **Résultat principal.** Un modèle combinant VO2max et masse grasse explique une grande partie de la variance du temps de course (R² rapporté de 0,839; erreur standard d'environ 11 min).
- **Ce que cela apporte ici.** Bon comparateur pour les modèles « profil de l'athlète → temps final ».
- **Limite à retenir.** Modèle transversal, ajusté sur une seule course et sans trajectoire de parcours, d'effort ou de fatigue. Son R² n'est donc pas directement comparable à un leave-one-activity-out.
- **Source.** [doi:10.3389/fphys.2019.01306](https://doi.org/10.3389/fphys.2019.01306).

#### Scheer et al. (2019) — prédiction par tests de laboratoire et résultat de terrain

- **Question.** Des tests de laboratoire peuvent-ils prédire un trail de 31,1 km ?
- **Données et méthode.** Petit échantillon de coureurs ayant réalisé des tests graduels; prédiction du temps de course; ajout du temps réalisé sur la même course l'année précédente.
- **Résultat principal.** Les indicateurs de laboratoire, notamment la vitesse à VO2max, expliquent une part de la performance. L'ajout du temps de la même course l'année précédente augmente fortement le R², jusqu'à environ 0,99.
- **Ce que cela apporte ici.** Un historique très spécifique au coureur et au parcours peut être un baseline extrêmement fort.
- **Limite à retenir.** Le temps de l'année précédente est proche de la cible conceptuelle; le très petit échantillon et l'ajustement interne peuvent produire un R² optimiste. Il faut comparer des tâches de prédiction équivalentes.
- **Source.** [doi:10.1123/ijspp.2018-0390](https://doi.org/10.1123/ijspp.2018-0390).

#### Pastor et al. (2022) — *Performance Determinants in Trail-Running Races of Different Distances*

- **Question.** Les déterminants changent-ils avec la distance du trail ?
- **Données et méthode.** Soixante-quinze participants aux événements UTMB répartis sur plusieurs distances; batterie de mesures physiologiques, anthropométriques et neuromusculaires.
- **Résultat principal.** La capacité aérobie est le déterminant le plus constant, mais l'utilisation des lipides et d'autres caractéristiques deviennent pertinentes selon la distance.
- **Ce que cela apporte ici.** La relation effort–performance n'est probablement pas stationnaire entre une course courte et une course ultra; un même paramètre de fatigue ne doit pas être supposé universel sans test.
- **Limite à retenir.** L'étude explique les différences entre coureurs et distances, pas la dynamique interne d'une activité.
- **Source.** [doi:10.1123/ijspp.2021-0362](https://doi.org/10.1123/ijspp.2021-0362).

#### Fornasiero et al. (2018) — profil d'intensité d'un ultra de 65 km

- **Question.** À quelle intensité physiologique se court un ultra-trail de 65 km et quelle charge représente-t-il ?
- **Données et méthode.** Vingt-trois traileurs amateurs; test incrémental en montée; HR enregistrée pendant 65 km et +4000 m; zones définies par seuils ventilatoires; TRIMP total.
- **Résultat principal.** L'intensité moyenne est d'environ 77% de HRmax; environ 86% du temps est passé sous le premier seuil, pour un TRIMP total très élevé. Le premier seuil ventilatoire semble délimiter l'intensité durable.
- **Ce que cela apporte ici.** C'est une justification trail-spécifique pour normaliser HR par des repères individuels et pour mesurer une charge cumulée.
- **Limite à retenir.** Le TRIMP est un résumé global de la course. L'étude ne valide pas l'usage de son cumul instantané comme multiplicateur causal de vitesse.
- **Source.** [doi:10.1080/02640414.2017.1374707](https://doi.org/10.1080/02640414.2017.1374707).

### 2.2 Physique du parcours et digital twins

#### Minetti et al. (2002) — coût énergétique des pentes extrêmes

- **Question.** Comment varie le coût métabolique de la marche et de la course avec la pente ?
- **Données et méthode.** Dix coureurs; tapis roulant sur des pentes de −45% à +45%; consommation d'oxygène et coût de locomotion.
- **Résultat principal.** Relation fortement non linéaire et asymétrique entre pente et coût, devenue la base de nombreux équivalents de vitesse ou de distance.
- **Ce que cela apporte ici.** Fondation scientifique de la correction de pente du modèle physique.
- **Limite à retenir.** Surface lisse, vitesse et contexte de laboratoire. La technicité, les changements de direction, le risque et les impacts excentriques d'un vrai trail ne sont pas capturés.
- **Source.** [doi:10.1152/japplphysiol.01177.2001](https://doi.org/10.1152/japplphysiol.01177.2001).

#### Nicot et al. (2022) — effet de la technicité du sol

- **Question.** Une montée réelle coûte-t-elle plus cher qu'une montée équivalente sur tapis ?
- **Données et méthode.** Dix traileurs masculins expérimentés; deux trails montants d'environ 10,5 min à allure de course, reproduits sur tapis à pente, vitesse et distance identiques.
- **Résultat principal.** Le coût en oxygène est supérieur d'environ 10,5% sur trail; la ventilation augmente d'environ 21%; l'amplitude et la variabilité médio-latérales du pied augmentent fortement. Les deux niveaux de technicité définis a priori sont néanmoins difficiles à distinguer.
- **Ce que cela apporte ici.** Preuve directe que pente et distance ne suffisent pas. Un résidu attribué à HR peut en réalité provenir de technicité non mesurée.
- **Limite à retenir.** Petit échantillon masculin, montées courtes, et absence de descente ou de fatigue longue.
- **Source.** [doi:10.1080/17461391.2021.1995507](https://doi.org/10.1080/17461391.2021.1995507).

#### Holler et Jaén-Carrillo (2025) — puissance mécanique paramétrique en trail

- **Question.** Peut-on estimer de façon analytique la puissance mécanique nécessaire sur terrain alpin à partir de capteurs grand public ?
- **Données et méthode.** Extension d'un modèle mécanique avec pertes de collision, stockage élastique, force-rate, mouvements du haut du corps, pente et terrain irrégulier; comparaison aux coûts métaboliques publiés et démonstration sur une sortie.
- **Résultat principal.** Bonne cohérence avec le coût métabolique mesuré sur plat et en pente; cadre transparent et adaptable.
- **Ce que cela apporte ici.** Baseline route-only plus riche qu'un simple GAP. Il aide à séparer demande mécanique du parcours et réponse interne.
- **Limite à retenir.** Validation de démonstration sur trail, pas validation multi-course du temps final; pas d'état de fatigue physiologique long.
- **Source.** [doi:10.1016/j.jbiomech.2025.112892](https://doi.org/10.1016/j.jbiomech.2025.112892).

#### Boillet et al. (2024) — digital twin physiologique Margaria–Morton

- **Question.** Peut-on donner un sens physiologique aux paramètres d'un modèle hydraulique de production de puissance et créer le jumeau d'un athlète ?
- **Données et méthode.** Quatre cyclistes féminines de niveau national; VO2max, seuil, puissance critique, 3-min all-out, efficacité et masse musculaire; modèle à trois compartiments métaboliques.
- **Résultat principal.** Les simulations de performances, d'oxygénation et de métabolites sont cohérentes avec des observations de terrain et de laboratoire pour des efforts proches du maximal d'environ 1 à 15–20 min.
- **Ce que cela apporte ici.** Exemple fort de digital twin interprétable et individualisé.
- **Limite à retenir.** Cyclisme, durée courte, plusieurs tests de laboratoire et paramètres partiellement issus de la littérature. Ce n'est pas un modèle prêt à être transposé à plusieurs heures de trail.
- **Correction.** Le DOI correct est **10.1038/s41598-024-56042-0**; celui du fichier GitHub actuel renvoie à un autre article.
- **Source.** [article Scientific Reports](https://www.nature.com/articles/s41598-024-56042-0).

#### Jaén-Carrillo et Pattis (2026) — digital twin physique du trail

- **Question.** Un modèle physique individualisé peut-il prédire le temps de plusieurs courses de trail d'un même athlète ?
- **Données et méthode.** Un homme très entraîné, 13 courses; coût de pente/GAP, altitude, charge de type TRIMP et décroissance de pacing; calibration de paramètres individuels; leave-one-race-out.
- **Résultat principal.** MAE 18,2 min, MAPE 11,1%, R² 0,864 et biais moyen +2,0 min au niveau course.
- **Ce que cela apporte ici.** C'est le comparateur direct. Il établit déjà un digital twin trail individualisé avec physique, altitude, charge et pacing.
- **Limite à retenir.** Un seul athlète et seulement 13 courses; validation inter-course mais pas inter-athlète; représentation de l'état interne moins directement fondée sur la trajectoire HRR observée.
- **Conséquence pour les claims.** Le nouveau papier doit être présenté comme une extension/ablation HRR+charge de ce cadre, pas comme le premier digital twin trail.
- **Source.** [doi:10.3390/s26123731](https://doi.org/10.3390/s26123731).

### 2.3 Fréquence cardiaque, HRR, charge et fatigue

#### Swain et Leutholtz (1997) — HRR et réserve de VO2

- **Question.** À quelle grandeur physiologique le pourcentage de réserve cardiaque correspond-il ?
- **Données et méthode.** Soixante-trois adultes; test maximal incrémental sur cyclo-ergomètre; régressions individuelles entre HRR, VO2max et réserve de VO2.
- **Résultat principal.** %HRR est proche de %réserve de VO2, et non de %VO2max. La pente moyenne HRR–VO2R est proche de 1 avec intercept proche de 0.
- **Ce que cela apporte ici.** Justification solide de la normalisation HRR plutôt qu'un simple %HRmax.
- **Limite à retenir.** Situation contrôlée et incrémentale; aucune preuve que HRR est une mesure instantanée sans retard sur un terrain alternant fortement montée et descente.
- **Source.** [doi:10.1097/00005768-199703000-00018](https://doi.org/10.1097/00005768-199703000-00018).

#### Banister/Calvert — modèle système et TRIMP

- **Question.** Comment relier la dose d'entraînement à deux réponses antagonistes, fitness et fatigue ?
- **Données et méthode.** Modèle de systèmes dynamiques : une impulsion d'entraînement alimente des réponses de fitness et de fatigue avec constantes de temps distinctes; TRIMP résume durée et intensité cardiaque.
- **Résultat principal.** Cadre fondateur de la relation charge–performance sur plusieurs jours ou semaines.
- **Ce que cela apporte ici.** Inspiration mathématique pour un état accumulé et décroissant.
- **Limite à retenir.** La signification originale est longitudinale entre séances. Utiliser le cumul de TRIMP à l'intérieur d'une course comme fatigue instantanée est une nouvelle hypothèse, pas une conséquence établie du modèle de Banister.
- **Source indicative.** Calvert et al., *A systems model of the effects of training on physical performance*, IEEE Transactions on Systems, Man, and Cybernetics, 1976.

#### Manzi et al. (2009) — TRIMP individualisé

- **Question.** Une pondération individuelle de la relation HR–lactate prédit-elle mieux l'adaptation à l'entraînement ?
- **Données et méthode.** Huit coureurs récréatifs de fond; huit semaines; TRIMPi construit à partir du profil HR–lactate individuel.
- **Résultat principal.** Le TRIMPi hebdomadaire est fortement associé aux améliorations de vitesse aux seuils lactate; il est plus individualisé qu'un coefficient moyen.
- **Ce que cela apporte ici.** Montre pourquoi une fonction d'intensité générique peut être moins valable qu'une calibration propre à l'athlète.
- **Limite à retenir.** Dose hebdomadaire et adaptation sur huit semaines, pas ralentissement minute par minute dans une course. Le petit n limite aussi la généralisation.
- **Source.** [doi:10.1249/MSS.0b013e3181a6a959](https://doi.org/10.1249/MSS.0b013e3181a6a959).

#### Born et al. (2017) — HR versus oxygénation musculaire en terrain vallonné

- **Question.** HR suit-elle correctement les changements d'intensité imposés par les montées et descentes ?
- **Données et méthode.** Dix-sept coureurs compétitifs; contre-la-montre trail de 7 km et +486 m; VO2 portable, HR, GPS et NIRS.
- **Résultat principal.** HR atteint environ 94% de HRmax en montée et 91% en descente, alors que VO2 est respectivement d'environ 84% et 68% de VO2max. HR change peu avec la pente; l'oxygénation musculaire suit mieux les variations de VO2.
- **Ce que cela apporte ici.** Avertissement central : HRR ne doit pas être nommée « effort instantané vrai ». C'est une réponse interne lissée et retardée.
- **Limite à retenir.** Course courte et très intense; NIRS d'un groupe musculaire n'est pas non plus un gold standard global.
- **Source.** [doi:10.1123/ijspp.2016-0101](https://doi.org/10.1123/ijspp.2016-0101).

#### Lemire et al. (2021) — déterminants montée versus descente

- **Question.** Les performances en montée et en descente reposent-elles sur les mêmes déterminants ?
- **Données et méthode.** Dix traileurs masculins compétitifs; contre-la-montre de 5 km en montée et en descente; réponses cardio-respiratoires, vitesse aérobie, raideur et force.
- **Résultat principal.** HR moyenne est très proche entre conditions, mais les demandes et prédicteurs diffèrent. Les déterminants de montée sont davantage cardio-métaboliques; ceux de descente comportent davantage de dimensions mécaniques et neuromusculaires.
- **Ce que cela apporte ici.** Justifie des paramètres ou au moins des diagnostics asymétriques montée/descente. Une seule courbe HRR→vitesse risque de confondre deux régimes.
- **Limite à retenir.** Dix hommes, efforts courts et deux courses séparées plutôt qu'un long trail mixte.
- **Source.** [doi:10.1016/j.jsams.2020.06.004](https://doi.org/10.1016/j.jsams.2020.06.004).

#### Feist et al. (2026) — seuil de laboratoire prédit depuis un test trail

- **Question.** Peut-on retrouver la HR au seuil anaérobie sans analyse respiratoire portable ?
- **Données et méthode.** Seize coureurs entraînés, dont sept femmes; protocole extérieur 3 × 1,2 km avec intensité croissante; test lactate sur tapis; secteurs GNSS–IMU et ceinture Polar; régression avec LOOCV.
- **Résultat principal.** Corrélations de 0,89 à 0,94 entre certains secteurs du second tour et la HR du seuil anaérobie; RMSE LOOCV 4,343 bpm et CCC 0,940. Le seuil aérobie est moins bien identifié.
- **Ce que cela apporte ici.** Montre qu'une HR de terrain peut être calibrée vers un repère physiologique individuel avec un protocole standardisé.
- **Limite à retenir.** Il s'agit d'un **résumé de conférence dans un book of abstracts**, pas encore d'une validation complète publiée. Le protocole est standardisé et court, différent d'une course libre longue.
- **Source.** [doi:10.36950/2026.2ciss025](https://doi.org/10.36950/2026.2ciss025).

#### Wehrlin et Hallén (2006) — altitude, VO2max et HRmax

- **Question.** Quelle est la perte de capacité aérobie et de performance entre 300 et 2800 m ?
- **Données et méthode.** Huit athlètes d'endurance en chambre hypobare; VO2max et temps jusqu'à épuisement à plusieurs altitudes.
- **Résultat principal.** VO2max diminue en moyenne d'environ 6,3% par 1000 m et le temps jusqu'à épuisement se dégrade fortement; HRmax diminue plus modestement, d'environ 1,9 bpm par 1000 m, avec variabilité individuelle.
- **Ce que cela apporte ici.** Altitude agit à la fois sur capacité et sur la calibration de HR. Une correction purement mécanique/oxygène sans adaptation de HRmax peut créer un double comptage ou un biais.
- **Limite à retenir.** Le résultat de performance est un temps jusqu'à épuisement à vitesse fixée, pas un temps de course trail.
- **Source.** [doi:10.1007/s00421-005-0081-9](https://doi.org/10.1007/s00421-005-0081-9).

#### Wingo et al. (2005) — dérive cardiovasculaire en chaleur

- **Question.** La dérive de HR lors d'un effort prolongé en chaleur correspond-elle à une modification de l'intensité relative ?
- **Données et méthode.** Exercice sous-maximal prolongé en chaleur suivi d'une mesure de VO2max; analyse de la dérive cardiovasculaire.
- **Résultat principal.** HR augmente tandis que le volume d'éjection diminue; VO2max disponible diminue, ce qui augmente l'intensité relative même lorsque la charge externe reste constante.
- **Ce que cela apporte ici.** La dérive HR n'est ni un bruit pur ni une mesure univoque de fatigue mécanique. Température, hydratation et perte de capacité disponible sont mêlées.
- **Limite à retenir.** Protocole contrôlé en chaleur et durée modérée, pas trail de montagne; le signe et l'interprétation de la dérive peuvent changer selon les conditions.
- **Source.** [PubMed 15692320](https://pubmed.ncbi.nlm.nih.gov/15692320/).

#### Navalta et al. (2020) — validité des capteurs HR en trail

- **Question.** Les wearables mesurent-ils correctement HR pendant un trail à intensité variable ?
- **Données et méthode.** Vingt-et-un participants; 3,22 km montée/descente; plusieurs dispositifs PPG comparés à une ceinture Polar H7; MAPE, limites d'accord et concordance.
- **Résultat principal.** Tous les dispositifs PPG ont un accord insuffisant; par exemple Garmin Fenix 5 MAPE ≈13%. La ceinture ECG connectée à une montre atteint environ 2% de MAPE et une forte concordance.
- **Ce que cela apporte ici.** Le type de capteur doit être un critère d'inclusion ou une covariable de qualité. Les résultats HRR ne sont défendables qu'avec provenance et nettoyage explicités.
- **Limite à retenir.** Modèles de 2020, trail de seulement ~22 min; les dispositifs récents peuvent différer, mais l'artefact de mouvement reste une menace.
- **Source.** [doi:10.1371/journal.pone.0238569](https://doi.org/10.1371/journal.pone.0238569).

### 2.4 Modèles statistiques, historiques et checkpoints

#### Emig et Peltonen (2020) — performance issue de big data réel

- **Question.** Peut-on extraire des indices individualisés de puissance aérobie et d'endurance depuis les activités de wearables ?
- **Données et méthode.** Environ 14 000 coureurs et 1,6 million de séances; modèle universel à deux indices, ajusté sur les meilleures performances 5 km, 10 km, semi et marathon.
- **Résultat principal.** Erreur moyenne d'environ 2% lorsque trois courses ou plus sont disponibles; prédiction du marathon à partir de distances plus courtes généralement meilleure que 10%; relations entre indices et entraînement.
- **Ce que cela apporte ici.** Démonstration qu'un historique individuel massif peut remplacer une partie des tests de laboratoire.
- **Limite à retenir.** Les données centrales sont durée et distance globales; le modèle ne traite ni pente, ni altitude, ni technicité, ni HR continue. Il ne faut donc pas le décrire comme une référence « wearable HR → race time ».
- **Source.** [doi:10.1038/s41467-020-18737-6](https://doi.org/10.1038/s41467-020-18737-6).

#### Fogliato, Oliveira et Yurko (2021) — TRAP

- **Question.** Peut-on prédire abandon et temps au prochain checkpoint avant et pendant l'UTMB ?
- **Données et méthode.** Historique ITRA 2010–2018, caractéristiques du coureur, informations de terrain et checkpoints; modèles statistiques avec intervalles de prédiction.
- **Résultat principal.** Le cadre prédit probabilité d'atteindre le prochain checkpoint, heure de passage et incertitude, avec une vocation opérationnelle pour l'organisation et le suivi.
- **Ce que cela apporte ici.** Baseline conceptuel fort pour une prédiction « connaissant l'historique et la course déjà parcourue »; il rappelle l'importance des intervalles de prédiction.
- **Limite à retenir.** Dépendance à la base ITRA, aux historiques et parfois aux checkpoints de la course cible; pas d'explication physiologique ni de simulation pré-course pure.
- **Source.** [doi:10.1515/jqas-2020-0013](https://doi.org/10.1515/jqas-2020-0013). [Données et description](https://github.com/ricfog/TRAP-data).

#### Gutiérrez et al. (2025) — prédiction en temps réel après le premier tiers

- **Question.** Peut-on estimer le temps total à partir des premiers secteurs et de leur difficulté ?
- **Données et méthode.** 947 résultats, non nécessairement 947 individus uniques, sur deux distances et trois éditions du Trail Valle de Tena; temps pondéré par difficulté, variabilité du pacing et percentile au checkpoint.
- **Résultat principal.** R² ajusté de 0,959 à 0,967 selon le secteur utilisé; prédiction après environ le premier tiers de course.
- **Ce que cela apporte ici.** Montre la puissance d'un baseline in-race simple et spécifique au terrain.
- **Limite à retenir.** Une partie importante de la cible est déjà observée; le percentile utilise les autres concurrents; une seule manifestation; possibles participations répétées; pas de validation externe sur un autre événement. Son R² ne doit pas être comparé à une prédiction pré-course.
- **Source.** [doi:10.3390/sports13110385](https://doi.org/10.3390/sports13110385).

### 2.5 Fatigue et biomécanique écologique

#### Genitrini et al. (2024) — changements entre début et fin de course

- **Question.** Comment les paramètres spatio-temporels et la cinématique changent-ils avec la fatigue en montée et en descente ?
- **Données et méthode.** Vingt participants recrutés, 9,1 km en sept boucles avec GPS et combinaison IMU; 13 conservés après exclusions; comparaison tours initiaux et finaux.
- **Résultat principal.** Vitesse plus faible en fin; contact au sol et duty factor plus élevés; longueur de foulée et temps de vol plus faibles; changements cinématiques différents en montée et en descente.
- **Ce que cela apporte ici.** Preuve écologique que la fatigue est terrain-dépendante et modifie le mécanisme de vitesse, pas seulement la volonté d'effort.
- **Limite à retenir.** Course courte, terrain relativement sûr et répétitif; perte importante de sujets à l'analyse; pas de modèle de temps final.
- **Source.** [doi:10.3389/fspor.2024.1406824](https://doi.org/10.3389/fspor.2024.1406824).

## 3. Lacunes de la littérature et opportunité scientifique

### Lacune 1 — Les trois couches sont rarement réunies

La littérature dispose séparément de bonnes descriptions de la demande du parcours, de la réponse cardiaque et de la fatigue/du pacing. Très peu de travaux les réunissent dans un modèle explicite, continu et individualisé évalué sur des activités non utilisées pour l'ajustement. C'est la lacune centrale que le manuscrit peut revendiquer, sous la formule prudente « to our knowledge ».

### Lacune 2 — HR est informative, mais son construit physiologique est ambigu

HRR normalise utilement les différences interindividuelles, mais HR réagit avec retard et peut rester élevée en descente quand VO2 chute. Elle est également influencée par chaleur, hydratation, altitude, stress, sommeil, caféine et dérive cardiovasculaire. Une amélioration prédictive montre que HR contient de l'information; elle ne prouve pas que le modèle estime mieux « l'effort métabolique instantané ».

**Formulation recommandée :** *continuous individualized internal-response signal* plutôt que *ground-truth instantaneous effort*.

### Lacune 3 — Le cumul HRR et le TRIMP partagent le même signal

Le terme instantané \(E(HRR_t)\) et le terme cumulatif \(F(\sum TRIMP_t)\) dérivent tous deux de HR. Leur contribution peut être statistiquement identifiée dans une ablation, mais pas nécessairement physiologiquement séparée. Le terme cumulatif peut absorber dérive cardiaque, durée, chaleur ou technicité manquante.

**Tests nécessaires :** comparer TRIMP cumulé à (i) durée seule, (ii) distance/élévation cumulée, (iii) charge mécanique estimée, (iv) TRIMP construit avec HR décalée ou lissée; rapporter corrélations, profils de vraisemblance et stabilité des paramètres.

### Lacune 4 — Reconstruction avec HR cible et prédiction prospective sont deux tâches différentes

Lorsque la trajectoire HR de l'activité test est fournie, le modèle connaît une réponse produite pendant la cible. C'est une **reconstruction rétrospective informée par HR**, même si les paramètres ont été entraînés sans cette activité. Une vraie prédiction pré-course ne connaît pas cette trajectoire et doit imposer une stratégie HRR, prédire HR ou simuler plusieurs scénarios.

**Conséquence.** Rapporter séparément :

1. reconstruction route + HR observée;
2. prévision in-race à partir de HR observée jusqu'au temps \(t\);
3. simulation pré-course sous stratégie HRR prescrite;
4. éventuelle prédiction autonome de HR.

### Lacune 5 — La validation inter-activité n'est pas une validation inter-athlète

Le leave-one-activity-out actuel mesure la généralisation vers une nouvelle activité du même athlète. Il ne démontre pas que la forme fonctionnelle ou les paramètres se transfèrent à d'autres coureurs. Avec 3–5 athlètes, le futur travail sera une réplication mécanistique multi-cas, pas une validation populationnelle définitive.

**Plan recommandé :** modèles par athlète + estimations groupées par partial pooling; validation leave-one-activity-out à l'intérieur de chaque athlète; résultats individuels et médiane inter-athlètes; leave-one-athlete-out seulement comme analyse exploratoire de transférabilité.

### Lacune 6 — Montée et descente ne sont pas symétriques

Les études de Born, Lemire, Nicot et Genitrini montrent que HR, VO2, coût mécanique et fatigue neuromusculaire changent différemment selon la pente. Un facteur unique de grade ou une seule relation HRR–vitesse peut laisser un biais systématique en descente technique.

**Tests nécessaires :** erreurs par classes de pente et technicité; interaction HRR × pente; paramètres ou résidus séparés montée/plat/descente; validation sur segments jamais vus.

### Lacune 7 — Les états environnementaux sont mal observés

Altitude, température, humidité, exposition solaire, hydratation et acclimatation changent la capacité disponible et HR. Ils sont rarement intégrés ensemble. Une correction d'altitude basée sur VO2max et une HRR calculée avec HRmax au niveau de la mer peuvent être incohérentes.

**Tests nécessaires :** inclure température/WBGT si disponible; analyse de sensibilité avec HRmax ajustée à l'altitude; ne pas prétendre séparer fatigue et chaleur sans mesure indépendante.

### Lacune 8 — La qualité et la synchronisation des capteurs sont sous-traitées

GPS, altitude barométrique, auto-pause et HR peuvent être désynchronisés. HR a en plus un retard physiologique. Une simple jointure seconde par seconde peut attribuer à la descente la HR de la montée précédente.

**Tests nécessaires :** ceinture ECG requise ou stratification par capteur; détection des dropouts et cadence-lock; analyse de lags 0–60 s; lissage causal pour le prospectif; documentation de l'auto-pause et du temps mobile versus écoulé.

### Lacune 9 — Les modèles publient peu d'incertitude réellement prospective

MAE, MAPE et R² ne suffisent pas à un outil de course. Il faut des intervalles calibrés et une analyse des échecs. TRAP est un bon exemple par ses intervalles de passage.

**Tests nécessaires :** erreur médiane et quantiles; intervalles de prédiction; couverture à 50/80/95%; biais selon durée et dénivelé; erreurs absolues et relatives; prédictions par segment et par activité.

### Lacune 10 — Échantillons petits, masculins et dépendants

Une grande part des études trail repose sur de petits groupes masculins. Certaines bases contiennent plusieurs éditions ou plusieurs activités du même individu, ce qui viole l'indépendance si le split est aléatoire.

**Plan recommandé :** inclure si possible des femmes dans le futur échantillon; rapporter chaque athlète; grouper les splits par activité, parcours ou événement; ne jamais mélanger des segments de la même activité entre train et test.

## 4. Ce que le manuscrit peut revendiquer — et ce qu'il doit éviter

### Claims défendables

- Le modèle teste la valeur prédictive additionnelle d'une trajectoire HRR individualisée après correction du parcours.
- Il évalue séparément effets instantanés et cumulés par une échelle de modèles et des ablations.
- Il fournit une validation inter-activité conditionnelle à un athlète, avec diagnostics par terrain.
- Les simulations sans HR cible illustrent des scénarios prospectifs et leurs limites de transfert.

### Claims à éviter ou à reformuler

- **À éviter :** « first digital twin for trail running ». Jaén-Carrillo et Pattis (2026) est antérieur.
- **À éviter :** « HRR measures true instantaneous effort ». Born et al. montrent le contraire sur terrain vallonné.
- **À éviter :** « prospective prediction » pour une évaluation utilisant toute la HR de la course cible.
- **À éviter :** « validated across athletes » avec un seul athlète, ou avec 3–5 athlètes sans validation groupée.
- **À reformuler :** « TRIMP models within-race fatigue » en « a TRIMP-like cumulative state is evaluated as a predictive proxy for within-activity degradation ».

## 5. Baselines et ablations indispensables

| Niveau | Modèle | Question isolée |
|---|---|---|
| B0 | vitesse moyenne/historique simple | Le modèle bat-il une règle naïve individualisée ? |
| B1 | physique du parcours uniquement | Quel gain vient de pente/altitude ? |
| B2 | Jaén-like physique + pacing/charge | Le nouveau modèle bat-il le précédent direct à tâche égale ? |
| A1 | B1 + HRR instantané | HR cible apporte-t-elle une information résiduelle ? |
| A2 | B1 + durée ou distance cumulée | Le gain « fatigue » est-il seulement le temps écoulé ? |
| A3 | B1 + TRIMP cumulé | Le signal HR pondéré fait-il mieux que durée seule ? |
| M3 | B1 + HRR + TRIMP | Les deux composantes sont-elles complémentaires ? |
| P | route + stratégie HRR prescrite | Quelle précision reste disponible avant la course ? |

Chaque comparaison doit utiliser exactement les mêmes activités test, la même cible et la même règle de sélection. Les paramètres de grade, filtres, caps et hyperparamètres doivent être choisis à l'intérieur du fold d'entraînement, idéalement par validation imbriquée.

## 6. Corrections immédiates du fichier bibliographique actuel

1. **Boillet et al. (2024).** Remplacer `10.1038/s41598-024-71772-x` par [`10.1038/s41598-024-56042-0`](https://doi.org/10.1038/s41598-024-56042-0). Le DOI actuel correspond à un article sans rapport avec le sport.
2. **Feist et al. (2026).** Le décrire comme une contribution au *Book of Abstracts* de la conférence 4S, pas comme un article complet.
3. **Jaén-Carrillo et Pattis (2026).** Le déplacer au centre de la section Related Work et l'utiliser comme baseline direct, pas seulement comme inspiration.
4. **« Free-living wearable HR → race prediction ».** Remplacer cette entrée vague. Emig et Peltonen (2020) est une référence solide de données réelles, mais son modèle utilise surtout distance et durée, pas HR continue.
5. **GAP opérationnel.** Ne pas faire reposer la justification scientifique sur Strava/GAP. Citer Minetti pour la pente, Nicot pour la technicité manquante et Holler pour un modèle mécanique de trail plus complet.
6. **TRIMP.** Séparer clairement le modèle fitness–fatigue de Banister, la pondération individualisée de Manzi et l'usage descriptif whole-race de Fornasiero.
7. **Terminologie.** Employer systématiquement *retrospective HR-informed reconstruction*, *in-race updating* et *prospective prescribed-HRR simulation* pour trois tâches différentes.

## 7. Priorités de lecture

Si le temps est limité, lire dans cet ordre :

1. [Jaén-Carrillo & Pattis (2026)](https://doi.org/10.3390/s26123731) — précédent direct et métriques à battre.
2. [Born et al. (2017)](https://doi.org/10.1123/ijspp.2016-0101) — limite fondamentale de HR en terrain vallonné.
3. [Boillet et al. (2024)](https://www.nature.com/articles/s41598-024-56042-0) — ce qu'un digital twin physiologique fort exige réellement.
4. [Fogliato et al. (2021), TRAP](https://doi.org/10.1515/jqas-2020-0013) — différence entre modèle opérationnel et mécaniste, plus incertitude.
5. [Gutiérrez et al. (2025)](https://doi.org/10.3390/sports13110385) — baseline in-race récent et puissant.
6. [Nicot et al. (2022)](https://doi.org/10.1080/17461391.2021.1995507) — écart entre tapis/pente et vrai trail.
7. [Manzi et al. (2009)](https://doi.org/10.1249/MSS.0b013e3181a6a959) et [Fornasiero et al. (2018)](https://doi.org/10.1080/02640414.2017.1374707) — portée réelle de TRIMP.
8. [de Waal et al. (2021)](https://doi.org/10.1123/ijspp.2020-0812) — carte générale des déterminants du trail.
