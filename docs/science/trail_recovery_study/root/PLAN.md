# Plan d'expérience RunningManager

Version initiale du 20 novembre 2026, avant lecture des résultats des nouvelles expériences.

| Lot | Protocole | Critère de décision |
|---|---|---|
| PR #5 | Vérifier le diff, les tests, la reproductibilité sur les entrées disponibles, le sens des comparaisons et l'antériorité des entrées. | Accepter si les conclusions correspondent au protocole, si les validations passent et si les défauts susceptibles de modifier les conclusions sont résolus. |
| H1, récupération continue | Comparer sans fatigue, TRIMP cumulé, oubli par segment et oubli en temps réel, avec le même jeu d'informations fourni à l'inférence et réglages sur les seules sorties d'apprentissage. | Gain sur sorties entières mises de côté. Donner MAE, MAPE, biais et différences appariées. Séparer HRR moyenne réalisée et HRR fixe. |
| H2, marche et arrêt | Vérifier les transitions sur trajectoires synthétiques. Auditer les flux horodatés et quantifier les épisodes observables, avec seuils de détection explicites. | Conclure sur les propriétés mathématiques, l'observabilité et les associations effectivement calculées. Une HRR faible ou une vitesse lente seule ne constitue pas une annotation de marche. |
| H3, ravitos et persistance | Comparer ARD1, ARD∞ et récupération continue dans des cas synthétiques. Ne calibrer un effet d'intervention que si horaires, durée et nature d'un nombre suffisant d'arrêts sont établis. | Distinguer bénéfice pendant l'arrêt, perte après reprise, modulation tardive et consommation. Demander validation hors course. |
| H4, ETA | Tester statique, AR(1), OU à temps variable et une persistance de course à partir des seuls résidus déjà disponibles au moment de la prédiction. | Mesurer erreur et couverture à un pas avec séparation par sortie. Qualifier le statut des prédictions de base et des signaux d'entrée. |

## Journal des révisions

- L'audit de la PR #5 indique une référence E2 mal choisie dans le gate, un dénominateur E3 incorrect et une référence de normalisation E4 qui utilise des données futures. La décision GitHub dépendra de leur vérification indépendante.
- Les flux ne figurent pas au head de la PR #5. La branche portable #4 est examinée comme source compatible, avec vérification des empreintes et des couvertures. Les exports anonymisés fournissent un second support stable pour les expériences nouvelles.

- Rejeu complet réalisé avec les flux #4. B0 à B3, E5 et les points E4 reproduisent l'archive. Les sept échecs de la suite se retrouvent sur la base. Le correctif local change le statut E4 en passe sur le rejeu de 18 événements. Les routes E4 proviennent toujours des traces réalisées. La PR reste en brouillon, et son patch d'audit est fourni pour une future revue.
- H1 close pour cet historique. L'oubli par segment demeure le comparateur principal. Les variantes 30 min, 90 min et le choix des deux sur historique n'ont pas apporté d'amélioration reproductible du chrono global.
- H2 close pour les tests déterministes et l'inventaire des épisodes. L'effet physiologique reste ouvert. La prochaine acquisition doit distinguer marche volontaire, course lente, pause réelle et lacune d'enregistrement, avec durée et intensité. Utiliser des répétitions sur la même boucle et des ordres équilibrés, avec une même portion de reprise. Mesurer le temps perdu pendant le bloc et le temps regagné ensuite.
- H3 close pour la comparaison synthétique et l'audit de faisabilité. Pour un essai, enregistrer heure et durée de chaque intervention, apports éventuels, charge antérieure et reprise. Au moins deux interventions annotées dans les sorties permettant de comparer ARD₁ et ARD∞, avec répétitions en début et fin et une variation de la durée suffisamment indépendante de la fatigue initiale. Comparer d'abord le modèle continu seul, puis le terme de saut additionnel uniquement s'il améliore les prédictions hors sortie.
- H4 close pour le rejeu en course. Prochaine étape de développement si retenue, instrumenter une correction du résidu à partir du dernier segment terminé et tracer l'erreur du temps restant avec et sans correction. Comparateurs fixes : dernier résidu, moyenne de course, AR(1). Réserver un nouveau groupe de parcours et tester des horizons au chrono écoulé, en intégrant les arrêts. Couverture d'un intervalle avant départ à évaluer séparément avec des traces GPX disponibles avant les événements.

- Le correctif et les annexes ont été regroupés dans une PR de suivi. La PR #5 reste la base du diff.
