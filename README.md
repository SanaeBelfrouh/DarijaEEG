# DarijaEEG — décodage direct des 15 mots imaginés

Évaluation du protocole de collecte, correction d'une conclusion du carnet de
timing, et banc d'essai de méthodes de décodage qui n'ont pas encore été
essayées. Aucun transfert depuis la condition articulée : tout ce qui est
proposé ici décode les 15 mots **directement** depuis la condition imaginée.

---

## 1. Le protocole de collecte est-il juste ?

Oui dans l'ensemble, et sur plusieurs points il est plus rigoureux que la
majorité de ce qui se publie en parole imaginée. Six points qui tiennent :

| Point | Pourquoi c'est important |
|---|---|
| ISI aléatoire de 0,8–1,2 s entre l'indice et le go | L'indice visuel est décorrélé du début de production. C'est **la** protection contre la fuite la plus répandue du domaine : un classifieur qui lit en réalité le potentiel évoqué par la lecture du mot. |
| Fenêtre de production fixée à 2,0 s pour tous les mots | La condition imaginée ne porte aucun confondu de durée. L'articulée en porte un (la durée réelle varie avec le mot), ce qui suffit à expliquer une partie de ses 12,6 %. |
| Ordre des mots tiré au hasard dans chaque bloc | Les plis chronologiques ne sont pas des plis par classe. Beaucoup de jeux publiés échouent ici. |
| 15 répétitions × 15 mots × 8 sessions | 1800 essais imaginés, 120 par mot. C'est beaucoup pour un sujet unique. |
| Condition articulée comme témoin positif | Sans elle, un résultat au hasard en imaginé serait ininterprétable : impossible de distinguer « pas de signal » de « pipeline cassé ». |
| Permutations groupées par session, correction pour tests multiples | L'analyse existante évite déjà la pseudo-réplication, ce qui est rare. |

### Les quatre problèmes réels

**1. L'ordre des blocs n'est pas contrebalancé.** Imaginé est toujours le bloc 1,
articulé toujours le bloc 2, dans les huit sessions. La condition est donc
parfaitement confondue avec le temps passé sur la tâche : fatigue, dérive
d'impédance, assèchement du gel. Cela n'affecte pas le décodage des 15 mots
*à l'intérieur* de l'imaginé, mais cela affecte toute comparaison
imaginé/articulé — y compris la conclusion « pas d'EMG en imaginé ». Le ratio de
puissance articulé/imaginé de 1,37 sur le montage complet est peut-être en
partie un effet d'ordre. Ce n'est pas rattrapable a posteriori ; il faut
l'énoncer comme limite. Pour les sessions futures : alterner A-B / B-A.

**2. Aucune vérification de la conformité de l'imagerie.** Rien dans le
protocole n'atteste que le sujet a effectivement imaginé le mot à un essai
donné, ni ne permet d'écarter les essais où l'attention a décroché. Sur 225
essais consécutifs d'une tâche sans retour ni réponse, c'est la limite la plus
lourde. Remède pour les sessions futures : des essais sondes occasionnels (5 %)
qui demandent après coup « quel mot venait d'être imaginé ? » en choix forcé.
Cela donne un taux de conformité, et de quoi écarter les blocs décrochés.

**3. Pas d'EMG dédié.** FT9/FT10/TP9/TP10 sont utilisées comme approximation,
ce qui est défendable, mais une affirmation aussi forte que « aucune
articulation subvocale jusqu'à 300 Hz » mériterait une dérivation EMG de surface
sur l'orbiculaire des lèvres ou le menton.

**4. L'indice est un mot écrit en arabizi.** Voir `3afak` à l'écran puis
l'imaginer engage probablement une part de traitement orthographique, d'autant
que les chiffres-lettres (`3`, `7`) n'ont pas de correspondance phonologique
apprise aussi automatique qu'un graphème. Un indice **auditif** produirait une
représentation phonologique plus régulière. À considérer pour une V2 du
protocole.

À quoi s'ajoute une limite qu'il suffit d'assumer : **un seul sujet**. Aucune
généralisation possible, mais une preuve de concept intra-sujet reste légitime
si elle est présentée comme telle.

---

## 2. Correction : le carnet de timing conclut à l'envers

Le carnet mesure un écart-type de **114,4 ms** sur l'intervalle
`cue_end -> start` et conclut : *« REAL JITTER (> 50 ms) : les époques à
décalage fixe sont étalées, V3 est justifié. »*

Cette conclusion est fausse, et le fichier PsychoPy le montre directement :

```python
isiDur = random.uniform(0.8, 1.2)     # routine isi, EEG_Darija_Multimodal_FINAL_15.psyexp
```

L'écart-type d'un tirage uniforme sur 0,4 s vaut `0.4/√12 = 115,5 ms`. Les
114,4 ms mesurés sont donc **la gigue voulue par le protocole**, reproduite à
1 ms près, et non une imprécision d'horloge.

La conséquence pratique est l'inverse de celle qui a été tirée :

- les époques sont ancrées sur `{cond}_start`, donc cette variabilité se situe
  **entièrement avant t = 0** et ne peut pas étaler la fenêtre de production ;
- l'intervalle qui détermine réellement l'alignement est `start -> end`, dont
  l'écart-type mesuré vaut **1,1 ms**, soit un quart d'échantillon à 250 Hz ;
- par la règle de décision du carnet lui-même (« < 20 ms → timing logiciel,
  re-découper ne peut rien récupérer »), appliquée à l'intervalle pertinent, la
  réponse est : **V3 ne vaut pas la peine d'être construit**.

`scripts/00_timing_check.py` refait la mesure en comparant chaque intervalle à
ce que le protocole prescrit, et attribue la gigue au lieu de la constater.

Une réserve subsiste, mineure : une ligne de base prise sur −1,0 à 0 s déborde
sur la fin de l'indice pour les essais dont l'ISI est le plus court (0,8 s).
C'est pourquoi la fenêtre par défaut ici commence à **−0,5 s**.

---

## 3. Ce que ces données permettaient de détecter

La question à poser avant d'interpréter un résultat non significatif :
l'expérience avait-elle les moyens de voir l'effet recherché ? Test binomial
unilatéral, puissance 80 %, α = 0,05 :

| Contraste | n | hasard | seuil détectable | écart requis |
|---|---:|---:|---:|---:|
| 15 mots | 1800 | 6,67 % | **8,18 %** | +1,5 pt (1,23 × hasard) |
| syllabes 1 vs 2 | 1800 | 53,3 % | 56,3 % | +2,9 pt (1,055 ×) |
| pharyngale /ʕ/ | 1800 | 66,7 % | 69,4 % | +2,7 pt (1,041 ×) |
| détection vs repos | 3600 | 50,0 % | 52,1 % | +2,1 pt (1,041 ×) |
| oui/non (`Ah` vs `lla`) | 240 | 50,0 % | 58,0 % | +8,0 pt (1,16 ×) |

Trois lectures en découlent.

**Le meilleur résultat imaginé rapporté, 8,22 %, tombe exactement sur le seuil
de détectabilité (8,18 %).** Il n'est donc pas surprenant qu'il ne survive pas à
la correction pour 36 tests. Ce résultat ne dit pas « pas de signal » ; il dit
« pas de signal au-dessus de ~8,2 % ». La nuance change ce qui peut être écrit
dans un article.

**Les contrastes binaires à grand effectif sont bien plus sensibles**, en écart
relatif au hasard : détecter un effet demande 1,04 × le hasard contre 1,23 × pour
les 15 classes. C'est le meilleur levier disponible sans enregistrer un essai de
plus.

**Le contraste oui/non est le moins puissant du lot**, contrairement à
l'intuition : `Ah` et `lla` n'ont que 120 essais chacun. S'il compte
cliniquement — et pour une ICM destinée à la paralysie, il compte — il faut le
sur-échantillonner dans le protocole, pas l'extraire après coup.

Pour référence, atteindre 80 % de puissance sur une exactitude vraie de 8 % à 15
classes demanderait **2295 essais**, soit environ 10 sessions ; pour 7,5 %, il en
faudrait 5752, soit 26 sessions.

---

## 4. Les techniques qui n'ont pas encore été essayées

Par ordre de rapport gain attendu / effort. Tout est implémenté dans `src/darija/`.

### Niveau 1 — le schéma d'évaluation (gain le plus probable, aucun modèle nouveau)

Le résultat actuel utilise `GroupKFold` par session, c'est-à-dire du
**leave-one-session-out**. C'est le test le plus sévère qui existe : il exige
que le décodeur survive au déplacement de domaine entre sessions. Un résultat au
hasard en LOSO ne prouve pas l'absence de signal, il prouve l'absence d'un
signal *stable entre sessions*.

Deux schémas manquent, et ils répondent à des questions différentes :

- **`within_session`** — un modèle par session, plis chronologiques contigus,
  jamais mélangés. C'est le test de « y a-t-il du signal, tout court ». S'il
  existe un effet, c'est ici qu'il apparaîtra en premier.
- **`pooled_blocked`** — le bloc chronologique *k* de chaque session forme
  ensemble le pli de test. Compromis entre les deux.

Les trois sont dans `src/darija/cv.py` et le banc d'essai les compare
systématiquement. **Ne jamais mélanger les essais à l'intérieur d'une session** :
des essais voisins partagent une dérive lente, et un mélange aléatoire la
transforme en information apparemment décodable. C'est l'origine d'une bonne
part des 70–90 % publiés.

### Niveau 2 — le recentrage riemannien par session

Chaque session est transportée sur la variété de sorte que sa propre moyenne
géométrique devienne l'identité. **Aucune étiquette n'est utilisée** : c'est une
adaptation de domaine non supervisée, pas du transfert au sens où tu l'écartes.
Pour une ICM réelle c'est légitime, puisqu'on dispose toujours de données de
calibration non étiquetées.

C'est le levier unique le plus rentable en EEG. Sur données synthétiques avec un
déplacement de domaine réaliste, il fait passer le décodage de 46,5 % à 54,3 %
(`tests/test_pipeline.py::test_recentering_rescues_a_session_shift`). À rapporter
séparément du résultat sans recentrage, puisqu'il touche aussi la session de
test.

### Niveau 3 — les représentations

La ligne de base actuelle applique une LDA à des **enveloppes de puissance par
bande** : elle jette toute la structure de covariance spatiale. C'est la marge
de progression la plus évidente.

| Modèle | Ce qu'il apporte |
|---|---|
| `cov_tangent` | Covariance spatiale → espace tangent → régression logistique multinomiale. La référence classique la plus solide sur les petits jeux EEG. |
| `bankcov_tangent` | Une covariance par bande (δ θ α β₁ β₂ γ₁), espace tangent par bande, concaténation. |
| `augcov_tangent` | Covariances augmentées par retards : la covariance spatiale ordinaire est invariante par permutation temporelle, donc elle ignore entièrement la dynamique. Le plongement par retards la réintroduit. |
| `cov_mdm` | Distance minimale à la moyenne riemannienne. Sans hyperparamètre, quasi insensible au sur-apprentissage : s'il est au hasard alors que `cov_tangent` ne l'est pas, la différence vient de la capacité du modèle, pas du signal. |
| `csp_lda` | CSP un-contre-tous. À avoir comme point de comparaison. |

### Niveau 4 — reformuler le problème (meilleure chance d'un résultat positif)

Attaquer 15 classes de front est le pire usage possible de ces 1800 essais. Le
module `src/darija/contrasts.py` fournit les décompositions :

- **détection** — parole imaginée contre repos (ITI/ISI). Si même cela échoue,
  cela borne tout le reste. À faire en premier.
- **syllabes**, **attaque**, **pharyngale** — si l'imagerie porte une trace
  phonologique, elle apparaîtra sur ces axes avant d'apparaître sur l'identité
  du mot.
- **sémantique** — requête / objet / action / état / polarité / politesse.
- **oui/non** — le plus utile cliniquement, le moins puissant tel quel (voir §3).

Un décodeur hiérarchique (détecter → catégorie → mot) offre une voie vers une
performance utilisable qu'un classifieur plat à 15 voies n'offrira pas.

### Niveau 5 — l'ASR, plus sensible que la classification

`scripts/03_rsa.py` calcule une matrice de dissimilarité **validée croisée**
(crossnobis) entre les 15 mots, puis la corrèle à des matrices modèles
phonologiques et sémantiques par test de Mantel.

L'intérêt : l'ASR agrège les 105 paires au lieu de tout résumer par une
exactitude, et elle peut ressortir significative là où le classifieur reste au
hasard. Une matrice de confusion structurée par la phonologie est une preuve de
signal réel même à 8 % d'exactitude — argument bien plus fort qu'un p à la
limite.

L'estimateur crossnobis est d'espérance nulle sous l'hypothèse nulle,
contrairement à une distance de Mahalanobis ordinaire qui est toujours positive
et donc ininterprétable seule. Vérifié : 0,10 ± 0,57 sans effet planté, +2,79
avec.

### Niveau 6 — l'apprentissage profond, sans illusion

EEGNet, ShallowFBCSPNet, EEG-Conformer : avec 1800 essais et 15 classes,
l'attente raisonnable est qu'ils **égalent ou perdent** contre la représentation
tangente. Ils ne sont pas implémentés ici, délibérément — une nuit de calcul
pour un résultat prévisible n'est pas prioritaire tant que les niveaux 1 à 5 ne
sont pas faits.

La seule direction profonde avec un vrai potentiel : un **pré-entraînement
auto-supervisé** sur les données continues non étiquetées des 8 sessions
(masquage ou prédiction contrastive), puis une sonde linéaire. C'est du
pré-entraînement sur ses propres données, pas du transfert inter-sujets.

---

## 5. Ce que dit la littérature, honnêtement

Aucune étude en parole imaginée sur EEG de scalp avec une validation croisée
propre et un test de permutation n'a démontré un décodage à 15 classes
nettement au-dessus du hasard. Le plus grand jeu public d'inner speech
(Nieto et al., *Scientific Data* 2022, 10 sujets, 4 classes) donne des
résultats essentiellement au niveau du hasard. Les 70–90 % que l'on croise
viennent presque toujours de l'une de ces quatre sources : époques contenant
l'indice, normalisation appliquée avant le découpage en plis, validation croisée
mélangeant des essais voisins dans le temps, ou ordre des blocs confondu avec
la classe.

Ce qui veut dire qu'un **résultat nul rigoureux est ici un résultat publiable**,
et probablement plus utile au domaine qu'un cinquième article à 80 %. Les
ingrédients sont déjà réunis : un témoin positif qui fonctionne (l'articulé
décode), une analyse de puissance qui borne ce qui pouvait être détecté, et un
protocole dont la gigue d'ISI exclut la fuite par potentiel évoqué. C'est un
ensemble solide. Ce qui manque, c'est le niveau 1 et le niveau 4 — et ils
peuvent encore faire apparaître quelque chose.

---

## 6. Utilisation

```bash
pip install -r requirements.txt

# 0. attribution de la gigue — marqueurs seuls, aucun EEG chargé, ~2 min
python scripts/00_timing_check.py --input /kaggle/input

# 1. XDF bruts -> archive d'époques ancrées sur le go, une seule fois
python scripts/01_build_epochs.py --input /kaggle/input --out epochs.npz

# 2. l'échelle complète : modèle x schéma x contraste
python scripts/02_benchmark.py --epochs epochs.npz --condition imagined --recenter

# le témoin positif : doit ressortir, sinon c'est le pipeline qui est en cause
python scripts/02_benchmark.py --epochs epochs.npz --condition articulated

# 3. similarité représentationnelle des 15 mots
python scripts/03_rsa.py --epochs epochs.npz --condition imagined

# tests (9 tests, dont un témoin positif et un témoin négatif du code lui-même)
python tests/test_pipeline.py
```

### Si l'étape 1 rejette tous les essais

Le script imprime l'amplitude crête-à-crête observée par session. La lire avant
de toucher au moindre seuil :

- **amplitude médiane de l'ordre de 10–60 µV** → l'échelle est bonne. Si le rejet
  est malgré tout massif, desserrer `--max-bad-fraction 0.3` ou augmenter
  `--reject-uv`.
- **amplitude médiane très loin de cette plage** → le flux LSL n'est pas en
  microvolts et aucun seuil exprimé en µV n'a de sens. Relancer avec
  `--scale 1e6` (flux en volts) et vérifier que la médiane revient dans la plage.
- **pour voir les données sans aucun filtre de qualité** : `--no-reject`.

Le critère porte sur la **proportion de canaux** hors bornes, pas sur le pire
canal. Sur un montage 64 canaux avec fixation visuelle, exiger que les 64
restent sous un seuil revient à exiger qu'aucun clignement ni aucune électrode
bruyante n'ait touché une seule dérivation pendant 2,5 s — ce qui rejette la
quasi-totalité des essais exploitables.

### Comment lire le tableau de sortie

- `puissance_lda` / `loso` reproduit le résultat actuel. Toute autre ligne se lit
  comme un écart **par rapport à elle, sur exactement les mêmes plis**.
- `seuil_detectable` donne, ligne par ligne, la plus petite exactitude vraie que
  ce nombre d'essais permettait de distinguer du hasard. Un p non significatif
  au-dessous de ce seuil n'est pas une absence d'effet.
- `classes_manquantes` doit valoir 0. Toute autre valeur signale un découpage
  dans lequel l'entraînement ne voit pas une classe testée, ce qui pousse
  mécaniquement l'exactitude sous le hasard.
- La correction de Holm est appliquée **par famille de contraste**, pas
  globalement : c'est le contraste qui définit la question posée.

## 7. Garde-fous intégrés

Ce qui, par construction, ne peut pas fuir ici :

- la référence de l'espace tangent est la moyenne géométrique du **pli
  d'entraînement seul** ; la calculer sur toutes les données ferait passer de
  l'information du test dans la représentation ;
- toute mise à l'échelle est un étage de `Pipeline` sklearn, donc ajustée sur
  l'entraînement seul ;
- les plis intra-session sont **chronologiques et contigus**, jamais mélangés ;
- les permutations se font **à l'intérieur de chaque session**, ce qui préserve
  la structure de groupe ;
- la fenêtre par défaut commence à −0,5 s, ce qui reste dans l'ISI même pour
  l'ISI le plus court : aucune époque ne contient le mot affiché ;
- le hasard est celui du **classifieur majoritaire**, pas 1/n_classes : avec des
  classes déséquilibrées, 1/n_classes sous-estime le hasard et rend significatif
  un décodeur qui n'a rien appris.

Et l'ordre des opérations : le filtrage a lieu sur le signal **continu**, avant
le découpage. Filtrer époque par époque laisse des transitoires de bord, et sur
un amplificateur couplé en continu comme l'actiCHamp, dont la ligne de base
dérive de plusieurs millivolts, ces transitoires dépassent l'amplitude de l'EEG
lui-même.

La suite de tests contient son propre témoin positif et son propre témoin
négatif : le pipeline doit retrouver un effet planté, et ne doit pas en
inventer quand il n'y en a pas. Elle a déjà servi deux fois — elle a attrapé un
bug dans `distance_riemann`, qui appliquait `eigvalsh` à `A⁻¹B`, matrice non
symétrique, ce qui corrompait silencieusement les prédictions de `cov_mdm` ;
et elle porte désormais la non-régression du critère de rejet, qui éliminait
100 % des essais sur des données parfaitement exploitables.

## 8. Annotations linguistiques à relire

`src/darija/contrasts.py` encode pour chaque mot le nombre de syllabes, la
classe articulatoire de l'attaque, le voisement, la présence de la pharyngale et
la catégorie sémantique. Ce sont des hypothèses de travail. La syllabation de
`dwa`, `lma`, `lla` et `tbib` dépend du débit et mérite une relecture de
locutrice native avant toute analyse phonologique.
