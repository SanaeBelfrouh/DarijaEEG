#!/usr/bin/env python3
"""Etape 2 : l'echelle complete, modele x schema x contraste.

    python scripts/02_benchmark.py --epochs epochs.npz --condition imagined

Ce que le tableau de sortie permet de lire, et que la ligne de base actuelle ne
permettait pas :

1. ``puissance_lda`` en ``loso`` reproduit le resultat actuel. Toute autre ligne
   se lit comme un ecart par rapport a elle, sur les memes plis.
2. La colonne ``schema`` separe deux questions que le resultat a 6-8 % melangeait :
   "y a-t-il du signal" (``within_session``) et "le signal survit-il au changement
   de session" (``loso``).
3. La metrique rapportee et testee est l'exactitude EQUILIBREE, dont le hasard
   vaut exactement 1/n_classes quel que soit le desequilibre des classes.
   L'exactitude brute et le taux de la classe majoritaire restent dans le CSV,
   a titre descriptif, mais ne sont jamais testes : sur des classes
   desequilibrees ils produisent des lignes ininterpretables.
4. La colonne ``seuil_eq`` donne, pour chaque ligne, la plus petite exactitude
   equilibree vraie que ce nombre d'essais permettait de distinguer du hasard.
   Un p non significatif au-dessous de ce seuil n'est pas une absence d'effet.
5. La condition ``articulated`` sert de temoin positif : elle doit ressortir. Si
   elle ne ressort pas, c'est le pipeline qui est en cause, pas le sujet.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.evaluate import COVARIANCE_REPRESENTATIONS, evaluate_many  # noqa: E402
from darija.pipelines import build_representations, representation_for  # noqa: E402
from darija.preprocess import crop                                     # noqa: E402
from darija.stats import holm_bonferroni                               # noqa: E402

warnings.filterwarnings("ignore")

# cov_mdm est disponible mais hors defaut : il coute 300-500 s par combinaison et
# s'effondre systematiquement sur une classe, ce que l'exactitude equilibree rend
# desormais visible. Le rajouter explicitement avec --models si on veut la
# reference sans hyperparametre.
DEFAULT_MODELS = ("puissance_lda", "csp_lda", "cov_tangent", "bankcov_tangent")
DEFAULT_SCHEMES = ("within_session", "pooled_blocked", "loso")
DEFAULT_CONTRASTS = ("mots15", "oui_non", "syllabes", "semantique")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", default="epochs.npz")
    parser.add_argument("--condition", default="imagined",
                        choices=("imagined", "articulated", "both"))
    parser.add_argument("--tmin", type=float, default=0.0,
                        help="debut de la fenetre analysee, relatif au go")
    parser.add_argument("--tmax", type=float, default=2.0)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--schemes", nargs="+", default=list(DEFAULT_SCHEMES))
    parser.add_argument("--contrasts", nargs="+", default=list(DEFAULT_CONTRASTS))
    parser.add_argument("--recenter", action="store_true",
                        help="ajoute la variante recentree par session")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--out", default="results/benchmark.csv")
    args = parser.parse_args()

    archive = np.load(args.epochs, allow_pickle=True)
    X, times, sfreq = archive["X"], archive["times"], float(archive["sfreq"])
    y, cond = archive["y"], archive["cond"]
    session, trial_index = archive["session"], archive["trial_index"]

    if args.condition != "both":
        mask = cond == args.condition
        X, y, session, trial_index = X[mask], y[mask], session[mask], trial_index[mask]

    X, times = crop(X, times, args.tmin, args.tmax)
    print(f"{len(X)} essais | {X.shape[1]} canaux | fenetre "
          f"[{times[0]:.2f}, {times[-1]:.2f}] s | {len(np.unique(session))} sessions\n")

    needed = {representation_for(m) for m in args.models}
    t0 = time.time()
    representations = build_representations(X, sfreq, include=tuple(needed))
    print(f"representations calculees en {time.time() - t0:.0f} s : "
          f"{ {k: v.shape for k, v in representations.items()} }\n")

    recenter_options = (False, True) if args.recenter else (False,)
    rows, results = [], []

    # les contrastes sont en boucle INTERNE : pour un couple (modele, schema), le
    # prefixe independant des etiquettes est ajuste une fois par pli et sert a
    # tous les contrastes partageant le meme masque d'essais
    for model, scheme, recenter in itertools.product(
        args.models, args.schemes, recenter_options
    ):
        if recenter and representation_for(model) not in COVARIANCE_REPRESENTATIONS:
            continue
        t0 = time.time()
        block = evaluate_many(
            representations, y, session, trial_index,
            model=model, scheme=scheme, contrasts=tuple(args.contrasts),
            recenter=recenter, n_splits=args.n_splits, n_perm=args.n_perm,
        )
        elapsed = time.time() - t0
        results.extend(block)
        for result in block:
            row = result.row()
            rows.append(row)
            flag = "  <-- au-dessus du seuil" if (
                row["equilibree"] > row["seuil_eq"]) else ""
            print(f"  {model:18s} {scheme:15s} {row['contraste']:12s} "
                  f"{'recentre' if recenter else '        '} : "
                  f"equilibree {row['equilibree']:.4f} "
                  f"(hasard {row['hasard_eq']:.4f}, seuil {row['seuil_eq']:.4f})  "
                  f"p={row['p_perm']:.4f}{flag}", flush=True)
        print(f"    [{elapsed:.0f}s pour {len(block)} contrastes]", flush=True)

    table = pd.DataFrame(rows)
    # Holm par contraste : c'est la famille de tests qui repond a une question
    table["p_holm"] = np.nan
    for _, block in table.groupby("contraste"):
        table.loc[block.index, "p_holm"] = holm_bonferroni(block["p_perm"].values)
    table["significatif_holm"] = table["p_holm"] < 0.05

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    print("\n" + "=" * 100)
    print("TABLEAU COMPLET")
    print("=" * 100)
    print(table.to_string(index=False))

    print("\n" + "=" * 100)
    print("LECTURE")
    print("=" * 100)
    survivors = table[table.significatif_holm]
    if len(survivors):
        print("Combinaisons survivant a la correction de Holm dans leur famille :")
        print(survivors.to_string(index=False))
    else:
        print("Aucune combinaison ne survit a la correction pour tests multiples.")

    print("\nPar contraste, meilleure exactitude equilibree obtenue :")
    for contrast, block in table.groupby("contraste"):
        best = block.loc[block.equilibree.idxmax()]
        verdict = ("AU-DESSUS du seuil" if best.equilibree > best.seuil_eq
                   else "sous le seuil de detectabilite")
        print(f"  {contrast:12s} {best.equilibree:.4f} "
              f"(hasard {best.hasard_eq:.4f}, seuil {best.seuil_eq:.4f}) "
              f"par {best.modele}/{best.schema}  -> {verdict}")

    flat = table[table.contraste == "mots15"]
    if len(flat) and flat.equilibree.max() < flat.seuil_eq.iloc[0]:
        print("\n  Sur les 15 mots, aucun modele n'atteint le seuil de detectabilite.")
        print("  Cela ne dit PAS 'aucun signal' : cela dit 'aucun signal au-dessus de")
        print(f"  {flat.seuil_eq.iloc[0]:.4f} en exactitude equilibree'. Avec {int(flat.n.iloc[0])} essais et")
        print("  15 classes, l'experience ne pouvait pas trancher en dessous de ce seuil.")

    np.savez_compressed(
        Path(args.out).with_suffix(".confusions.npz"),
        **{f"{r.model}|{r.scheme}|{r.contrast}|{int(r.recentered)}": r.confusion
           for r in results},
    )
    print(f"\nresultats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
