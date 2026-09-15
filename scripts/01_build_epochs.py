#!/usr/bin/env python3
"""Etape 1 : XDF bruts -> archive d'epoques verrouillees par essai.

A executer une fois, sur le noyau qui a les XDF attaches. Produit un seul .npz
que les etapes suivantes relisent en quelques secondes.

    python scripts/01_build_epochs.py --input /kaggle/input --out epochs.npz

Les epoques sont ancrees sur ``{condition}_start`` (le go), pas sur l'indice
visuel. La fenetre par defaut commence a -0.5 s, ce qui reste dans l'ISI meme
pour l'ISI le plus court (0.8 s) : aucune epoque ne contient le mot affiche.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from darija.io_xdf import DEFAULT_TMAX, DEFAULT_TMIN, discover_xdf, load_session  # noqa: E402
from darija.preprocess import prepare, reject_trials                             # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="/kaggle/input", help="racine des XDF")
    parser.add_argument("--out", default="epochs.npz")
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--low", type=float, default=0.5)
    parser.add_argument("--high", type=float, default=45.0)
    parser.add_argument("--reject-uv", type=float, default=200.0,
                        help="amplitude crete-a-crete maximale toleree")
    args = parser.parse_args()

    files = discover_xdf(args.input)
    if not files:
        print(f"aucun .xdf sous {args.input}", file=sys.stderr)
        return 1

    all_X, all_y, all_cond, all_session, all_index = [], [], [], [], []
    ch_names = None

    for session, path in sorted(files.items()):
        epochs = load_session(path, tmin=args.tmin, tmax=args.tmax,
                              target_sfreq=args.sfreq, session=session)
        X = prepare(epochs.X, epochs.sfreq, low=args.low, high=args.high)
        keep = reject_trials(X, amplitude_uv=args.reject_uv)

        if ch_names is None:
            ch_names = epochs.ch_names
        elif ch_names != epochs.ch_names:
            raise RuntimeError(f"{session}: montage different des sessions precedentes")

        all_X.append(X[keep].astype(np.float32))
        all_y.append(epochs.y[keep])
        all_cond.append(epochs.cond[keep])
        all_session.append(np.full(int(keep.sum()), session))
        all_index.append(epochs.trial_index[keep])

        print(f"[{session}] {keep.sum()}/{len(keep)} essais conserves "
              f"({100 * (1 - keep.mean()):.1f} % rejetes) | {epochs.orig_sfreq:.0f} Hz "
              f"-> {epochs.sfreq:.0f} Hz", flush=True)

    X = np.concatenate(all_X)
    times = args.tmin + np.arange(X.shape[-1]) / args.sfreq

    np.savez_compressed(
        args.out,
        X=X,
        y=np.concatenate(all_y).astype(str),
        cond=np.concatenate(all_cond),
        session=np.concatenate(all_session),
        trial_index=np.concatenate(all_index),
        times=times,
        ch_names=np.asarray(ch_names),
        sfreq=args.sfreq,
    )

    size_gb = Path(args.out).stat().st_size / 1e9
    print(f"\n{len(X)} epoques, {X.shape[1]} canaux, {X.shape[2]} echantillons "
          f"-> {args.out} ({size_gb:.2f} Go)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
