"""Moteur d'evaluation : un modele, un schema de plis, un contraste.

Une seule fonction fait tout le travail et renvoie un enregistrement complet :
exactitude, F1 macro, intervalle de Wilson, p par permutation groupee, matrice
de confusion, et l'exactitude minimale detectable pour ce nombre d'essais. Ce
dernier champ est la raison d'etre du module : un p non significatif accompagne
d'une exactitude minimale detectable de 8,1 % ne dit pas "pas de signal", il dit
"pas de signal superieur a 8,1 %".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contrasts import apply_contrast
from .cv import make_splits
from .pipelines import make_model, representation_for
from .riemann import recenter_by_group
from .stats import (
    accuracy,
    binomial_ci,
    macro_f1,
    minimum_detectable_accuracy,
    permutation_pvalue,
)

#: representations sur lesquelles le recentrage riemannien a un sens
COVARIANCE_REPRESENTATIONS = ("cov", "bankcov", "augcov")


@dataclass
class Result:
    model: str
    scheme: str
    contrast: str
    recentered: bool
    n_trials: int
    n_classes: int
    chance: float
    accuracy: float
    macro_f1: float
    ci_low: float
    ci_high: float
    pvalue: float
    mde: float
    #: essais de test dont la classe manquait au pli d'entrainement. Doit valoir
    #: zero ; toute autre valeur invalide l'exactitude de la ligne.
    unreachable: int = 0
    labels: list[str] = field(default_factory=list)
    confusion: np.ndarray | None = None
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None

    def row(self) -> dict:
        """Version plate, pour un tableau de resultats."""
        return {
            "modele": self.model,
            "schema": self.scheme,
            "contraste": self.contrast,
            "recentre": self.recentered,
            "n": self.n_trials,
            "classes": self.n_classes,
            "hasard": round(self.chance, 4),
            "exactitude": round(self.accuracy, 4),
            "f1_macro": round(self.macro_f1, 4),
            "ic95": f"[{self.ci_low:.3f}, {self.ci_high:.3f}]",
            "p_perm": round(self.pvalue, 4),
            "seuil_detectable": round(self.mde, 4),
            "classes_manquantes": self.unreachable,
            "significatif": self.pvalue < 0.05,
        }


def _recenter(representation: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recentrage par session, en gerant les representations a banc de bandes."""
    if representation.ndim == 4:      # (n, n_bands, c, c)
        return np.stack(
            [recenter_by_group(representation[:, b], groups)
             for b in range(representation.shape[1])],
            axis=1,
        )
    return recenter_by_group(representation, groups)


def evaluate(
    representations: dict[str, np.ndarray],
    words: np.ndarray,
    groups: np.ndarray,
    trial_index: np.ndarray,
    *,
    model: str,
    scheme: str = "loso",
    contrast: str = "mots15",
    recenter: bool = False,
    n_splits: int = 5,
    n_perm: int = 5000,
    seed: int = 0,
) -> Result:
    """Evalue une combinaison et renvoie l'enregistrement complet."""
    labels, mask, chance = apply_contrast(contrast, words)

    representation_name = representation_for(model)
    X = representations[representation_name]
    if recenter and representation_name not in COVARIANCE_REPRESENTATIONS:
        raise ValueError(f"recentrage inapplicable a la representation {representation_name}")

    X = X[mask]
    groups_c = np.asarray(groups)[mask]
    trial_index_c = np.asarray(trial_index)[mask]

    if recenter:
        X = _recenter(X, groups_c)

    splits = make_splits(scheme, groups_c, trial_index_c, n_splits=n_splits)
    predictions = np.empty(len(labels), dtype=object)
    predicted = np.zeros(len(labels), dtype=bool)
    unreachable = 0

    for train, test in splits:
        train_classes = np.unique(labels[train])
        if len(train_classes) < 2:
            continue
        # un essai dont la classe est absente du pli d'entrainement ne peut pas
        # etre predit correctement : le compter comme une erreur ferait passer
        # un defaut de decoupage pour un resultat au-dessous du hasard
        unreachable += int(np.sum(~np.isin(labels[test], train_classes)))

        estimator = make_model(model)
        estimator.fit(X[train], labels[train])
        predictions[test] = estimator.predict(X[test])
        predicted[test] = True

    if unreachable:
        import warnings

        warnings.warn(
            f"{model}/{scheme}/{contrast} : {unreachable} essais de test ont une "
            f"classe absente de leur pli d'entrainement. L'exactitude rapportee "
            f"est mecaniquement sous le hasard ; verifier que trial_index reflete "
            f"bien un ordre de presentation aleatoire par rapport aux classes.",
            RuntimeWarning,
            stacklevel=2,
        )

    # un schema intra-session ne predit pas forcement tous les essais si une
    # session manque une classe : on n'evalue que ce qui a ete predit
    y_true = labels[predicted]
    y_pred = predictions[predicted].astype(object)
    groups_eval = groups_c[predicted]

    classes = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    acc = accuracy(y_true, y_pred)
    pvalue, _ = permutation_pvalue(y_true, y_pred, groups_eval, n_perm=n_perm, seed=seed)
    low, high = binomial_ci(int(round(acc * len(y_true))), len(y_true))

    from .stats import confusion_matrix

    return Result(
        model=model,
        scheme=scheme,
        contrast=contrast,
        recentered=recenter,
        n_trials=len(y_true),
        n_classes=len(np.unique(y_true)),
        chance=chance,
        accuracy=acc,
        macro_f1=macro_f1(y_true, y_pred, classes),
        ci_low=low,
        ci_high=high,
        pvalue=pvalue,
        mde=minimum_detectable_accuracy(len(y_true), chance),
        unreachable=unreachable,
        labels=classes,
        confusion=confusion_matrix(y_true, y_pred, classes),
        y_true=y_true,
        y_pred=y_pred,
    )
