"""Inference : permutations groupees, intervalles, puissance statistique.

La question de la puissance est ici plus importante que le choix du test. Avec
1800 essais imagines et un hasard a 6,67 %, la plus petite exactitude qu'on
puisse declarer differente du hasard tourne autour de 8 %. Autrement dit : meme
un effet reel de 1,5 point serait indetectable. Il faut le savoir AVANT
d'interpreter un resultat non significatif comme une absence d'effet.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def permutation_pvalue(y_true, y_pred, groups, n_perm: int = 5000,
                       seed: int = 0) -> tuple[float, np.ndarray]:
    """p par permutation des etiquettes A L'INTERIEUR de chaque session.

    Les predictions sont figees ; seules les etiquettes bougent. Permuter
    librement sur les 1800 essais supposerait qu'ils sont independants, alors
    qu'ils proviennent de 8 sessions dont chacune a sa propre derive : la
    distribution nulle serait trop etroite et le p trop optimiste. Permuter dans
    la session preserve la structure de groupe.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    observed = accuracy(y_true, y_pred)

    rng = np.random.default_rng(seed)
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    permuted = y_true.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        for block in blocks:
            permuted[block] = rng.permutation(y_true[block])
        null[i] = np.mean(permuted == y_pred)

    pvalue = (1.0 + np.sum(null >= observed)) / (n_perm + 1.0)
    return float(pvalue), null


def binomial_ci(n_correct: int, n_total: int, alpha: float = 0.05) -> tuple[float, float]:
    """Intervalle de Wilson, preferable a Wald pres du hasard et aux petits n."""
    if n_total == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - alpha / 2)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(centre - half), float(centre + half)


def minimum_detectable_accuracy(n: int, chance: float, power: float = 0.80,
                                alpha: float = 0.05) -> float:
    """Plus petite exactitude vraie detectable avec la puissance demandee.

    Test binomial unilateral, approximation normale. Sert a repondre a la
    question qui precede toute interpretation d'un resultat nul : cette
    experience avait-elle seulement les moyens de voir l'effet recherche ?
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    se0 = np.sqrt(chance * (1 - chance) / n)

    # resolution par point fixe : l'ecart-type sous H1 depend de la cible
    p = chance + z_alpha * se0
    for _ in range(100):
        se1 = np.sqrt(p * (1 - p) / n)
        new = chance + (z_alpha * se0 + z_beta * se1)
        if abs(new - p) < 1e-10:
            break
        p = new
    return float(min(p, 1.0))


def required_trials(effect: float, chance: float, power: float = 0.80,
                    alpha: float = 0.05) -> int:
    """Nombre d'essais necessaires pour detecter une exactitude vraie ``effect``."""
    if effect <= chance:
        return -1
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    num = z_alpha * np.sqrt(chance * (1 - chance)) + z_beta * np.sqrt(effect * (1 - effect))
    return int(np.ceil((num / (effect - chance)) ** 2))


def holm_bonferroni(pvalues) -> np.ndarray:
    """Correction de Holm : uniformement plus puissante que Bonferroni simple."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def confusion_matrix(y_true, y_pred, labels) -> np.ndarray:
    """Matrice de confusion brute, lignes = verite, colonnes = prediction."""
    labels = list(labels)
    index = {lab: i for i, lab in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[index[t], index[p]] += 1
    return M


def macro_f1(y_true, y_pred, labels) -> float:
    """F1 moyenne par classe : moins trompeuse que l'exactitude si le
    classifieur s'effondre sur quelques classes majoritaires."""
    M = confusion_matrix(y_true, y_pred, labels)
    tp = np.diag(M).astype(float)
    precision = np.divide(tp, M.sum(axis=0), out=np.zeros_like(tp), where=M.sum(axis=0) > 0)
    recall = np.divide(tp, M.sum(axis=1), out=np.zeros_like(tp), where=M.sum(axis=1) > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())
