"""Representations : banc de filtres, covariances par bande, CSP, puissance.

Les fonctions qui produisent des covariances renvoient un dictionnaire bande ->
tableau de covariances. Les concatener dans l'espace tangent (une reference par
bande) est prefere a une grande matrice bloc-diagonale : le resultat est
identique en information et evite d'inverser une matrice 320 x 320.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin

from .preprocess import bandpass
from .riemann import augmented_covariances, covariances

#: banc par defaut. 30-45 Hz s'arrete avant le secteur ; au-dela, la
#: caracterisation haute frequence deja menee ne montre rien en imagine.
FILTER_BANK = (
    (0.5, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 20.0),  # beta bas
    (20.0, 30.0),  # beta haut
    (30.0, 45.0),  # gamma bas
)


def filter_bank(X: np.ndarray, sfreq: float, bands=FILTER_BANK) -> dict[tuple, np.ndarray]:
    """Applique chaque bande du banc et renvoie les signaux filtres."""
    return {band: bandpass(X, sfreq, band[0], band[1]) for band in bands}


def bank_covariances(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
                     estimator: str = "oas") -> dict[tuple, np.ndarray]:
    """Une matrice de covariance par bande et par essai."""
    return {band: covariances(sig, estimator=estimator)
            for band, sig in filter_bank(X, sfreq, bands).items()}


def bank_augmented_covariances(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
                               order: int = 3, lag: int = 8,
                               estimator: str = "oas") -> dict[tuple, np.ndarray]:
    """Covariances augmentees par retards, bande par bande."""
    return {band: augmented_covariances(sig, order=order, lag=lag, estimator=estimator)
            for band, sig in filter_bank(X, sfreq, bands).items()}


def band_power(X: np.ndarray, sfreq: float, bands=FILTER_BANK,
               n_windows: int = 4, log: bool = True) -> np.ndarray:
    """Puissance d'enveloppe par canal, bande et sous-fenetre temporelle.

    C'est la representation de la ligne de base actuelle, conservee ici comme
    point de comparaison : tout gain rapporte pour une methode riemannienne doit
    etre lu par rapport a elle, sur exactement les memes plis.
    """
    feats = []
    for band, sig in filter_bank(X, sfreq, bands).items():
        env = np.abs(hilbert(sig, axis=-1)) ** 2
        chunks = np.array_split(np.arange(env.shape[-1]), n_windows)
        feats.append(np.concatenate([env[..., c].mean(axis=-1) for c in chunks], axis=-1))
    out = np.concatenate(feats, axis=-1)
    return np.log(out + 1e-12) if log else out


# --------------------------------------------------------------------------
# CSP multiclasse un-contre-tous
# --------------------------------------------------------------------------

class MulticlassCSP(BaseEstimator, TransformerMixin):
    """CSP un-contre-tous, ``n_components`` filtres par classe.

    Les filtres sont appris par decomposition simultanee de la covariance de la
    classe et de celle du reste. Avec 15 classes et ``n_components=2`` cela fait
    30 filtres ; au-dela le nombre de parametres devient comparable au nombre
    d'essais par classe et le sur-apprentissage domine.
    """

    def __init__(self, n_components: int = 2, log: bool = True):
        self.n_components = n_components
        self.log = log

    def fit(self, X, y):
        covs = covariances(X)
        self.classes_ = np.unique(y)
        filters = []
        for cls in self.classes_:
            mask = np.asarray(y) == cls
            c1 = covs[mask].mean(axis=0)
            c2 = covs[~mask].mean(axis=0)
            w, V = np.linalg.eigh(np.linalg.solve(c1 + c2, c1))
            order = np.argsort(w)[::-1]
            V = V[:, order]
            # extremes du spectre : variance maximale pour la classe, puis minimale
            filters.append(np.concatenate(
                [V[:, :self.n_components], V[:, -self.n_components:]], axis=1))
        self.filters_ = np.concatenate(filters, axis=1).T   # (n_filters, n_channels)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        projected = self.filters_ @ X                       # (n_trials, n_filters, n_times)
        var = projected.var(axis=-1)
        var = var / (var.sum(axis=-1, keepdims=True) + 1e-12)
        return np.log(var + 1e-12) if self.log else var


class CovariancesTransformer(BaseEstimator, TransformerMixin):
    """Enveloppe sklearn autour de :func:`darija.riemann.covariances`."""

    def __init__(self, estimator: str = "oas"):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return covariances(X, estimator=self.estimator)
