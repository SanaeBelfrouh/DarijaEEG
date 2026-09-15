"""L'echelle de modeles, du plus simple au plus expressif.

L'ordre est volontaire. ``puissance_lda`` reproduit la ligne de base actuelle ;
tout ce qui suit doit etre lu comme un gain PAR RAPPORT a elle, sur exactement
les memes plis. Une methode qui ne bat pas la ligne de base sur les memes plis
n'apporte rien, quelle que soit sa sophistication.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    FILTER_BANK,
    MulticlassCSP,
    band_power,
    bank_augmented_covariances,
    bank_covariances,
)
from .riemann import (
    covariances,
    distance_riemann,
    geometric_mean,
    tangent_vectors,
    TangentSpace,
)


class BankTangentSpace(BaseEstimator, TransformerMixin):
    """Espace tangent bande par bande, puis concatenation.

    Une reference geometrique par bande, toutes ajustees sur le pli
    d'entrainement. Concatener dans l'espace tangent plutot que construire une
    grande matrice bloc-diagonale donne la meme information sans avoir a
    inverser une matrice de taille (n_bandes x n_canaux)."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)       # (n_trials, n_bands, c, c)
        self.references_ = [geometric_mean(X[:, b]) for b in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate(
            [tangent_vectors(X[:, b], ref) for b, ref in enumerate(self.references_)], axis=1
        )


class MDM(BaseEstimator, ClassifierMixin):
    """Distance minimale a la moyenne riemannienne.

    Sans aucun hyperparametre a regler et avec un seul point par classe, ce
    classifieur ne peut pratiquement pas sur-apprendre. C'est pour cette raison
    qu'il vaut la peine d'etre rapporte meme s'il est moins performant : s'il
    est au hasard alors que la regression logistique tangente ne l'est pas, la
    difference vient probablement de la capacite du modele, pas du signal."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([geometric_mean(X[np.asarray(y) == c]) for c in self.classes_])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        distances = np.stack(
            [[distance_riemann(m, x) for m in self.means_] for x in X]
        )
        return self.classes_[np.argmin(distances, axis=1)]


def _logreg(C: float = 0.1) -> LogisticRegression:
    """Regression logistique multinomiale fortement regularisee.

    Avec 2080 dimensions tangentes pour 1575 essais d'entrainement, la
    regularisation n'est pas un reglage fin : sans elle le probleme est
    sous-determine et la solution est arbitraire."""
    # depuis scikit-learn 1.7 le multinomial est le comportement par defaut de
    # lbfgs et le parametre multi_class a ete retire
    return LogisticRegression(C=C, max_iter=3000, solver="lbfgs")


# --------------------------------------------------------------------------
# representations : calculees une fois, hors validation croisee
# --------------------------------------------------------------------------

def build_representations(X: np.ndarray, sfreq: float, *, bands=FILTER_BANK,
                          include: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Calcule toutes les representations une seule fois.

    Le filtrage et l'estimation des covariances ne dependent d'aucune etiquette
    et sont identiques pour tous les essais : les sortir de la boucle de
    validation croisee ne cree pas de fuite et divise le temps de calcul par le
    nombre de plis.
    """
    wanted = set(include) if include else None

    def want(name):
        return wanted is None or name in wanted

    out: dict[str, np.ndarray] = {}
    if want("brut"):
        out["brut"] = np.asarray(X, dtype=np.float64)
    if want("puissance"):
        out["puissance"] = band_power(X, sfreq, bands)
    if want("cov"):
        out["cov"] = covariances(X)
    if want("bankcov"):
        bank = bank_covariances(X, sfreq, bands)
        out["bankcov"] = np.stack([bank[b] for b in bands], axis=1)
    if want("augcov"):
        # plongement par retards : reintroduit la dynamique que la covariance
        # spatiale ordinaire ignore. Limite a 3 bandes basses, ou la dynamique
        # lente de la parole imaginee est attendue, pour rester tractable.
        low_bands = tuple(b for b in bands if b[1] <= 13.0)
        bank = bank_augmented_covariances(X, sfreq, low_bands, order=3, lag=8)
        out["augcov"] = np.stack([bank[b] for b in low_bands], axis=1)
    return out


#: nom du modele -> (representation requise, fabrique du pipeline)
MODELS: dict[str, tuple[str, callable]] = {
    # ligne de base : ce que fait le pipeline actuel
    "puissance_lda": (
        "puissance",
        lambda: Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    ),
    "puissance_logreg": (
        "puissance",
        lambda: Pipeline([("scale", StandardScaler()), ("clf", _logreg(0.1))]),
    ),
    # spatial classique
    "csp_lda": (
        "brut",
        lambda: Pipeline([
            ("csp", MulticlassCSP(n_components=2)),
            ("scale", StandardScaler()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]),
    ),
    # riemannien
    "cov_mdm": ("cov", lambda: MDM()),
    "cov_tangent": (
        "cov",
        lambda: Pipeline([
            ("ts", TangentSpace()),
            ("scale", StandardScaler()),
            ("clf", _logreg(0.1)),
        ]),
    ),
    "bankcov_tangent": (
        "bankcov",
        lambda: Pipeline([
            ("ts", BankTangentSpace()),
            ("scale", StandardScaler()),
            ("clf", _logreg(0.05)),
        ]),
    ),
    "augcov_tangent": (
        "augcov",
        lambda: Pipeline([
            ("ts", BankTangentSpace()),
            ("scale", StandardScaler()),
            ("clf", _logreg(0.02)),
        ]),
    ),
}


def make_model(name: str):
    """Instancie un modele de l'echelle."""
    if name not in MODELS:
        raise ValueError(f"modele inconnu : {name} (disponibles : {sorted(MODELS)})")
    return MODELS[name][1]()


def representation_for(name: str) -> str:
    return MODELS[name][0]
