"""Filtrage, reference et rejet d'essais.

Aucune de ces operations n'utilise les etiquettes, elles peuvent donc etre
appliquees avant le decoupage en plis sans creer de fuite. Les operations qui
en utiliseraient (mise a l'echelle par classe, selection de canaux guidee par la
performance) n'ont pas leur place ici : elles appartiennent a un ``Pipeline``
sklearn ajuste sur le pli d'entrainement.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

LINE_FREQ = 50.0


def notch(X: np.ndarray, sfreq: float, freq: float = LINE_FREQ, quality: float = 30.0,
          n_harmonics: int = 4) -> np.ndarray:
    """Retire le secteur et ses harmoniques sous la frequence de Nyquist."""
    X = np.asarray(X, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        f0 = freq * k
        if f0 >= 0.45 * sfreq:
            break
        b, a = iirnotch(f0, quality, sfreq)
        X = filtfilt(b, a, X, axis=-1)
    return X


def bandpass(X: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Passe-bande Butterworth a phase nulle."""
    nyq = 0.5 * sfreq
    high = min(high, 0.98 * nyq)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, np.asarray(X, dtype=np.float64), axis=-1)


def common_average_reference(X: np.ndarray) -> np.ndarray:
    """Reference moyenne commune, calculee canal par canal sur chaque essai."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=-2, keepdims=True)


def crop(X: np.ndarray, times: np.ndarray, tmin: float, tmax: float
         ) -> tuple[np.ndarray, np.ndarray]:
    """Restreint les epoques a [tmin, tmax)."""
    mask = (times >= tmin) & (times < tmax)
    return X[..., mask], times[mask]


def reject_trials(X: np.ndarray, *, amplitude_uv: float = 200.0,
                  mad_threshold: float = 6.0) -> np.ndarray:
    """Masque booleen des essais a conserver.

    Deux criteres, tous deux sans etiquette : une amplitude crete-a-crete
    absolue, et un ecart robuste par rapport a la session (mediane + k * MAD de
    l'amplitude crete-a-crete maximale par essai). Le second attrape les derives
    qui restent sous le seuil absolu tout en etant aberrantes pour la session.
    """
    X = np.asarray(X, dtype=np.float64)
    ptp = np.ptp(X, axis=-1).max(axis=-1)          # (n_trials,) pire canal
    keep = ptp < amplitude_uv

    median = np.median(ptp)
    mad = np.median(np.abs(ptp - median)) * 1.4826
    if mad > 0:
        keep &= ptp < median + mad_threshold * mad
    return keep


def prepare(X: np.ndarray, sfreq: float, *, low: float = 0.5, high: float = 45.0,
            do_notch: bool = True, car: bool = True) -> np.ndarray:
    """Chaine standard : secteur, passe-bande, reference moyenne commune."""
    if do_notch:
        X = notch(X, sfreq)
    X = bandpass(X, sfreq, low, high)
    if car:
        X = common_average_reference(X)
    return np.ascontiguousarray(X, dtype=np.float64)
