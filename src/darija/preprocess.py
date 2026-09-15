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


def reject_trials(X: np.ndarray, *, amplitude_uv: float = 150.0,
                  max_bad_fraction: float = 0.15,
                  mad_threshold: float = 6.0) -> tuple[np.ndarray, dict]:
    """Masque booleen des essais a conserver, plus un rapport de diagnostic.

    Le critere porte sur la PROPORTION de canaux hors bornes, pas sur le pire
    canal. Exiger que les 64 canaux restent sous un seuil revient a exiger
    qu'aucun clignement, aucune electrode bruyante et aucun mouvement n'ait
    touche une seule derivation pendant 2,5 s : sur un montage 64 canaux avec
    fixation visuelle, cela rejette la quasi-totalite des essais. Un essai est
    exploitable si l'immense majorite de ses canaux est propre.

    Le second critere est robuste et relatif a la session : mediane + k * MAD de
    l'amplitude mediane par essai. Il attrape les derives qui restent sous le
    seuil absolu tout en etant aberrantes pour cette session, et il ne depend
    d'aucune hypothese sur les unites du flux.
    """
    X = np.asarray(X, dtype=np.float64)
    ptp = np.ptp(X, axis=-1)                        # (n_trials, n_channels)

    bad_fraction = (ptp > amplitude_uv).mean(axis=1)
    keep = bad_fraction <= max_bad_fraction

    trial_level = np.median(ptp, axis=1)            # amplitude typique de l'essai
    median = np.median(trial_level)
    mad = np.median(np.abs(trial_level - median)) * 1.4826
    if mad > 0:
        keep &= trial_level < median + mad_threshold * mad

    report = {
        "ptp_median_uv": float(median),
        "ptp_p05_uv": float(np.percentile(trial_level, 5)),
        "ptp_p95_uv": float(np.percentile(trial_level, 95)),
        "ptp_worst_channel_median_uv": float(np.median(ptp.max(axis=1))),
        "rejected_by_amplitude": int(np.sum(bad_fraction > max_bad_fraction)),
        "rejected_total": int(np.sum(~keep)),
        "n_trials": int(len(keep)),
    }
    return keep, report


def looks_like_microvolts(ptp_median: float) -> bool:
    """L'amplitude mediane est-elle compatible avec de l'EEG exprime en uV ?

    Apres un passe-bande 0,5-45 Hz, un essai d'EEG de scalp a typiquement une
    amplitude crete-a-crete de quelques dizaines de uV. Une mediane a 1e-5 ou a
    1e6 signale que le flux n'est pas en microvolts, et rend tout seuil absolu
    exprime en uV denue de sens.
    """
    return 1.0 <= ptp_median <= 500.0


def filter_continuous(data: np.ndarray, sfreq: float, *, low: float = 0.5,
                     high: float = 45.0, do_notch: bool = True,
                     chunk: int = 8) -> np.ndarray:
    """Secteur et passe-bande sur l'enregistrement CONTINU, avant decoupage.

    C'est l'ordre correct, et il n'est pas interchangeable avec l'autre. Filtrer
    chaque epoque separement fait apparaitre des transitoires de bord : un
    passe-haut a 0,5 Hz a une constante de temps de l'ordre de 0,3 s, et un
    amplificateur couple en continu comme l'actiCHamp presente des decalages de
    ligne de base de plusieurs millivolts. Sur une epoque de 2,5 s, ces
    transitoires dominent l'amplitude crete-a-crete et font rejeter des essais
    parfaitement propres.

    Le filtrage se fait par paquets de canaux pour borner la memoire : 64 canaux
    x 1,5 million d'echantillons en double precision font 768 Mo.
    """
    data = np.asarray(data)
    out = np.empty_like(data, dtype=np.float32)
    for start in range(0, data.shape[0], chunk):
        block = np.asarray(data[start:start + chunk], dtype=np.float64)
        if do_notch:
            block = notch(block, sfreq)
        block = bandpass(block, sfreq, low, high)
        out[start:start + chunk] = block.astype(np.float32)
    return out


def detrend_epochs(X: np.ndarray) -> np.ndarray:
    """Retire une tendance lineaire par epoque et par canal.

    Garde-fou bon marche contre ce qui subsiste de derive lente apres le
    passe-bande, notamment les rampes de sueur et les derives d'impedance.
    """
    from scipy.signal import detrend

    return detrend(np.asarray(X, dtype=np.float64), axis=-1, type="linear")


def prepare(X: np.ndarray, sfreq: float, *, car: bool = True,
            detrend: bool = True) -> np.ndarray:
    """Finition au niveau de l'epoque : detendance puis reference moyenne.

    Le filtrage n'est deliberement PAS fait ici : il appartient au signal
    continu (voir :func:`filter_continuous`). La reference moyenne commune est
    en revanche une operation point de temps par point de temps, donc identique
    qu'on l'applique avant ou apres le decoupage.
    """
    X = np.asarray(X, dtype=np.float64)
    if detrend:
        X = detrend_epochs(X)
    if car:
        X = common_average_reference(X)
    return np.ascontiguousarray(X, dtype=np.float64)
