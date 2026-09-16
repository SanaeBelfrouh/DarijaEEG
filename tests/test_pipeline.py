"""Tests du pipeline : il doit voir un effet planté, et ne pas en inventer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from darija import cv, riemann, stats                      # noqa: E402
from darija.preprocess import (                            # noqa: E402
    bandpass,
    filter_continuous,
    looks_like_microvolts,
    reject_trials,
)
from darija.evaluate import evaluate, evaluate_many        # noqa: E402
from darija.pipelines import build_representations         # noqa: E402
from synthetic import make_dataset                         # noqa: E402


def test_riemann_identities():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 40))
    C = A @ A.T / 40 + np.eye(8)

    assert np.allclose(riemann.sqrtm(C) @ riemann.sqrtm(C), C, atol=1e-8)
    assert np.allclose(riemann.invsqrtm(C) @ riemann.sqrtm(C), np.eye(8), atol=1e-8)
    assert np.allclose(riemann.expm(riemann.logm(C)), C, atol=1e-8)
    assert riemann.distance_riemann(C, C) < 1e-8

    # la moyenne geometrique d'une matrice repetee est cette matrice
    assert np.allclose(riemann.geometric_mean(np.stack([C] * 5)), C, atol=1e-6)


def test_tangent_space_is_isometric_at_reference():
    """La norme du vecteur tangent doit egaler la distance riemannienne a la
    reference : c'est ce qui rend un classifieur lineaire tangent sense."""
    rng = np.random.default_rng(1)
    covs = riemann.covariances(rng.normal(size=(6, 8, 200)))
    ref = riemann.geometric_mean(covs)
    vectors = riemann.tangent_vectors(covs, ref)
    for v, c in zip(vectors, covs):
        assert abs(np.linalg.norm(v) - riemann.distance_riemann(ref, c)) < 1e-6


def test_recentering_makes_session_means_identity():
    rng = np.random.default_rng(2)
    covs = riemann.covariances(rng.normal(size=(40, 8, 200)))
    groups = np.repeat(["a", "b"], 20)
    covs[groups == "b"] *= 7.0            # deplacement de domaine grossier

    out = riemann.recenter_by_group(covs, groups)
    for g in ("a", "b"):
        mean = riemann.geometric_mean(out[groups == g])
        assert np.allclose(mean, np.eye(8), atol=1e-4)


def test_cv_splits_are_disjoint_and_grouped():
    groups = np.repeat(["S1", "S2", "S3"], 20)
    order = np.tile(np.arange(20), 3)

    for train, test in cv.leave_one_session_out(groups):
        assert not set(train) & set(test)
        assert len(np.unique(groups[test])) == 1

    for train, test in cv.within_session_blocked(groups, order, n_splits=4):
        assert not set(train) & set(test)
        # entrainement et test viennent de la meme session
        assert set(groups[train]) == set(groups[test])
        # les blocs sont contigus dans le temps : le test ne chevauche pas
        assert np.ptp(order[test]) < 20

    for train, test in cv.pooled_blocked(groups, order, n_splits=4):
        assert not set(train) & set(test)
        assert len(np.unique(groups[test])) == 3


def test_power_helpers_are_consistent():
    n = stats.required_trials(0.10, 1 / 15)
    mde = stats.minimum_detectable_accuracy(n, 1 / 15)
    assert abs(mde - 0.10) < 0.005, mde
    # plus d'essais, seuil plus bas
    assert stats.minimum_detectable_accuracy(4000, 1 / 15) < \
        stats.minimum_detectable_accuracy(1000, 1 / 15)


def test_no_signal_stays_at_chance():
    """Le controle negatif : sans effet plante, le pipeline ne doit rien trouver.

    Le test porte sur la MOYENNE de plusieurs graines, pas sur une seule : un
    test a alpha = 5 % ressort une fois sur vingt par construction, et faire
    echouer la suite la-dessus serait exiger du pipeline qu'il soit anti-
    conservateur. Ce qui doit tenir, c'est que l'exactitude moyenne reste au
    niveau du hasard."""
    accuracies = []
    for seed in range(5):
        X, y, groups, order, sfreq = make_dataset(
            n_words=4, n_per_word=20, n_channels=8, n_times=200, effect=0.0, seed=seed
        )
        reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
        result = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                          contrast="mots15", n_perm=200, seed=0)
        accuracies.append(result.accuracy)

    chance = 0.25
    n = 4 * 20 * 4
    tolerance = 3.0 * np.sqrt(chance * (1 - chance) / n) / np.sqrt(len(accuracies))
    assert abs(np.mean(accuracies) - chance) < tolerance, \
        f"exactitude moyenne {np.mean(accuracies):.4f} loin du hasard {chance}"


def test_planted_signal_is_recovered():
    """Le controle positif : avec un effet, le pipeline doit le voir."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=4, n_per_word=25, n_channels=8, n_times=200, effect=0.9, seed=4
    )
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
    result = evaluate(reps, y, groups, order, model="cov_tangent",
                      scheme="loso", contrast="mots15", n_perm=400, seed=0)
    assert result.balanced_accuracy > result.chance_balanced + 0.10, result.balanced_accuracy
    assert result.pvalue < 0.01, result.pvalue


def test_recentering_rescues_a_session_shift():
    """Un deplacement de domaine peut ramener au hasard un signal bien present ;
    le recentrage doit le recuperer. C'est l'argument central du rapport."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=4, n_per_word=25, n_channels=8, n_times=200,
        effect=0.35, session_shift=1.2, seed=5
    )
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))
    plain = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                     n_perm=300, seed=0)
    centred = evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                       recenter=True, n_perm=300, seed=0)
    gain = centred.accuracy - plain.accuracy
    assert gain > 0.03, f"gain du recentrage trop faible : {gain:.4f}"


def _realistic_64ch(rng, n_trials=200, n_channels=64, n_times=625):
    """EEG de scalp plausible : ~35 uV, quelques electrodes bruyantes, clignements."""
    X = rng.normal(0, 6.0, size=(n_trials, n_channels, n_times))
    X[:, :4] *= 6.0                                    # electrodes a haute impedance
    t = np.arange(n_times)
    for i in np.flatnonzero(rng.random(n_trials) < 0.35):
        peak = np.exp(-((t - rng.integers(100, 500)) ** 2) / (2 * 25.0 ** 2))
        X[i, :6] += peak * rng.uniform(150, 400)       # clignement sur les frontales
    return X


def test_rejection_keeps_usable_64_channel_trials():
    """Non-regression : un critere portant sur le PIRE des 64 canaux rejette tout.

    Sur un montage 64 canaux avec fixation visuelle, quelques derivations
    frontales depassent 200 uV crete-a-crete des qu'il y a un clignement, et une
    ou deux electrodes sont toujours plus bruyantes que les autres. Exiger que
    les 64 restent sous le seuil revient a exiger un enregistrement parfait
    pendant 2,5 s : le taux de rejet atteint 100 % sur des donnees parfaitement
    exploitables. Le critere doit porter sur la PROPORTION de canaux atteints."""
    rng = np.random.default_rng(0)
    X = _realistic_64ch(rng)

    worst_channel = np.ptp(X, axis=-1).max(axis=1)
    assert (worst_channel < 200.0).sum() == 0, "le scenario ne reproduit pas le bug"

    keep, report = reject_trials(X, amplitude_uv=150.0, max_bad_fraction=0.15)
    assert keep.mean() > 0.9, f"{keep.mean():.2%} conserves, {report}"
    assert 5.0 < report["ptp_median_uv"] < 150.0, report


def test_rejection_catches_widespread_artifacts():
    """Le critere doit rester capable de rejeter ce qui est vraiment mauvais."""
    rng = np.random.default_rng(1)
    X = rng.normal(0, 6.0, size=(200, 64, 625))
    bad = rng.choice(200, 20, replace=False)
    X[bad] += rng.normal(0, 90.0, size=(20, 64, 1)) * np.linspace(0, 1, 625)

    keep, _ = reject_trials(X)
    assert not keep[bad].any(), "artefacts generalises non detectes"
    assert keep.sum() == 180, f"faux rejets : {180 - keep.sum()}"


def test_unit_check_flags_non_microvolt_streams():
    assert looks_like_microvolts(35.0)
    assert not looks_like_microvolts(3.5e-5)     # flux en volts
    assert not looks_like_microvolts(4.0e4)      # decalage continu non retire


def test_continuous_filtering_beats_per_epoch():
    """Le filtrage appartient au continu, pas a l'epoque.

    Un amplificateur couple en continu comme l'actiCHamp presente des decalages
    de ligne de base de plusieurs millivolts. Filtrer chaque epoque separement
    laisse des transitoires de bord qui gonflent l'amplitude crete-a-crete."""
    rng = np.random.default_rng(2)
    sfreq, n = 1000.0, 60000
    continuous = rng.normal(0, 6.0, size=(4, n)) + 40000.0     # decalage de 40 mV

    filtered = filter_continuous(continuous, sfreq, low=0.5, high=45.0, do_notch=False)
    starts = range(2000, 20000, 2500)
    first = np.stack([filtered[:, i:i + 2500] for i in starts])

    raw = np.stack([continuous[:, i:i + 2500] for i in starts])
    second = bandpass(raw, sfreq, 0.5, 45.0)

    assert np.median(np.ptp(first, axis=-1)) < np.median(np.ptp(second, axis=-1))


def test_balanced_accuracy_neutralises_class_collapse():
    """Non-regression : l'exactitude brute rendait significatif un effondrement.

    Avec des classes desequilibrees, un classifieur qui repond toujours la classe
    la plus RARE obtient une exactitude tres inferieure au taux de la classe
    majoritaire, tout en etant au-dessus de sa propre loi nulle par permutation
    — celle-ci valant somme_c P(pred=c) P(vrai=c), elle s'effondre avec lui. Le
    banc produisait alors des lignes du type "0,083 (hasard 0,266) p = 0,010".

    L'exactitude equilibree ramene un tel effondrement a exactement 1/k."""
    y_true = np.repeat(["a", "b", "c"], [100, 50, 10])
    collapsed = np.full(len(y_true), "c")

    raw = float(np.mean(collapsed == y_true))
    assert raw < 0.10                                   # tres en dessous de 100/160
    assert abs(stats.balanced_accuracy(y_true, collapsed) - 1 / 3) < 1e-9

    perfect = y_true.copy()
    assert stats.balanced_accuracy(y_true, perfect) == 1.0


def test_balanced_threshold_penalises_rare_classes():
    """Le seuil de detectabilite doit se degrader quand une classe est rare."""
    balanced = stats.minimum_detectable_balanced_accuracy([300] * 6)
    skewed = stats.minimum_detectable_balanced_accuracy([479, 359, 359, 240, 240, 120])
    assert skewed > balanced, (balanced, skewed)
    # a effectif egal, plus de classes => hasard plus bas
    assert stats.minimum_detectable_balanced_accuracy([120] * 15) < balanced


def test_shared_folds_match_independent_evaluation():
    """Mutualiser les plis entre contrastes ne doit rien changer aux resultats.

    Le prefixe independant des etiquettes est desormais ajuste une fois par pli
    et reutilise pour tous les contrastes. Ce test verifie que cette
    optimisation est exactement neutre."""
    X, y, groups, order, sfreq = make_dataset(
        n_words=15, n_per_word=6, n_channels=8, n_times=200, effect=0.5, seed=8
    )
    from darija import WORDS

    y = np.asarray(WORDS, dtype=object)[[int(w[1:]) for w in y]]
    reps = build_representations(X, sfreq, bands=((8.0, 13.0),), include=("cov",))

    together = evaluate_many(reps, y, groups, order, model="cov_tangent",
                             scheme="loso", contrasts=("mots15", "syllabes"), n_perm=100)
    separate = [
        evaluate(reps, y, groups, order, model="cov_tangent", scheme="loso",
                 contrast=name, n_perm=100)
        for name in ("mots15", "syllabes")
    ]
    for a, b in zip(together, separate):
        assert a.contrast == b.contrast
        assert np.array_equal(a.y_pred, b.y_pred), a.contrast


def test_trial_reconstruction_from_markers():
    """Reconstruction des essais a partir du seul flux de marqueurs.

    Reproduit la sequence exacte du protocole, avec l'ISI tire dans 0.8-1.2 s,
    et verifie que l'intervalle mesure comme 'isi' retrouve bien ce tirage tandis
    que 'production' reste exact. C'est la verification qui corrige le verdict du
    carnet de timing."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "timing_check",
        Path(__file__).resolve().parents[1] / "scripts" / "00_timing_check.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rng = np.random.default_rng(0)
    events, t = [], 100.0
    words = ["3afak", "lla", "Ah"]
    for i in range(60):
        word = words[i % len(words)]
        fixation = rng.uniform(1.0, 1.5)
        isi = rng.uniform(0.8, 1.2)
        for stage, duration in (
            ("fixation_start", 0.0), ("fixation_end", fixation),
            ("cue_start", 0.0), ("cue_end", 1.0),
            ("isi_start", 0.0), ("isi_end", isi),
            ("start", 0.0), ("end", 2.0),
            ("iti_start", 0.0), ("iti_end", 0.5),
        ):
            t += duration
            events.append(dict(cond="imagined", stage=stage,
                               word=word if "cue" in stage or stage in ("start", "end") else None,
                               t=t))

    trials = module.reconstruct(events, "S001")
    assert len(trials) == 60, len(trials)
    assert all(tr["word"] == w for tr, w in zip(trials, [words[i % 3] for i in range(60)]))

    isi_measured = np.array([tr["start"] - tr["cue_end"] for tr in trials])
    production = np.array([tr["end"] - tr["start"] for tr in trials])

    # l'ISI reproduit le tirage uniforme : ~115 ms d'ecart-type
    assert 0.8 <= isi_measured.min() and isi_measured.max() <= 1.2
    assert abs(isi_measured.std() - 0.4 / np.sqrt(12)) < 0.03, isi_measured.std()
    # la production est exacte : c'est elle qui aligne les epoques
    assert production.std() < 1e-9, production.std()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print("\nechecs :", failures)
    raise SystemExit(1 if failures else 0)
