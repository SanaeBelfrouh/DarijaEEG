"""Tests du pipeline : il doit voir un effet planté, et ne pas en inventer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from darija import cv, riemann, stats                      # noqa: E402
from darija.evaluate import evaluate                       # noqa: E402
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
    assert result.accuracy > result.chance + 0.10, result.accuracy
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
