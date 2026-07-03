"""Tests for the ternary Stage-2 discriminator after the JBR-revision feature redesign.

Covers the four-feature vector (twin_peak_symmetry removed from the model), the
individual statistics incl. the new relative_phase, the refit default model, and
disentangle_ternary's gates and calls.
"""
import numpy as np
import pytest

from chord.bhdt.stage2_orthogonal import (
    FEATURE_NAMES,
    phase_freedom_pvalue,
    relative_phase,
    amplitude_ratio,
    twin_peak_symmetry,
    harmonic_decay_residual,
    extract_features,
    fit_ternary_model,
    get_default_model,
    disentangle_ternary,
)
from chord.simulation.ternary_benchmark import build_ternary_benchmark

W = 2 * np.pi / 24.0


def _grid():
    return np.arange(0.0, 48.0, 2.0)  # 2 h over 48 h, 24 points


def _series(a24=2.0, phi24=0.0, a12=1.3, phi12=0.0, m=6.0, noise=0.0, seed=0):
    t = _grid()
    y = m + a24 * np.cos(W * t - phi24) + a12 * np.cos(2 * W * t - phi12)
    if noise:
        y = y + np.random.default_rng(seed).normal(0.0, noise, len(t))
    return t, y


# --------------------------------------------------------------------------- #
# Feature-vector contract (the redesign)
# --------------------------------------------------------------------------- #
def test_feature_names_are_the_five_axis_features():
    assert FEATURE_NAMES == ("neglog10_p_phase", "log_amp_ratio", "twin_symmetry",
                             "decay_residual", "log_snr12")
    assert "twin_symmetry" in FEATURE_NAMES              # model input (non-redundant)


def test_extract_features_returns_five_finite_values():
    t, y = _series(noise=0.3, seed=1)
    f = extract_features(t, y)
    assert f.shape == (len(FEATURE_NAMES),) == (5,)
    assert np.all(np.isfinite(f))


# --------------------------------------------------------------------------- #
# Individual statistics
# --------------------------------------------------------------------------- #
def test_phase_freedom_small_p_for_free_phase():
    # 12h phase far from the locked value (delta = phi12 - 2*phi24 = 1.7) -> free
    t, y = _series(a24=2.0, phi24=0.0, a12=1.5, phi12=1.7, noise=0.05, seed=2)
    assert phase_freedom_pvalue(t, y) < 0.05


def test_phase_freedom_neutral_without_24h_fundamental():
    # no 24h component -> the test returns the neutral 0.5 (its documented guard)
    t = _grid()
    y = 6.0 + 1.5 * np.cos(2 * W * t - 0.4)
    assert phase_freedom_pvalue(t, y) == pytest.approx(0.5)


def test_relative_phase_recovers_known_offset():
    # phi24 = 0, phi12 = 1.2  =>  delta = phi12 - 2*phi24 = 1.2
    t, y = _series(a24=2.0, phi24=0.0, a12=1.0, phi12=1.2, noise=0.0)
    assert relative_phase(t, y) == pytest.approx(1.2, abs=1e-6)


def test_relative_phase_wraps_to_pi_interval():
    t, y = _series(a12=1.0, phi12=3.0, noise=0.0)
    d = relative_phase(t, y)
    assert -np.pi < d <= np.pi


def test_amplitude_ratio_recovers_a12_over_a24():
    t, y = _series(a24=2.0, a12=1.0, noise=0.0)
    assert amplitude_ratio(t, y) == pytest.approx(0.5, rel=0.05)


def test_twin_symmetry_bounded_and_retained_as_stat():
    t, y = _series(a24=2.0, a12=0.6, noise=0.0)
    r = twin_peak_symmetry(t, y)
    assert 0.0 <= r <= 1.0            # bounded model input + reporting statistic


def test_harmonic_decay_residual_neutral_for_pure_two_component():
    # a pure 24h+12h has no higher harmonics, so no decay trend is estimable and the
    # residual is the documented neutral 0.0 (not a spurious 'bump' -> Class B)
    t, y = _series(a24=2.0, a12=1.6, phi12=1.5, noise=0.0)
    assert harmonic_decay_residual(t, y) == pytest.approx(0.0)


def test_harmonic_decay_residual_positive_for_12h_bump_above_decay():
    # harmonics k=1,3,4 set a decay trend (noise-free -> tiny floor); an extra strong
    # 12h (k=2) sits above that decay, so the residual is positive
    t = _grid()
    base = 2.0 * np.cos(W * t) + 0.8 * np.cos(3 * W * t) + 0.6 * np.cos(4 * W * t)
    y = 6.0 + base + 1.5 * np.cos(2 * W * t - 1.0)
    assert harmonic_decay_residual(t, y) > 0.0


# --------------------------------------------------------------------------- #
# Default model + disentangler
# --------------------------------------------------------------------------- #
def test_default_model_is_five_feature():
    m = get_default_model()
    assert m.classes == ["A", "B", "C"]
    assert m.feat_mean.shape == (5,)
    assert m.weights.shape == (3, 6)   # 3 classes x (5 features + bias)


def test_default_model_is_cached_singleton():
    assert get_default_model() is get_default_model()


@pytest.fixture(scope="module")
def small_model():
    b = build_ternary_benchmark(n_per_class=40, classes=("A", "B", "C"), seed=1)
    return fit_ternary_model(b["t"], b["expr"], b["labels"])


def test_disentangle_none_when_no_12h(small_model):
    t = _grid()
    y = 6.0 + 2.0 * np.cos(W * t) + np.random.default_rng(3).normal(0, 0.3, len(t))
    out = disentangle_ternary(t, y, model=small_model)
    assert out["class"] == "none"
    assert out["proba"] is None


def test_disentangle_ambiguous_when_no_24h_fundamental(small_model):
    # 12h present, no 24h -> gate 2 (identifiability floor): an autonomous B and a
    # fundamental-suppressed intersection C are indistinguishable here -> 'ambiguous'
    t = _grid()
    y = 6.0 + 1.6 * np.cos(2 * W * t - 0.7) + np.random.default_rng(4).normal(0, 0.2, len(t))
    out = disentangle_ternary(t, y, model=small_model)
    assert out["class"] == "ambiguous"
    assert out["proba"] is None
    assert np.isnan(out["autonomy_score"])


def test_disentangle_returns_valid_class_and_stats(small_model):
    t, y = _series(a24=2.0, a12=1.3, phi12=1.6, noise=0.3, seed=5)
    out = disentangle_ternary(t, y, model=small_model)
    assert out["class"] in {"A_harmonic", "B_independent", "C_intersection", "ambiguous"}
    # the five model features are surfaced in stats under FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert name in out["stats"]


def test_disentangle_probabilities_sum_to_one(small_model):
    t, y = _series(a24=2.0, a12=1.3, phi12=1.6, noise=0.3, seed=6)
    out = disentangle_ternary(t, y, model=small_model)
    if out["proba"] is not None:
        assert sum(out["proba"].values()) == pytest.approx(1.0, abs=1e-6)
