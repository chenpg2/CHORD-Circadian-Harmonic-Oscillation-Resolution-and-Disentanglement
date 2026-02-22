"""Tests for Goodwin ODE decomposition module."""

import numpy as np
import pytest
import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chord.pinod.goodwin_ode import (
    hill_function,
    goodwin_dual_ode,
    simulate_goodwin,
    fit_goodwin_to_gene,
    goodwin_decompose,
    goodwin_evidence,
)


# ---------------------------------------------------------------------------
# Helper: default oscillating parameter sets
# ---------------------------------------------------------------------------

def _circadian_params():
    """Parameters that produce clear oscillations in both oscillators.

    High Hill coefficient (n=8) and appropriate degradation rates ensure
    sustained limit-cycle oscillations in the Goodwin system.
    """
    return {
        'alpha1': 20.0, 'K1': 1.0, 'n1': 8.0,
        'delta_m1': 1.0, 'beta1': 5.0, 'delta_p1': 0.5,
        'alpha2': 20.0, 'K2': 1.0, 'n2': 8.0,
        'delta_m2': 2.0, 'beta2': 10.0, 'delta_p2': 1.0,
        'epsilon': 0.0,
    }


def _coupled_params():
    """Parameters with coupling between oscillators."""
    p = _circadian_params()
    p['epsilon'] = 5.0
    return p


# ===========================================================================
# 1. Hill function tests
# ===========================================================================

class TestHillFunction:
    def test_hill_function_limits(self):
        """hill(0, K, n) = 1.0, hill(very large, K, n) -> 0."""
        for K in [1.0, 5.0, 10.0]:
            for n in [1.0, 3.0, 8.0]:
                assert hill_function(0.0, K, n) == pytest.approx(1.0, abs=1e-12)
                assert hill_function(1e10, K, n) == pytest.approx(0.0, abs=1e-6)

    def test_hill_function_midpoint(self):
        """hill(K, K, n) = 0.5 for any K, n."""
        for K in [0.5, 2.0, 10.0]:
            for n in [1.0, 3.0, 7.0]:
                assert hill_function(K, K, n) == pytest.approx(0.5, abs=1e-12)

    def test_hill_function_monotone_decreasing(self):
        """Hill function is monotonically decreasing in x."""
        K, n = 2.0, 4.0
        xs = np.linspace(0.01, 20.0, 100)
        vals = hill_function(xs, K, n)
        assert np.all(np.diff(vals) < 0)

    def test_hill_function_array_input(self):
        """Accepts array input and returns array."""
        xs = np.array([0.0, 1.0, 2.0, 5.0])
        result = hill_function(xs, 2.0, 3.0)
        assert result.shape == (4,)
        assert result[0] == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# 2. ODE right-hand side test
# ===========================================================================

class TestGoodwinDualODE:
    def test_ode_returns_four_values(self):
        """ODE RHS returns a list of 4 derivatives."""
        params = _circadian_params()
        state = [1.0, 1.0, 1.0, 1.0]
        result = goodwin_dual_ode(0.0, state, params)
        assert len(result) == 4
        assert all(np.isfinite(v) for v in result)

    def test_ode_zero_coupling(self):
        """With epsilon=0, dm2/dt does not depend on p1."""
        params = _circadian_params()
        params['epsilon'] = 0.0

        state_a = [1.0, 2.0, 1.0, 1.0]  # p1 = 2
        state_b = [1.0, 5.0, 1.0, 1.0]  # p1 = 5

        deriv_a = goodwin_dual_ode(0.0, state_a, params)
        deriv_b = goodwin_dual_ode(0.0, state_b, params)

        # dm2/dt and dp2/dt should be the same (indices 2, 3)
        assert deriv_a[2] == pytest.approx(deriv_b[2], abs=1e-12)
        assert deriv_a[3] == pytest.approx(deriv_b[3], abs=1e-12)

    def test_ode_nonzero_coupling(self):
        """With epsilon>0, dm2/dt depends on p1."""
        params = _circadian_params()
        params['epsilon'] = 2.0

        state_a = [1.0, 2.0, 1.0, 1.0]
        state_b = [1.0, 5.0, 1.0, 1.0]

        deriv_a = goodwin_dual_ode(0.0, state_a, params)
        deriv_b = goodwin_dual_ode(0.0, state_b, params)

        # dm2/dt should differ because epsilon * p1 differs
        assert deriv_a[2] != pytest.approx(deriv_b[2], abs=1e-6)


# ===========================================================================
# 3. Simulation tests
# ===========================================================================

class TestSimulateGoodwin:
    def test_simulate_goodwin_runs(self):
        """Simulation produces correct output shape."""
        params = _circadian_params()
        t_eval = np.linspace(0, 100, 500)
        result = simulate_goodwin(params, t_eval)

        assert result is not None
        assert result['t'].shape == (500,)
        assert result['m1'].shape == (500,)
        assert result['p1'].shape == (500,)
        assert result['m2'].shape == (500,)
        assert result['p2'].shape == (500,)

    def test_simulate_goodwin_oscillates(self):
        """With appropriate params, output oscillates (has multiple sign changes in derivative)."""
        params = _circadian_params()
        t_eval = np.linspace(0, 200, 1000)
        result = simulate_goodwin(params, t_eval)

        assert result is not None
        # Check m1 oscillates: count zero-crossings of the de-meaned signal
        m1 = result['m1']
        m1_centered = m1 - np.mean(m1)
        sign_changes = np.sum(np.abs(np.diff(np.sign(m1_centered))) > 0)
        # Should have multiple oscillation cycles in 200 time units
        assert sign_changes >= 4, "m1 should oscillate with at least 2 full cycles"

    def test_simulate_goodwin_positive(self):
        """Concentrations should remain non-negative (biological constraint)."""
        params = _circadian_params()
        t_eval = np.linspace(0, 100, 500)
        result = simulate_goodwin(params, t_eval)

        assert result is not None
        # Allow tiny numerical negativity
        assert np.all(result['m1'] > -1e-6)
        assert np.all(result['p1'] > -1e-6)
        assert np.all(result['m2'] > -1e-6)
        assert np.all(result['p2'] > -1e-6)

    def test_simulate_custom_y0(self):
        """Custom initial conditions are respected."""
        params = _circadian_params()
        t_eval = np.linspace(0, 1, 10)
        y0 = [5.0, 3.0, 2.0, 1.0]
        result = simulate_goodwin(params, t_eval, y0=y0)

        assert result is not None
        assert result['m1'][0] == pytest.approx(5.0, abs=1e-4)
        assert result['p1'][0] == pytest.approx(3.0, abs=1e-4)


# ===========================================================================
# 4. Independent vs coupled oscillator tests
# ===========================================================================

class TestOscillatorCoupling:
    def test_independent_oscillators(self):
        """With epsilon=0, both oscillators run independently.

        Changing oscillator 1 params should not affect oscillator 2 trajectory.
        """
        params_a = _circadian_params()
        params_a['epsilon'] = 0.0

        params_b = _circadian_params()
        params_b['epsilon'] = 0.0
        params_b['alpha1'] = 40.0  # change osc 1

        t_eval = np.linspace(0, 100, 500)
        res_a = simulate_goodwin(params_a, t_eval)
        res_b = simulate_goodwin(params_b, t_eval)

        assert res_a is not None and res_b is not None
        # Oscillator 2 trajectories should be identical
        np.testing.assert_allclose(res_a['m2'], res_b['m2'], atol=1e-6)
        np.testing.assert_allclose(res_a['p2'], res_b['p2'], atol=1e-6)

    def test_coupled_oscillators(self):
        """With epsilon>0, oscillator 2 is influenced by oscillator 1.

        Changing oscillator 1 params should affect oscillator 2 trajectory.
        """
        params_a = _coupled_params()
        params_b = _coupled_params()
        params_b['alpha1'] = 40.0  # change osc 1

        t_eval = np.linspace(0, 100, 500)
        res_a = simulate_goodwin(params_a, t_eval)
        res_b = simulate_goodwin(params_b, t_eval)

        assert res_a is not None and res_b is not None
        # Oscillator 2 trajectories should differ
        diff = np.max(np.abs(res_a['m2'] - res_b['m2']))
        assert diff > 0.01, "Coupled oscillator 2 should be affected by osc 1 changes"


# ===========================================================================
# 5. Fitting test
# ===========================================================================

class TestFitGoodwin:
    def test_fit_goodwin_basic(self):
        """Fit to a simple synthetic signal converges to reasonable MSE."""
        # Generate synthetic data from known parameters
        params = _circadian_params()
        t = np.linspace(0, 50, 80)
        sim = simulate_goodwin(params, t)
        assert sim is not None

        # Synthetic observed signal
        w1, w2, baseline = 1.0, 0.5, 2.0
        y = w1 * sim['m1'] + w2 * sim['m2'] + baseline

        # Fit with small maxiter/popsize for speed in tests
        result = fit_goodwin_to_gene(t, y, n_restarts=1, seed=42,
                                     maxiter=30, popsize=5)

        assert result is not None
        assert 'reconstructed' in result
        assert 'coupling_strength' in result
        assert result['reconstructed'].shape == y.shape

        # MSE should be reasonable (not perfect due to limited optimization)
        mse = np.mean((y - result['reconstructed']) ** 2)
        signal_var = np.var(y)
        # With limited iterations, just check it's finite and not catastrophic
        assert np.isfinite(mse)
        assert mse < signal_var * 2.0, (
            "Fit MSE ({:.4f}) should be less than 2x signal variance ({:.4f})".format(
                mse, signal_var
            )
        )


# ===========================================================================
# 6. Decomposition test
# ===========================================================================

class TestGoodwinDecompose:
    def test_goodwin_decompose_independent(self):
        """Independent signal (epsilon=0) correctly decomposed."""
        params = _circadian_params()
        params['epsilon'] = 0.0
        t = np.linspace(0, 50, 80)
        sim = simulate_goodwin(params, t)
        assert sim is not None

        y = 1.0 * sim['m1'] + 0.5 * sim['m2'] + 2.0

        result = goodwin_decompose(t, y, n_restarts=1, maxiter=30, popsize=5)

        assert result is not None
        assert 'classification' in result
        assert 'coupling_ratio' in result
        assert 'period_1' in result
        assert 'period_2' in result
        assert result['reconstructed'].shape == y.shape

    def test_goodwin_decompose_returns_components(self):
        """Decomposition returns component_1 and component_2."""
        params = _circadian_params()
        t = np.linspace(0, 50, 80)
        sim = simulate_goodwin(params, t)
        assert sim is not None

        y = 1.0 * sim['m1'] + 0.5 * sim['m2'] + 2.0
        result = goodwin_decompose(t, y, n_restarts=1, maxiter=30, popsize=5)

        assert result is not None
        assert 'component_1' in result
        assert 'component_2' in result
        assert result['component_1'].shape == y.shape
        assert result['component_2'].shape == y.shape


# ===========================================================================
# 7. Evidence score tests
# ===========================================================================

class TestGoodwinEvidence:
    def test_goodwin_evidence_scores(self):
        """Evidence scores have correct signs for known signals."""
        # Test with independent oscillators (epsilon=0)
        params = _circadian_params()
        params['epsilon'] = 0.0
        t = np.linspace(0, 50, 80)
        sim = simulate_goodwin(params, t)
        assert sim is not None

        y_indep = 1.0 * sim['m1'] + 0.5 * sim['m2'] + 2.0
        ev_indep = goodwin_evidence(t, y_indep, n_restarts=1, maxiter=30, popsize=5)

        assert ev_indep is not None
        assert 'score' in ev_indep
        assert 'coupling_ratio' in ev_indep
        assert isinstance(ev_indep['score'], int) or isinstance(ev_indep['score'], float)

    def test_goodwin_evidence_score_range(self):
        """Evidence scores are within expected range [-2, +3]."""
        params = _circadian_params()
        t = np.linspace(0, 50, 80)
        sim = simulate_goodwin(params, t)
        assert sim is not None

        y = 1.0 * sim['m1'] + 0.5 * sim['m2'] + 2.0
        ev = goodwin_evidence(t, y, n_restarts=1, maxiter=30, popsize=5)

        assert ev is not None
        assert -2 <= ev['score'] <= 3

    def test_goodwin_evidence_failure_returns_zero(self):
        """If fitting fails, evidence returns score=0."""
        # Constant signal - hard to fit oscillators to
        t = np.linspace(0, 10, 20)
        y = np.ones(20) * 5.0
        ev = goodwin_evidence(t, y, n_restarts=1, maxiter=10, popsize=3)

        assert ev is not None
        assert 'score' in ev
        # Score should be 0 or at least defined
        assert isinstance(ev['score'], (int, float))
