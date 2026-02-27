import numpy as np
import pytest
from vectfit3 import vectfit, sortPoles
import numpy as np
import pytest
from scipy.constants import pi
from vectfit3 import vectfit, sortPoles

def test_vector_4xN_recovery_unitary(default_opts, generate_synthetic_fs):
    """
    Standard test: 4xN vector response with unitary (equal) weighting.
    """
    # 1. ARRANGE
    N = 200
    w = 2 * pi * np.logspace(1, 4, N)
    s = 1j * w
    expected_poles = np.array([-100, -50 + 2000j, -50 - 2000j], dtype=np.complex128)
    expected_residues = np.array([
        [500,  10 + 50j,  10 - 50j],
        [1200, 80 + 10j,  80 - 10j],
        [100,  200 - 5j,  200 + 5j],
        [900,  10 + 100j, 10 - 100j],
    ], dtype=np.complex128)

    F_true = generate_synthetic_fs(s, expected_poles, expected_residues)
    weights = np.ones(N, dtype=np.float64)
    initial_poles = np.array([-10, -10 + 500j, -10 - 500j], dtype=np.complex128)
    
    opts = default_opts.copy()
    opts.update({"spy1": False, "spy2": False, "asymp": 1, "cmplx_ss": True})

    # 2. ACT
    current_poles = initial_poles.copy()
    for _ in range(5):
        _, current_poles, rmserr, fit = vectfit(F_true, s, current_poles, weights, opts)

    # 3. ASSERT
    assert rmserr < 1e-3
    sorted_expected = sortPoles(expected_poles)
    sorted_actual = sortPoles(current_poles)
    np.testing.assert_allclose(sorted_actual, sorted_expected, rtol=1e-2)


def test_vector_4xN_recovery_random_weights(default_opts, generate_synthetic_fs):
    """
    Stress test: 4xN vector response with random weights.
    This tests the numerical stability of the weighted least-squares implementation.
    """
    # 1. ARRANGE
    N = 200
    w = 2 * pi * np.logspace(1, 4, N)
    s = 1j * w
    expected_poles = np.array([-100, -50 + 2000j, -50 - 2000j], dtype=np.complex128)
    expected_residues = np.array([
        [500,  10 + 50j,  10 - 50j],
        [1200, 80 + 10j,  80 - 10j],
        [100,  200 - 5j,  200 + 5j],
        [900,  10 + 100j, 10 - 100j],
    ], dtype=np.complex128)

    F_true = generate_synthetic_fs(s, expected_poles, expected_residues)
    
    # Random weights: Testing sensitivity to non-uniform priority
    np.random.seed(42) 
    weights = np.random.uniform(0.5, 5.0, N)
    
    initial_poles = np.array([-10, -10 + 500j, -10 - 500j], dtype=np.complex128)
    
    opts = default_opts.copy()
    opts.update({"spy1": False, "spy2": False, "asymp": 1, "cmplx_ss": True})

    # 2. ACT
    current_poles = initial_poles.copy()
    for _ in range(7): # Extra iterations for potentially slower convergence with random weights
        _, current_poles, rmserr, fit = vectfit(F_true, s, current_poles, weights, opts)

    # 3. ASSERT
    # We allow a slightly higher RMS error because random weights can pull 
    # the fit away from the global minimum of the unweighted error.
    assert rmserr < 5e-3, f"Random weight fitting failed to converge. RMS: {rmserr}"
    
    sorted_expected = sortPoles(expected_poles)
    sorted_actual = sortPoles(current_poles)
    
    # Check if the recovered poles are still physically accurate
    np.testing.assert_allclose(sorted_actual, sorted_expected, rtol=5e-2,
                               err_msg="Pole recovery was significantly skewed by random weights")