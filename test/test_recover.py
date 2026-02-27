import numpy as np
import pytest
from vectfit3 import vectfit, sortPoles 

def test_scalar_3_poles_recovery(default_opts, generate_synthetic_fs):
    """
    Tests if vectfit can recover a known 3-pole scalar transfer function.
    """
    # ==========================================
    # 1. ARRANGE (Set up the ground truth)
    # ==========================================
    N = 100
    w = 2 * np.pi * np.logspace(0, 4, N)
    s = 1j * w
    
    # Define our mathematical ground truth (stable poles in the left half-plane)
    expected_poles = np.array([
        -500 + 0j,                
        -100 + 5000j,             
        -100 - 5000j
    ], dtype=np.complex128)
    
    expected_residues = np.array([
        2000 + 0j,
        300 + 400j,
        300 - 400j
    ], dtype=np.complex128)

    F_true = generate_synthetic_fs(s, expected_poles, expected_residues)
    weights = np.ones(N, dtype=np.float64)
    n_poles = 3
    
    #bad poles
    initial_poles = np.array([
        -10 + 0j,
        -10 + 1000j,
        -10 - 1000j
    ], dtype=np.complex128)
    
    opts = default_opts
    opts["spy1"] = False
    opts["spy2"] = False
    opts["phaseplot"] = False
    opts["errplot"] = False
    opts["asymp"] = 1 # D=0, E=0 for this specific test

    # ==========================================
    # 2. ACT (Run the algorithm)
    # ==========================================
    Niter = 4 
    current_poles = initial_poles.copy()
    
    for itr in range(Niter):
        SER, current_poles, rmserr, fit = vectfit(F_true, s, current_poles, weights, opts)

    # ==========================================
    # 3. ASSERT (Verify the results)
    # ==========================================
    assert rmserr < 1e-4, f"Vector fitting failed to converge tightly. RMS Error: {rmserr}" #TODO: is rmserr < 1e-4 ok? 
    
    # B. Check if the reconstructed frequency response matches the true response
    # F_true is 1D, fit might be reshaped to (1, N) by vectfit, so we squeeze it to compare safely
    np.testing.assert_allclose(np.squeeze(fit), F_true, rtol=1e-3, atol=1e-3, err_msg="Fitted frequency response diverges from ground truth")

    sorted_expected = sortPoles(expected_poles)
    sorted_actual = sortPoles(current_poles)
    
    np.testing.assert_allclose(sorted_actual, sorted_expected, rtol=1e-2,  err_msg="Wrong poles recovered. ")
def test_scalar_recovery_asymp_2(default_opts, generate_synthetic_fs):
    """
    Tests recovery with asymp=2: includes constant D term.
    F(s) = sum(r/(s-p)) + D
    """
    # 1. ARRANGE
    N = 100
    w = 2 * np.pi * np.logspace(0, 4, N)
    s = 1j * w
    
    expected_poles = np.array([-100 + 1000j, -100 - 1000j], dtype=np.complex128)
    expected_residues = np.array([50 + 10j, 50 - 10j], dtype=np.complex128)
    expected_D = 0.5  # Constant offset
    
    # Generate ground truth with D
    F_true = generate_synthetic_fs(s, expected_poles, expected_residues, D=expected_D)
    
    initial_poles = np.array([-10 + 500j, -10 - 500j], dtype=np.complex128)
    weights = np.ones(N, dtype=np.float64)
    
    opts = default_opts.copy()
    opts.update({"spy1": False, "spy2": False, "asymp": 2})

    # 2. ACT
    current_poles = initial_poles.copy()
    for _ in range(5):
        SER, current_poles, rmserr, fit = vectfit(F_true, s, current_poles, weights, opts)

    # 3. ASSERT
    assert rmserr < 1e-4
    actual_D = np.squeeze(SER['D'])
    np.testing.assert_allclose(actual_D, expected_D, rtol=1e-2, err_msg="Failed to recover D term")


def test_scalar_recovery_asymp_3(default_opts, generate_synthetic_fs):
    """
    Tests recovery with asymp=3: includes D and E (proportional) terms.
    F(s) = sum(r/(s-p)) + D + s*E
    """
    # 1. ARRANGE
    N = 100
    w = 2 * np.pi * np.logspace(1, 5, N) # Higher frequency range to see E effect
    s = 1j * w
    
    expected_poles = np.array([-500 + 5000j, -500 - 5000j], dtype=np.complex128)
    expected_residues = np.array([1000 + 200j, 1000 - 200j], dtype=np.complex128)
    expected_D = 1.2
    expected_E = 2e-6  # Small value, but noticeable at high freq (s*E)
    
    F_true = generate_synthetic_fs(s, expected_poles, expected_residues, D=expected_D, E=expected_E)
    
    initial_poles = np.array([-100 + 2000j, -100 - 2000j], dtype=np.complex128)
    weights = np.ones(N, dtype=np.float64)
    
    opts = default_opts.copy()
    opts.update({"spy1": False, "spy2": False, "asymp": 3})

    # 2. ACT
    current_poles = initial_poles.copy()
    for _ in range(5):
        SER, current_poles, rmserr, fit = vectfit(F_true, s, current_poles, weights, opts)

    # 3. ASSERT
    assert rmserr < 1e-4
    actual_D = np.squeeze(SER['D'])
    actual_E = np.squeeze(SER['E'])
    
    np.testing.assert_allclose(actual_D, expected_D, rtol=1e-2, err_msg="Failed to recover D term")
    np.testing.assert_allclose(actual_E, expected_E, rtol=1e-2, err_msg="Failed to recover E term")