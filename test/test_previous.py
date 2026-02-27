import numpy as np
import pytest
import pandas as pd
from vectfit3 import vectfit, flat2full, buildRES, sortPoles

#Original test files: i wanted to benchmark speed


# ==============================================================================
# TEST 1: SCALAR ARTIFICIAL (Synthetic Ground Truth)
# ==============================================================================
def test_benchmark_scalar_artificial(benchmark, default_opts):
    """Benchmark for Test 1: Scalar artificial frequency domain function."""
    N = 101
    s = 2j * np.pi * np.logspace(0, 4, N, dtype=np.complex128)
    f = np.zeros(N, dtype=np.complex128)
    for n, sn in enumerate(s):
        f[n] = 2/(sn+5) + (30+40j)/(sn-(-100+500j)) + (30-40j)/(sn-(-100-500j)) + 0.5
    
    n_order = 3
    poles = -2 * np.pi * np.logspace(0, 4, n_order, dtype=np.complex128)
    opts = default_opts.copy()
    opts.update({"asymp": 3, "spy2": False})

    def run():
        return vectfit(f, s, poles, np.ones(N), opts)

    ser, final_poles, rmserr, fit = benchmark(run)
    benchmark.extra_info['Final_RMSE'] = f"{rmserr:.2e}"
    assert rmserr < 1e-4
    print(f"\n[METRIC] {benchmark.name} -> Final RMSE: {rmserr:.6e}")

# ==============================================================================
# TEST 2: 2D 18TH ORDER (Iterative Loop)
# ==============================================================================
def test_benchmark_2D_18th_order(benchmark, default_opts):
    """Benchmark for Test 2: 18th order frequency response (2 channels)."""
    N = 100
    w = 2 * np.pi * np.linspace(1, 1e5, N)
    s = 1j * w
    
    p_raw = np.array([-4500, -41000, -100+5e3j, -100-5e3j, -120+15e3j, -120-15e3j, 
                      -3e3+35e3j, -3e3-35e3j, -200+45e3j, -200-45e3j, -1500+45e3j, 
                      -1500-45e3j, -5e2+70e3j, -5e2-70e3j, -1e3+73e3j, -1e3-73e3j, 
                      -2e3+90e3j, -2e3-90e3j], dtype=np.complex128)
    p = 2 * np.pi * p_raw
    r_raw = np.array([-3000, -83000, -5+7e3j, -5-7e3j, -20+18e3j, -20-18e3j, 
                      6e3+45e3j, 6e3-45e3j, 40+60e3j, 40-60e3j, 90+10e3j, 
                      90-10e3j, 5e4+80e3j, 5e4-80e3j, 1e3+45e3j, 1e3-45e3j, 
                      -5e3+92e3j, -5e3-92e3j], dtype=np.complex128)
    r = 2 * np.pi * r_raw

    F = np.zeros((2, N), dtype=np.complex128)
    for idx, sn in enumerate(s):
        F[0, idx] = np.sum(r[:10] / (sn - p[:10])) + sn*2e-5 + 0.6
        F[1, idx] = np.sum(r[8:] / (sn - p[8:])) + sn*6e-5

    n_order = 18
    bet = np.linspace(w[0], w[-1], 9)
    initial_poles = np.zeros(n_order, dtype=np.complex128)
    for k in range(9):
        initial_poles[2*k] = -bet[k]*1e-2 - 1j*bet[k]
        initial_poles[2*k+1] = -bet[k]*1e-2 + 1j*bet[k]

    opts = default_opts.copy()
    opts.update({"asymp": 3, "cmplx_ss": False, "spy2": False})

    def run_iterative():
        curr_p = initial_poles.copy()
        for itr in range(3):
            res = vectfit(F, s, curr_p, np.ones(N), opts)
            curr_p = res[1]
        return res

    ser, final_poles, rmserr, fit = benchmark(run_iterative)
    benchmark.extra_info['Final_RMSE'] = f"{rmserr:.2e}"
    assert rmserr < 1e-2
    print(f"\n[METRIC] {benchmark.name} -> Final RMSE: {rmserr:.6e}")

# ==============================================================================
# TEST 3: TRANSFORMER (Measured Data)
# ==============================================================================
def test_benchmark_transformer_measured(benchmark, default_opts, load_vectfit_csv):
    """Benchmark for Test 3: Scalar measured transformer response."""
    Mdata = load_vectfit_csv("TRANSF_DATA.csv")
    f = Mdata[:160, 0] * np.exp(1j * Mdata[:160, 1] * np.pi / 180)
    w = 2 * np.pi * np.linspace(0, 10e6, 401)[1:161]
    s = 1j * w
    
    n_order = 30
    bet = np.linspace(w[0], w[-1], 15)
    poles = np.zeros(n_order, dtype=np.complex128)
    for k in range(15):
        poles[2*k] = -bet[k]*1e-2 - 1j*bet[k]
        poles[2*k+1] = -bet[k]*1e-2 + 1j*bet[k]

    opts = default_opts.copy()
    opts.update({"asymp": 3, "spy2": False})

    def run_transformer():
        curr_p = poles.copy()
        # Using inverse weighting as in your original script
        weights = 1/np.abs(f)
        for _ in range(5):
            res = vectfit(f, s, curr_p, weights, opts)
            curr_p = res[1]
        return res

    ser, final_poles, rmserr, fit = benchmark(run_transformer)
    benchmark.extra_info['Final_RMSE'] = f"{rmserr:.2e}"
    assert rmserr < 0.1
    print(f"\n[METRIC] {benchmark.name} -> Final RMSE: {rmserr:.6e}")

# ==============================================================================
# TEST 4: ADMITTANCE MATRIX 6x6 (Symmetric)
# ==============================================================================
def test_benchmark_admittance_matrix(benchmark, default_opts, load_vectfit_csv):
    """Benchmark for Test 4: 6x6 Symmetric Admittance Matrix."""
    Mdata = np.ravel(load_vectfit_csv("SYSADMITANCE_DATA.csv"))
    N = int(Mdata[0])
    F = np.zeros((21, N), dtype=np.complex128)
    s = np.zeros(N, dtype=np.complex128)
    
    k = 0
    for i in range(1, Mdata.size, 73):
        s[k] = 1j * Mdata[i]
        block = Mdata[i+1 : i+73]
        y_mat = block[0::2] + 1j*block[1::2]
        y_mat = y_mat.reshape((6, 6))
        # Extract lower triangular (21 elements)
        idx_f = 0
        for r in range(6):
            for c in range(r, 6):
                F[idx_f, k] = y_mat[r, c]
                idx_f += 1
        k += 1

    n_order = 50
    w = s.imag
    bet = np.linspace(w[0], w[-1], 25)
    poles = np.zeros(n_order, dtype=np.complex128)
    for k in range(25):
        poles[2*k] = -bet[k]/100 - 1j*bet[k]
        poles[2*k+1] = -bet[k]/100 + 1j*bet[k]

    opts = default_opts.copy()
    opts.update({"asymp": 3, "symm_mat": True, "spy2": False})

    def run_admittance():
        curr_p = poles.copy()
        weights = 1/np.sqrt(np.abs(F))
        for _ in range(3):
            res = vectfit(F, s, curr_p, weights, opts)
            curr_p = res[1]
        return res

    ser, final_poles, rmserr, fit = benchmark(run_admittance)
    benchmark.extra_info['Final_RMSE'] = f"{rmserr:.2e}"
    
    # Reconstruction Precision Check
    ser_full = flat2full(ser)
    Res = buildRES(ser_full["C"], ser_full["B"])
    f_rebuilt = np.zeros(N, dtype=np.complex128)
    for m in range(n_order):
        f_rebuilt += Res[0, 0, m] / (s - final_poles[m])
    f_rebuilt += ser_full["D"][0, 0] + s * ser_full["E"][0, 0]
    np.testing.assert_allclose(f_rebuilt, fit[0, :], rtol=1e-8)
    
    print(f"\n[METRIC] {benchmark.name} -> Final RMSE: {rmserr:.6e}")

# ==============================================================================
# TEST 5: PROPAGATION MATRIX 3x3 (Asymmetric)
# ==============================================================================
def test_benchmark_propagation_matrix(benchmark, default_opts, load_vectfit_csv):
    """Benchmark for Test 5: 3x3 Asymmetric Propagation Matrix."""
    Hdata = pd.DataFrame(load_vectfit_csv("MODEH_DATA.csv"))
    w = Hdata.iloc[:, 1].to_numpy()
    s = 1j * w
    N = len(w)
    
    F = np.zeros((9, N), dtype=np.complex128)
    # Trace of H for initial pole fitting
    trH = np.zeros(N, dtype=np.complex128)
    
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            # Data is in RMO: Real, Imag, Real, Imag... starting at col index 2
            col_idx = 2 + (idx * 2)
            F[idx, :] = Hdata.iloc[:, col_idx].to_numpy() + 1j*Hdata.iloc[:, col_idx+1].to_numpy()
            if row == col:
                trH += F[idx, :]

    n_order = 35
    bet = np.logspace(np.log10(w[0]), np.log10(w[-1]), 17)
    poles = np.zeros(35, dtype=np.complex128) # Note: 35 is odd, one real pole needed
    poles[0] = -bet[0]/100
    for k in range(17):
        poles[2*k+1] = -bet[k]/100 - 1j*bet[k]
        poles[2*k+2] = -bet[k]/100 + 1j*bet[k]

    opts = default_opts.copy()
    opts.update({"asymp": 1, "spy2": False, "cmplx_ss": True})

    def run_propagation():
        curr_p = poles.copy()
        # 1. Fit trace for 10 iterations
        for _ in range(10):
            res_tr = vectfit(trH, s, curr_p, np.ones(N), opts)
            curr_p = res_tr[1]
        # 2. Fit full matrix for 10 iterations
        for _ in range(10):
            res_full = vectfit(F, s, curr_p, np.ones(N), opts)
            curr_p = res_full[1]
        return res_full

    ser, final_poles, rmserr, fit = benchmark(run_propagation)
    benchmark.extra_info['Final_RMSE'] = f"{rmserr:.2e}"
    
    # Structural integrity check
    ser_full = flat2full(ser)
    Res = buildRES(ser_full["C"], ser_full["B"])
    f_rebuilt = np.zeros(N, dtype=np.complex128)
    for m in range(n_order):
        f_rebuilt += Res[0, 0, m] / (s - final_poles[m])
    np.testing.assert_allclose(f_rebuilt, fit[0, :], rtol=1e-8)
    
    print(f"\n[METRIC] {benchmark.name} -> Final RMSE: {rmserr:.6e}")