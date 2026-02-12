import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# 4-PAM (unit power) utilities
# ----------------------------
def pam4_levels_unit_power():
    # raw levels: [-3, -1, 1, 3], average energy = 5
    return np.array([-3, -1, 1, 3], dtype=float) / np.sqrt(5.0)

def pam4_generate(K, rng):
    levels = pam4_levels_unit_power()
    idx = rng.integers(0, 4, size=K)
    return levels[idx]

def pam4_detect(x):
    levels = pam4_levels_unit_power()
    # nearest-neighbor slicing
    # x shape: (M,) or scalar
    x = np.asarray(x)
    d = np.abs(x[..., None] - levels[None, ...])
    return levels[np.argmin(d, axis=-1)]


# ----------------------------
# Channel + noise
# ----------------------------
def channel_f():
    # F(z) = 1/sqrt(2) * (0.8 - z^-1 + 0.6 z^-2)
    return np.array([0.8, -1.0, 0.6], dtype=float) / np.sqrt(2.0)

def apply_channel(I, f):
    # r_sig(k) = sum_{l=0}^{L-1} f[l] I(k-l)
    # convolution gives length K+L-1; we take same length as I by padding implicitly
    return np.convolve(I, f, mode="full")

def add_awgn(x, snr_db, rng):
    # given SNR = 1 / sigma_v^2 (since signal power is 1)
    sigma_v2 = 10.0 ** (-snr_db / 10.0)
    v = rng.normal(0.0, np.sqrt(sigma_v2), size=x.shape)
    return x + v, sigma_v2


# ----------------------------
# SWF statistics: R and p
# ----------------------------
def r_autocorr_lags(f, sigma_I2, sigma_v2, max_lag):
    """
    Returns R_r[m] for m = 0..max_lag (nonnegative lags).
    r(k) = sum f[l] I(k-l) + v(k), with I iid, zero-mean, var sigma_I2,
    v white, var sigma_v2.
    """
    L = len(f)
    R = np.zeros(max_lag + 1, dtype=float)

    # For m >= 0:
    # R_r[m] = sigma_I2 * sum_{l} f[l] f[l-m] (where valid) + sigma_v2 * delta[m]
    for m in range(max_lag + 1):
        s = 0.0
        for l in range(L):
            lm = l - m
            if 0 <= lm < L:
                s += f[l] * f[lm]
        R[m] = sigma_I2 * s + (sigma_v2 if m == 0 else 0.0)

    return R

def build_R_toeplitz(Rlags, N):
    # R[i,j] = R_r[|i-j|] for WSS real process, using nonnegative lag array
    R = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            R[i, j] = Rlags[abs(i - j)]
    return R

def build_p_vector(f, sigma_I2, N, Delta):
    """
    p[i] = E[d(k) r(k-i)], d(k)=I(k-Delta)
    r(k-i)=sum_l f[l] I(k-i-l)+v
    Nonzero when k-Delta = k-i-l -> l = i-Delta
    """
    L = len(f)
    p = np.zeros(N, dtype=float)
    for i in range(N):
        l = i - Delta
        if 0 <= l < L:
            p[i] = sigma_I2 * f[l]
    return p

def swf_wopt_Jmin(f, sigma_I2, sigma_v2, N, Delta):
    Rlags = r_autocorr_lags(f, sigma_I2, sigma_v2, max_lag=N-1)
    R = build_R_toeplitz(Rlags, N)
    p = build_p_vector(f, sigma_I2, N, Delta)

    w = np.linalg.solve(R, p)  # wopt
    Jmin = sigma_I2 - p @ w
    return w, Jmin


# ----------------------------
# Simulation: SER for a given w
# ----------------------------
def equalize_and_ser(I, r, w, N, Delta):
    """
    Uses y(k)=w^T r_vec(k), r_vec=[r(k),...,r(k-N+1)]
    compares to d(k)=I(k-Delta) over valid k.
    """
    K = len(I)
    # r is longer due to convolution; ensure r has at least K samples aligned
    # We'll interpret r[k] as measurement at time k (same indexing as I),
    # assuming I starts at k=0 and missing past is zero.
    # For safety: if r shorter, truncate K accordingly.
    K_eff = min(K, len(r))
    I = I[:K_eff]
    r = r[:K_eff]

    k_start = N - 1
    k_end = K_eff - 1

    ys = []
    ds = []
    for k in range(k_start, k_end + 1):
        if k - Delta < 0:
            continue
        rvec = r[k : k - N : -1]  # r[k], r[k-1], ..., r[k-N+1]
        if len(rvec) != N:
            continue
        yk = w @ rvec
        ys.append(yk)
        ds.append(I[k - Delta])

    ys = np.array(ys)
    ds = np.array(ds)

    det = pam4_detect(ys)
    ser = np.mean(det != ds) if len(ds) > 0 else np.nan
    return ser


# ----------------------------
# TAWF build from P snapshots
# ----------------------------
def build_tawf_from_data(I, r, N, Delta, P):
    """
    Rhat = (1/P) sum rvec rvec^T
    phat = (1/P) sum d rvec, d=I(k-Delta)
    Uses overlapping snapshots, k successive.
    """
    K_eff = min(len(I), len(r))
    I = I[:K_eff]
    r = r[:K_eff]

    Rhat = np.zeros((N, N), dtype=float)
    phat = np.zeros(N, dtype=float)

    # choose k range that gives valid rvec and valid d
    k0 = N - 1
    count = 0
    k = k0
    while count < P and k < K_eff:
        if k - Delta >= 0:
            rvec = r[k : k - N : -1]
            if len(rvec) == N:
                d = I[k - Delta]
                Rhat += np.outer(rvec, rvec)
                phat += d * rvec
                count += 1
        k += 1

    if count == 0:
        raise ValueError("No valid snapshots collected. Check N/Delta/P and signal length.")

    Rhat /= count
    phat /= count

    w_hat = np.linalg.solve(Rhat, phat)
    # estimated Jmin using same quadratic form
    sigma_I2_hat = np.mean(I**2)
    Jmin_hat = sigma_I2_hat - phat @ w_hat
    return w_hat, Jmin_hat, count


# ----------------------------
# Trial-and-error best (N, Delta)
# ----------------------------
def find_best_N_Delta_swf(f, sigma_I2, sigma_v2, N_list, Delta_list):
    best = None
    for N in N_list:
        for Delta in Delta_list:
            if Delta < 0 or Delta > N - 1:
                continue
            _, Jmin = swf_wopt_Jmin(f, sigma_I2, sigma_v2, N, Delta)
            if (best is None) or (Jmin < best[0]):
                best = (Jmin, N, Delta)
    return best  # (Jmin, N, Delta)


# ----------------------------
# Main: runs (a1)-(a7)
# ----------------------------
def main():
    rng = np.random.default_rng(0)

    f = channel_f()
    sigma_I2 = 1.0  # unit power PAM

    # ----------------------------
    # (a1) N=3, Delta=0, SNR=10dB
    # ----------------------------
    snr_db = 10.0
    sigma_v2 = 10.0 ** (-snr_db / 10.0)
    w3, J3 = swf_wopt_Jmin(f, sigma_I2, sigma_v2, N=3, Delta=0)
    print("(a1) SWF: N=3, Δ=0, SNR=10dB")
    print("wopt =", w3)
    print("Jmin =", J3)

    # ----------------------------
    # (a2) N=10, Delta=0, SNR=10dB
    # ----------------------------
    w10_d0, J10_d0 = swf_wopt_Jmin(f, sigma_I2, sigma_v2, N=10, Delta=0)
    print("\n(a2) SWF: N=10, Δ=0, SNR=10dB")
    print("Jmin =", J10_d0)

    # ----------------------------
    # (a3) N=10, Delta=5, SNR=10dB
    # ----------------------------
    w10_d5, J10_d5 = swf_wopt_Jmin(f, sigma_I2, sigma_v2, N=10, Delta=5)
    print("\n(a3) SWF: N=10, Δ=5, SNR=10dB")
    print("Jmin =", J10_d5)

    # ----------------------------
    # (a4) best N,Δ for SNR=15dB (trial-and-error)
    # ----------------------------
    snr_db_15 = 15.0
    sigma_v2_15 = 10.0 ** (-snr_db_15 / 10.0)

    N_list = range(1, 21)           # try 1..20
    Delta_list = range(0, 21)       # will be clipped by Delta<=N-1
    best_J, best_N, best_D = find_best_N_Delta_swf(
        f, sigma_I2, sigma_v2_15, N_list, Delta_list
    )
    print("\n(a4) Best SWF at SNR=15dB over N=1..20, Δ=0..N-1")
    print(f"Best: Jmin={best_J:.6g}, N={best_N}, Δ={best_D}")

    # ----------------------------
    # (a5) SER curve for SWF using best (N,Δ) from (a4)
    # ----------------------------
    snr_grid = np.arange(0, 22, 2)  # 0..20 in steps of 2
    K_ser = 50000

    ser_swf = []
    for snr_db_i in snr_grid:
        # regenerate symbols and channel each point (fine for assignment)
        I = pam4_generate(K_ser + 50, rng)
        r_sig = apply_channel(I, f)
        r_noisy, sigma_v2_i = add_awgn(r_sig, snr_db_i, rng)

        w_i, _ = swf_wopt_Jmin(f, sigma_I2, sigma_v2_i, best_N, best_D)
        ser = equalize_and_ser(I, r_noisy, w_i, best_N, best_D)
        ser_swf.append(ser)

    ser_swf = np.array(ser_swf)

    # ----------------------------
    # (a6) TAWF Jmin(P) at SNR=10dB, N=10, Δ=5, P={20,100,500}
    # ----------------------------
    P_list = [20, 100, 500]
    # build one training stream
    I_train = pam4_generate(3000, rng)
    r_sig_train = apply_channel(I_train, f)
    r_train, sigma_v2_train = add_awgn(r_sig_train, 10.0, rng)

    print("\n(a6) TAWF at SNR=10dB, N=10, Δ=5")
    for P in P_list:
        w_hat, J_hat, used = build_tawf_from_data(I_train, r_train, N=10, Delta=5, P=P)
        print(f"P={P:4d} (used {used:4d}) -> Jmin(P)≈ {J_hat:.6g}")

    print(f"Compare with SWF Jmin from (a3): {J10_d5:.6g}")

    # ----------------------------
    # (a7) TAWF SER curve with best (N,Δ) from (a4), using P=500 snapshots
    #       then test on NEW 50k symbols (per SNR)
    # ----------------------------
    ser_tawf = []
    P_train = 500

    for snr_db_i in snr_grid:
        # training
        I_tr = pam4_generate(P_train + 200 + best_N + best_D + 50, rng)
        r_tr_sig = apply_channel(I_tr, f)
        r_tr, _ = add_awgn(r_tr_sig, snr_db_i, rng)

        w_hat, _, _ = build_tawf_from_data(I_tr, r_tr, best_N, best_D, P=P_train)

        # testing (new data)
        I_te = pam4_generate(K_ser + 50, rng)
        r_te_sig = apply_channel(I_te, f)
        r_te, _ = add_awgn(r_te_sig, snr_db_i, rng)

        ser = equalize_and_ser(I_te, r_te, w_hat, best_N, best_D)
        ser_tawf.append(ser)

    ser_tawf = np.array(ser_tawf)

    # ----------------------------
    # Fig 3: log10(SER) vs SNR(dB)
    # ----------------------------
    plt.figure()
    plt.plot(snr_grid, np.log10(ser_swf + 1e-12), marker="o", label="SWF (best N,Δ from 15dB)")
    plt.plot(snr_grid, np.log10(ser_tawf + 1e-12), marker="s", label="TAWF (P=500)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("log10(SER)")
    plt.grid(True)
    plt.legend()
    plt.title("Fig 3: SER vs SNR for SWF and TAWF (LE)")
    plt.savefig("figures/fig3_ser_vs_snr.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("\nSaved: figures/fig3_ser_vs_snr.png")


if __name__ == "__main__":
    main()
