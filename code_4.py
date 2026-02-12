import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# 2-PAM utilities (unit power)
# ----------------------------
def pam2_generate(K, rng):
    return rng.choice(np.array([-1.0, 1.0]), size=K)

def pam2_detect(x):
    x = np.asarray(x)
    return np.where(x >= 0, 1.0, -1.0)


# ----------------------------
# Channel G(z) normalization
# ----------------------------
def channel_g_normalized():
    raw = np.array([1.0, -0.95, 0.5, 0.15, -0.2, -0.1], dtype=float)  # L=6
    C = np.sqrt(np.sum(raw**2))  # makes sum g^2 = 1 (since sigma_I^2 = 1)
    g = raw / C
    return g, C

def apply_channel(I, h):
    return np.convolve(I, h, mode="full")

def add_awgn(x, snr_db, rng):
    # SNR on r(k): since channel normalized, signal power ~1 => sigma_v^2 = 10^(-SNR/10)
    sigma_v2 = 10.0 ** (-snr_db / 10.0)
    v = rng.normal(0.0, np.sqrt(sigma_v2), size=x.shape)
    return x + v, sigma_v2


# ----------------------------
# SWF statistics for LE
# ----------------------------
def r_autocorr_lags(h, sigma_I2, sigma_v2, max_lag):
    L = len(h)
    R = np.zeros(max_lag + 1, dtype=float)
    for m in range(max_lag + 1):
        s = 0.0
        for l in range(L):
            lm = l - m
            if 0 <= lm < L:
                s += h[l] * h[lm]
        R[m] = sigma_I2 * s + (sigma_v2 if m == 0 else 0.0)
    return R

def build_R_toeplitz(Rlags, N):
    R = np.empty((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            R[i, j] = Rlags[abs(i - j)]
    return R

def build_p_le(h, sigma_I2, N, Delta):
    # p[i] = E[d(k) r(k-i)], d(k)=I(k-Delta)
    # nonzero when l = i - Delta is a valid channel tap index
    L = len(h)
    p = np.zeros(N, dtype=float)
    for i in range(N):
        l = i - Delta
        if 0 <= l < L:
            p[i] = sigma_I2 * h[l]
    return p

def swf_le_wopt_Jmin(h, sigma_I2, sigma_v2, N, Delta):
    Rlags = r_autocorr_lags(h, sigma_I2, sigma_v2, max_lag=N-1)
    R = build_R_toeplitz(Rlags, N)
    p = build_p_le(h, sigma_I2, N, Delta)
    w = np.linalg.solve(R, p)
    Jmin = sigma_I2 - p @ w
    return w, Jmin


# ----------------------------
# SWF statistics for DFE
# ----------------------------
def swf_dfe_wopt_Jmin(h, sigma_I2, sigma_v2, N1, N2, Delta):
    """
    u(k) = [ r(k),...,r(k-N1+1),  -d(k-1),...,-d(k-N2) ]^T
    d(k)=I(k-Delta). Perfect feedback assumed in SWF stats.
    """
    L = len(h)
    # R_rr
    Rlags = r_autocorr_lags(h, sigma_I2, sigma_v2, max_lag=N1-1)
    R_rr = build_R_toeplitz(Rlags, N1)

    # R_ss: E[(-d(k-i))(-d(k-j))] = E[d(k-i)d(k-j)] = sigma_I2 * delta_ij
    R_ss = sigma_I2 * np.eye(N2)

    # R_rs: E[r(k-m) * (-d(k-i))] = - E[r(k-m) d(k-i)]
    # with r(k-m)=sum_l h[l] I(k-m-l)
    # d(k-i)=I(k-Delta-i)
    # match when k-m-l = k-Delta-i => l = Delta + i - m
    R_rs = np.zeros((N1, N2), dtype=float)
    for m in range(N1):      # r(k-m)
        for i in range(1, N2+1):  # d(k-i)
            l = Delta + i - m
            if 0 <= l < L:
                R_rs[m, i-1] = -sigma_I2 * h[l]

    # Assemble full R
    top = np.hstack([R_rr, R_rs])
    bot = np.hstack([R_rs.T, R_ss])
    R = np.vstack([top, bot])

    # p = E[d(k) u(k)]
    # p_r[m] = E[d(k) r(k-m)] same as LE p with N=N1
    p_r = build_p_le(h, sigma_I2, N1, Delta)
    # p_s entries: E[d(k) * (-d(k-i))] = 0 for i>=1 (iid zero-mean)
    p_s = np.zeros(N2, dtype=float)
    p = np.concatenate([p_r, p_s])

    w = np.linalg.solve(R, p)
    Jmin = sigma_I2 - p @ w
    return w, Jmin


# ----------------------------
# Simulation: SER for LE
# ----------------------------
def ser_le(I, r, w, N, Delta):
    K_eff = min(len(I), len(r))
    I = I[:K_eff]
    r = r[:K_eff]

    k_start = N - 1
    ys, ds = [], []
    for k in range(k_start, K_eff):
        if k - Delta < 0:
            continue
        rvec = r[k:k-N:-1]
        if len(rvec) != N:
            continue
        y = w @ rvec
        ys.append(y)
        ds.append(I[k - Delta])

    ys = np.array(ys)
    ds = np.array(ds)
    det = pam2_detect(ys)
    return np.mean(det != ds) if len(ds) else np.nan


# ----------------------------
# Simulation: SER for DFE
# ----------------------------
def ser_dfe(I, r, w_ff, w_fb, N1, N2, Delta):
    """
    y(k) = w_ff^T rvec(k) - w_fb^T d_hat_past
    d_hat_past corresponds to previously detected d(k-1),...,d(k-N2)
    """
    K_eff = min(len(I), len(r))
    I = I[:K_eff]
    r = r[:K_eff]

    # store detected d-hat indexed by time k (estimates of I(k-Delta))
    d_hat = np.zeros(K_eff, dtype=float)

    k_start = max(N1 - 1, Delta)  # need rvec and target
    err_count = 0
    total = 0

    for k in range(k_start, K_eff):
        rvec = r[k:k-N1:-1]
        if len(rvec) != N1:
            continue

        # build past detected symbols d_hat(k-1)...d_hat(k-N2)
        past = []
        for i in range(1, N2+1):
            if k - i >= 0:
                past.append(d_hat[k - i])
            else:
                past.append(0.0)
        past = np.array(past, dtype=float)

        y = w_ff @ rvec - w_fb @ past

        d_dec = pam2_detect(y)
        d_hat[k] = d_dec

        d_true = I[k - Delta]
        err_count += (d_dec != d_true)
        total += 1

    return err_count / total if total else np.nan


# ----------------------------
# Viterbi (MLSE) for 2-PAM ISI
# ----------------------------
def viterbi_mlse_2pam(I, r, h, delta=30):
    """
    MLSE using Viterbi for channel memory M=L-1.
    For 2-PAM => number of states = 2^M.
    Note: Your prompt says "25 state VA" but with L=6, M=5 => 32 states for 2-PAM.
    """
    L = len(h)
    M = L - 1
    states = 2 ** M

    K_eff = min(len(I), len(r))
    I = I[:K_eff]
    r = r[:K_eff]

    # Map state index -> past M symbols (most recent first)
    # each symbol in {-1, +1}
    def idx_to_bits(s):
        bits = np.array([(s >> b) & 1 for b in range(M)], dtype=int)
        # bit 0 corresponds to most recent past symbol; map 0->-1, 1->+1
        return np.where(bits == 1, 1.0, -1.0)

    past_symbols = np.stack([idx_to_bits(s) for s in range(states)], axis=0)

    # Initialize path metrics
    INF = 1e18
    pm = np.full(states, INF, dtype=float)
    pm[0] = 0.0  # arbitrary start (all -1 past)
    prev_state = np.zeros((K_eff, states), dtype=int)
    prev_sym = np.zeros((K_eff, states), dtype=float)

    # Precompute next-state transitions for input symbol a in {-1,+1}
    def next_state(s, a):
        bits = idx_to_bits(s)
        new_past = np.concatenate([[a], bits[:-1]])
        new_bits = (new_past > 0).astype(int)
        ns = 0
        for b in range(M):
            ns |= (new_bits[b] << b)
        return ns

    for k in range(K_eff):
        pm_new = np.full(states, INF, dtype=float)

        for s in range(states):
            if pm[s] >= INF/2:
                continue

            for a in (-1.0, 1.0):  # candidate current symbol
                ns = next_state(s, a)

                # predicted noiseless r(k) = sum_{l=0}^{L-1} h[l] I(k-l)
                # current a is I(k), past_symbols[s][0] is I(k-1), etc.
                past = past_symbols[s]
                seq = np.concatenate([[a], past])  # length L
                r_hat = np.dot(h, seq)

                metric = (r[k] - r_hat) ** 2
                cand = pm[s] + metric
                if cand < pm_new[ns]:
                    pm_new[ns] = cand
                    prev_state[k, ns] = s
                    prev_sym[k, ns] = a

        pm = pm_new

    # Traceback: choose best final state
    s_best = int(np.argmin(pm))
    x_hat = np.zeros(K_eff, dtype=float)
    for k in range(K_eff-1, -1, -1):
        x_hat[k] = prev_sym[k, s_best]
        s_best = prev_state[k, s_best]

    # Apply decoding delay delta: compare x_hat[k-delta] to true I[k-delta]
    start = delta
    if K_eff <= delta:
        return np.nan
    det = x_hat[:-delta]
    tru = I[delta:]
    return np.mean(det != tru)


# ----------------------------
# (b) runner
# ----------------------------
def main():
    rng = np.random.default_rng(1)

    h, C = channel_g_normalized()
    sigma_I2 = 1.0  # 2-PAM ±1 => unit power

    print("Normalized channel taps h =", h)
    print("Normalization constant C =", C)
    print("Check sum(h^2) =", np.sum(h**2))

    # ----------------------------
    # (b1) SWF-LE, N=20, Δ=9 at SNR=15dB
    # ----------------------------
    snr15 = 15.0
    sigma_v2_15 = 10.0 ** (-snr15 / 10.0)

    w_le, Jmin_le = swf_le_wopt_Jmin(h, sigma_I2, sigma_v2_15, N=20, Delta=9)
    print("\n(b1) SWF-LE: N=20, Δ=9, SNR=15dB")
    print("Jmin_LE =", Jmin_le)

    # ----------------------------
    # (b2) SWF-DFE, N1=15, N2=5, Δ=1 at SNR=15dB
    # ----------------------------
    w_dfe, Jmin_dfe = swf_dfe_wopt_Jmin(h, sigma_I2, sigma_v2_15, N1=15, N2=5, Delta=1)
    w_ff = w_dfe[:15]
    w_fb = w_dfe[15:]
    print("\n(b2) SWF-DFE: N1=15, N2=5, Δ=1, SNR=15dB")
    print("Jmin_DFE =", Jmin_dfe)
    print("Compare: DFE should typically have lower Jmin than LE on strong post-cursor ISI.")

    # ----------------------------
    # (b3) SER curves for LE and DFE, 0..20 dB in 2 dB steps
    # ----------------------------
    snr_grid = np.arange(0, 22, 2)
    K = 50000

    ser_le_list = []
    ser_dfe_list = []

    for snr_db in snr_grid:
        I = pam2_generate(K + 80, rng)
        r_sig = apply_channel(I, h)
        r, sigma_v2 = add_awgn(r_sig, snr_db, rng)

        # LE uses fixed N=20, Δ=9 as in (b1)
        w_le_i, _ = swf_le_wopt_Jmin(h, sigma_I2, sigma_v2, N=20, Delta=9)
        ser1 = ser_le(I, r, w_le_i, N=20, Delta=9)
        ser_le_list.append(ser1)

        # DFE uses fixed N1=15,N2=5,Δ=1 as in (b2)
        w_dfe_i, _ = swf_dfe_wopt_Jmin(h, sigma_I2, sigma_v2, N1=15, N2=5, Delta=1)
        w_ff_i = w_dfe_i[:15]
        w_fb_i = w_dfe_i[15:]
        ser2 = ser_dfe(I, r, w_ff_i, w_fb_i, N1=15, N2=5, Delta=1)
        ser_dfe_list.append(ser2)

    ser_le_arr = np.array(ser_le_list)
    ser_dfe_arr = np.array(ser_dfe_list)

    # ----------------------------
    # (b4) Viterbi (MLSE) SER curve (uses 50,004 symbols, delay δ=30)
    # ----------------------------
    K_va = 50004
    delta = 30
    ser_va_list = []

    for snr_db in snr_grid:
        I = pam2_generate(K_va + 80, rng)
        r_sig = apply_channel(I, h)
        r, _ = add_awgn(r_sig, snr_db, rng)
        ser_va = viterbi_mlse_2pam(I, r, h, delta=delta)
        ser_va_list.append(ser_va)

    ser_va_arr = np.array(ser_va_list)

    # ----------------------------
    # Fig 2: log10(SER) vs SNR(dB)
    # ----------------------------
    eps = 1e-12
    plt.figure()
    plt.plot(snr_grid, np.log10(ser_le_arr + eps), marker="o", label="SWF-LE (N=20, Δ=9)")
    plt.plot(snr_grid, np.log10(ser_dfe_arr + eps), marker="s", label="SWF-DFE (N1=15,N2=5,Δ=1)")
    plt.plot(snr_grid, np.log10(ser_va_arr + eps), marker="^", label=f"Viterbi MLSE (δ={delta})")
    plt.xlabel("SNR (dB)")
    plt.ylabel("log10(SER)")
    plt.grid(True)
    plt.legend()
    plt.title("Fig 2: SER vs SNR for LE, DFE, and VA (2-PAM)")
    plt.savefig("figures/fig2_ser_vs_snr_part_b.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("\nSaved: figures/fig2_ser_vs_snr_part_b.png")


if __name__ == "__main__":
    main()
