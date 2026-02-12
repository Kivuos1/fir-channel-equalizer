import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

# Unnormalized 6-tap channel (as given)
g = np.array([1.0, -0.95, 0.5, 0.15, -0.2, -0.1], dtype=float)

# (2.1) Unit-energy normalization: sum (g/C)^2 = 1
C = np.sqrt(np.sum(g**2))          # = sqrt(2.225) ≈ 1.491643389
f = g / C                          # normalized taps, length 6

print("C =", C)
print("Normalized taps f =", f)
print("Tap energy =", np.sum(f**2))  # should be 1 (up to floating error)

def gen_2pam_with_tail(N_unknown=100_000, tail_len=5, tail_symbol=+1):
    # 2-PAM: -1/+1 equally likely
    data = np.random.choice([-1.0, +1.0], size=N_unknown)
    tail = np.full(tail_len, float(tail_symbol))
    I = np.concatenate([data, tail])
    return data, I  # data = unknown part, I = full seq including tail

def sigma_v2_from_snr_db(snr_db):
    return 10.0 ** (-snr_db / 10.0)

def channel_output_full(I, f, sigma_v2):
    # Linear convolution for FIR channel
    s = np.convolve(I, f, mode="full")[:len(I)]  # keep same length as I
    v = np.sqrt(sigma_v2) * np.random.randn(len(I))
    r = s + v
    return r

np.random.seed(0)

data, I = gen_2pam_with_tail(N_unknown=100_000, tail_len=5, tail_symbol=+1)
snr_db = 12
sigma_v2 = sigma_v2_from_snr_db(snr_db)

r = channel_output_full(I, f, sigma_v2)
print("Generated r of length:", len(r))


def build_state_tables_binary(m=5):
    """
    Binary alphabet (M=2).
    State is the last m symbol indices: (i(k-1), i(k-2), ..., i(k-m)), each in {0,1}.
    Encode in base-2: state_id = i1*2^(m-1) + i2*2^(m-2) + ... + im*2^0
    Next state when new symbol a arrives: (a, i1, i2, ..., i(m-1))
    """
    M = 2
    nstates = M**m

    # decode state_id -> tuple (i1..im)
    states = np.zeros((nstates, m), dtype=np.int8)
    for sid in range(nstates):
        x = sid
        for j in range(m-1, -1, -1):
            states[sid, j] = x & 1
            x >>= 1

    # next_state[sid, a]
    next_state = np.zeros((nstates, M), dtype=np.int16)
    for sid in range(nstates):
        # current tuple: (i1..im) = states[sid]
        i = states[sid]
        for a in range(M):
            new = np.empty(m, dtype=np.int8)
            new[0] = a
            new[1:] = i[:-1]
            # encode
            ns = 0
            for j in range(m):
                ns = (ns << 1) | int(new[j])
            next_state[sid, a] = ns

    return states, next_state


def viterbi_mlse_L6_binary(r, f, delta=30, tail_symbol=+1):
    """
    Full MLSE VA for L=6, binary alphabet.
    r: received sequence length N
    f: normalized 6 taps (length 6)
    delta: traceback length / decision delay
    tail_symbol: +1 or -1 (known tail), determines forced final state
    Returns detected symbol values (not indices), length N.
    """
    assert len(f) == 6, "This VA is for L=6."
    sym_vals = np.array([-1.0, +1.0])
    tail_idx = 1 if tail_symbol == +1 else 0

    m = 5
    M = 2
    nstates = 2**m
    N = len(r)

    states, next_state = build_state_tables_binary(m=m)

    INF = 1e100
    pm = np.zeros(nstates, dtype=float)  # unknown start -> all equal metric 0

    # Survivors
    pred_state = np.zeros((N, nstates), dtype=np.int16)
    pred_sym   = np.zeros((N, nstates), dtype=np.int8)

    # Track best state at each time (for finite traceback decisions)
    best_state_at_k = np.zeros(N, dtype=np.int16)

    f0, f1, f2, f3, f4, f5 = f

    # Forward VA
    for k in range(N):
        new_pm = np.full(nstates, INF, dtype=float)

        for ps in range(nstates):
            # state provides indices for i(k-1..k-5)
            i1, i2, i3, i4, i5 = states[ps]
            x1 = sym_vals[i1]
            x2 = sym_vals[i2]
            x3 = sym_vals[i3]
            x4 = sym_vals[i4]
            x5 = sym_vals[i5]

            base = f1*x1 + f2*x2 + f3*x3 + f4*x4 + f5*x5

            for a in (0, 1):
                ns = next_state[ps, a]
                x0 = sym_vals[a]
                rhat = f0*x0 + base
                bm = (r[k] - rhat) ** 2
                cand = pm[ps] + bm

                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    pred_state[k, ns] = ps
                    pred_sym[k, ns] = a

        pm = new_pm
        best_state_at_k[k] = int(np.argmin(pm))

    # Force termination to known tail state: (tail, tail, tail, tail, tail)
    final_state = 0
    for _ in range(m):
        final_state = (final_state << 1) | tail_idx  # all bits = tail_idx

    # Finite-traceback decisions
    det_idx = np.full(N, -1, dtype=np.int8)

    for k in range(delta, N):
        st = int(best_state_at_k[k])
        # traceback delta steps to decide symbol at time k-delta
        for t in range(k, k - delta, -1):
            st = int(pred_state[t, st])
        # after moving back delta steps in state history, the decided symbol is:
        # easiest is to directly traceback once more step to retrieve symbol at (k-delta)
        # using the state at time (k-delta) best-state:
        det_idx[k - delta] = pred_sym[k - delta, int(best_state_at_k[k - delta])]

    # Fill remaining undecided symbols by full traceback from forced final state
    st = int(final_state)
    for k in range(N - 1, -1, -1):
        a = pred_sym[k, st]
        if det_idx[k] == -1:
            det_idx[k] = a
        st = int(pred_state[k, st])

    return sym_vals[det_idx]

# =======================
# SANITY TEST: Step 2 VA
# =======================
if __name__ == "__main__":
    np.random.seed(42)

    print("\nRunning sanity test for Step 2 (Full MLSE VA)...")

    # Small test so it runs fast
    N_unknown_test = 2000
    tail_len = 5

    data_test, I_test = gen_2pam_with_tail(
        N_unknown=N_unknown_test,
        tail_len=tail_len,
        tail_symbol=+1
    )

    # VERY high SNR → essentially noiseless
    snr_db_test = 40
    sigma_v2_test = sigma_v2_from_snr_db(snr_db_test)

    r_test = channel_output_full(I_test, f, sigma_v2_test)

    delta_test = 30
    I_hat_test = viterbi_mlse_L6_binary(
        r_test, f, delta=delta_test, tail_symbol=+1
    )

    ser_test = np.mean(I_hat_test[:N_unknown_test] != data_test)

    print(f"Test SNR = {snr_db_test} dB")
    print(f"Test SER  = {ser_test:.3e}")

    if ser_test < 1e-3:
        print("✅ Step 2 VA PASSED sanity test (near-zero SER at high SNR).")
    else:
        print("❌ Step 2 VA FAILED sanity test — check trellis / traceback.")


def viterbi_ddfse_L3_binary(r, f_full, delta=30, tail_symbol=+1):
    """
    DDFSE for reduced trellis (first 3 taps), binary alphabet.
    - Trellis memory: 2 -> 2^2 = 4 states
    - Feedback cancels ignored taps f3,f4,f5 using survivor-based estimates per state.
    Returns detected symbol values (±1), length N.
    """
    sym_vals = np.array([-1.0, +1.0])
    tail_idx = 1 if tail_symbol == +1 else 0

    # Full taps (r contains all 6)
    f0, f1, f2, f3, f4, f5 = f_full

    # Reduced trellis memory
    m = 2
    nstates = 2**m
    N = len(r)

    states, next_state = build_state_tables_binary(m=m)

    INF = 1e100
    pm = np.zeros(nstates, dtype=float)  # unknown start -> all equal

    # For each state, store survivor feedback symbols [I(k-3), I(k-4), I(k-5)] as VALUES (can be 0 at start)
    fb_mem = np.zeros((nstates, 3), dtype=float)

    # Survivors for traceback
    pred_state = np.zeros((N, nstates), dtype=np.int16)
    pred_sym   = np.zeros((N, nstates), dtype=np.int8)

    best_state_at_k = np.zeros(N, dtype=np.int16)

    # Forward VA with DDFSE branch metric
    for k in range(N):
        new_pm = np.full(nstates, INF, dtype=float)
        new_fb = np.zeros((nstates, 3), dtype=float)

        for ps in range(nstates):
            i1, i2 = states[ps]          # indices for I(k-1), I(k-2)
            x1 = sym_vals[i1]
            x2 = sym_vals[i2]

            fb0, fb1, fb2 = fb_mem[ps]   # values for I(k-3), I(k-4), I(k-5)

            # Cancel contribution of ignored taps using survivor feedback
            fb_contrib = f3*fb0 + f4*fb1 + f5*fb2
            rprime = r[k] - fb_contrib

            base = f1*x1 + f2*x2

            for a in (0, 1):
                ns = next_state[ps, a]
                x0 = sym_vals[a]

                rhat = f0*x0 + base
                bm = (rprime - rhat) ** 2
                cand = pm[ps] + bm

                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    pred_state[k, ns] = ps
                    pred_sym[k, ns] = a

                    # Update feedback memory for next time:
                    # At time k+1, needed: I(k-2), I(k-3), I(k-4) -> [x2, fb0, fb1]
                    new_fb[ns, 0] = x2
                    new_fb[ns, 1] = fb0
                    new_fb[ns, 2] = fb1

        pm = new_pm
        fb_mem = new_fb
        best_state_at_k[k] = int(np.argmin(pm))

    # Force termination to known reduced final state (last 2 tail symbols)
    final_state = (tail_idx << 1) | tail_idx

    # Finite-traceback decisions
    det_idx = np.full(N, -1, dtype=np.int8)

    for k in range(delta, N):
        st = int(best_state_at_k[k])

        a_dec = None
        for t in range(k, k - delta - 1, -1):
            a = pred_sym[t, st]
            st = int(pred_state[t, st])
            if t == k - delta:
                a_dec = a
                break
        det_idx[k - delta] = a_dec

    # Fill remaining undecided by full traceback from forced final state
    st = int(final_state)
    for k in range(N - 1, -1, -1):
        a = pred_sym[k, st]
        if det_idx[k] == -1:
            det_idx[k] = a
        st = int(pred_state[k, st])

    return sym_vals[det_idx]


def viterbi_truncated_L3_binary(r, f_full, delta=30, tail_symbol=+1):
    """
    Truncated VA: uses only first 3 taps (f0,f1,f2) in the metric,
    but r contains contributions from all 6 taps.
    2-PAM -> 2^2 = 4 states.
    Returns detected symbol values (±1), length N.
    """
    sym_vals = np.array([-1.0, +1.0])
    tail_idx = 1 if tail_symbol == +1 else 0

    # model uses only first 3 taps
    f0, f1, f2 = f_full[0], f_full[1], f_full[2]

    m = 2
    nstates = 2**m
    N = len(r)

    # build state tables for m=2
    states, next_state = build_state_tables_binary(m=m)

    INF = 1e100
    pm = np.zeros(nstates, dtype=float)  # unknown start -> all equal

    pred_state = np.zeros((N, nstates), dtype=np.int16)
    pred_sym   = np.zeros((N, nstates), dtype=np.int8)

    best_state_at_k = np.zeros(N, dtype=np.int16)

    # Forward VA
    for k in range(N):
        new_pm = np.full(nstates, INF, dtype=float)

        for ps in range(nstates):
            # state provides i(k-1), i(k-2)
            i1, i2 = states[ps]
            x1 = sym_vals[i1]
            x2 = sym_vals[i2]

            base = f1*x1 + f2*x2

            for a in (0, 1):
                ns = next_state[ps, a]
                x0 = sym_vals[a]
                rhat = f0*x0 + base           # truncated model prediction
                bm = (r[k] - rhat) ** 2
                cand = pm[ps] + bm

                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    pred_state[k, ns] = ps
                    pred_sym[k, ns] = a

        pm = new_pm
        best_state_at_k[k] = int(np.argmin(pm))

    # Force termination to known reduced final state (last 2 tail symbols)
    # state bits represent (i(k-1), i(k-2)) both should equal tail_idx
    final_state = (tail_idx << 1) | tail_idx

    # Decisions with finite traceback (properly done)
    det_idx = np.full(N, -1, dtype=np.int8)

    for k in range(delta, N):
        st = int(best_state_at_k[k])

        # traceback delta steps and grab the symbol at time (k-delta)
        a_dec = None
        for t in range(k, k - delta - 1, -1):
            a = pred_sym[t, st]
            st = int(pred_state[t, st])
            if t == k - delta:
                a_dec = a
                break
        det_idx[k - delta] = a_dec

    # Fill remaining undecided symbols by full traceback from forced final state
    st = int(final_state)
    for k in range(N - 1, -1, -1):
        a = pred_sym[k, st]
        if det_idx[k] == -1:
            det_idx[k] = a
        st = int(pred_state[k, st])

    return sym_vals[det_idx]

# ===== Fig.2: Full 6-tap VA vs Truncated 3-tap VA =====
np.random.seed(0)

N_unknown = 100_000
tail_len = 5
delta = 30

snr_db_list = np.arange(0, 21, 4)  # 0,4,8,12,16,20

ser_full = []
ser_trunc = []
ser_ddfse = []


for snr_db in snr_db_list:
    data, I = gen_2pam_with_tail(N_unknown=N_unknown, tail_len=tail_len, tail_symbol=+1)
    sigma_v2 = sigma_v2_from_snr_db(snr_db)

    # measurements ALWAYS from full 6-tap channel
    r = channel_output_full(I, f, sigma_v2)

    # (2.2) full MLSE (32-state)
    I_hat_full = viterbi_mlse_L6_binary(r, f, delta=delta, tail_symbol=+1)
    ser_full.append(np.mean(I_hat_full[:N_unknown] != data))

    # (2.3) truncated VA (4-state), metric uses only first 3 taps
    I_hat_trunc = viterbi_truncated_L3_binary(r, f, delta=delta, tail_symbol=+1)
    ser_trunc.append(np.mean(I_hat_trunc[:N_unknown] != data))

    I_hat_ddfse = viterbi_ddfse_L3_binary(r, f, delta=delta, tail_symbol=+1)
    ser_ddfse.append(np.mean(I_hat_ddfse[:N_unknown] != data))

    print(f"SNR={snr_db:2d} dB | SER full={ser_full[-1]:.3e} | SER trunc={ser_trunc[-1]:.3e}")
    print(snr_db, ser_full[-1], ser_trunc[-1], ser_ddfse[-1])


# Plot Fig.2 (partial: 2 curves)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(snr_db_list, np.log10(np.maximum(ser_full, 1e-12)), marker="o", label="Full VA (6 taps, 2^5=32 states)")
plt.plot(snr_db_list, np.log10(np.maximum(ser_trunc, 1e-12)), marker="o", label="Truncated VA (3 taps, 2^2=4 states)")
plt.plot(snr_db_list, np.log10(np.maximum(ser_ddfse, 1e-12)),
         marker="o", label="DDFSE (3 taps + FB for taps 4–6), 2^2=4 states")

plt.xlabel("10log10(SNR) [dB]")
plt.ylabel("log10(SER)")
plt.grid(True, which="both")
plt.legend()
plt.title("Fig.2: SER vs SNR (δ=30)")
plt.savefig("figures/figure2_log10SER_vs_SNR.png", dpi=300, bbox_inches="tight")
plt.show()

