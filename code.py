import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()


def pam4_unit_power():
    # Natural 4-PAM: {-3,-1,+1,+3} has average power 5.
    # Scale to unit power => divide by sqrt(5).
    return np.array([-3, -1, 1, 3], dtype=float) / np.sqrt(5.0)

def build_state_maps(M):
    # states are pairs (prev1, prev2) with indices in [0..M-1]
    # state_id = prev1*M + prev2
    states = [(i, j) for i in range(M) for j in range(M)]
    # next_state given current state (p1,p2) and new symbol a:
    # new state becomes (a, p1)
    next_state = np.zeros((M*M, M), dtype=int)
    for sid, (p1, p2) in enumerate(states):
        for a in range(M):
            ns = a * M + p1
            next_state[sid, a] = ns
    return states, next_state

def viterbi_mlse_finite_tb(r, f, sym_vals, tail_idx_pair, delta):
    """
    r: received samples (length N)
    f: channel taps [f0,f1,f2]
    sym_vals: alphabet values (length M)
    tail_idx_pair: (tail_symbol_index, tail_symbol_index) for last 2 known symbols
                   final state corresponds to (I(N-1), I(N-2)) = (t, t) in indices
    delta: traceback length / decision delay
    Returns: detected symbol indices (length N)
    """
    N = len(r)
    M = len(sym_vals)
    m = 2
    nstates = M**m

    states, next_state = build_state_maps(M)

    # Path metrics
    INF = 1e100
    pm = np.full(nstates, INF)
    pm[:] = 0.0  # unknown start state => all equal

    # Store survivor predecessor and decided symbol for each time and state
    pred_state = np.zeros((N, nstates), dtype=np.int16)
    pred_sym   = np.zeros((N, nstates), dtype=np.int8)

    f0, f1, f2 = f

    for k in range(N):
        new_pm = np.full(nstates, INF)

        # For each prev state, try all symbols
        for ps in range(nstates):
            p1, p2 = states[ps]
            x1 = sym_vals[p1]
            x2 = sym_vals[p2]
            base = f1 * x1 + f2 * x2

            for a in range(M):
                ns = next_state[ps, a]
                x0 = sym_vals[a]
                s_hat = f0 * x0 + base
                bm = (r[k] - s_hat) ** 2  # real-valued here
                cand = pm[ps] + bm

                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    pred_state[k, ns] = ps
                    pred_sym[k, ns] = a

        pm = new_pm

    # Force termination to known final state from tail symbols
    tail1, tail2 = tail_idx_pair  # indices for I(N-1), I(N-2)
    final_state = tail1 * M + tail2

    # Now produce decisions using finite traceback:
    # We'll fill decisions for all times, using:
    # - for early times: online finite-TB
    # - at the end: do one final full traceback from forced final_state
    det = np.full(N, -1, dtype=int)

    # Online decisions up to N-1-delta
    for k in range(delta, N):
        # best state at time k is argmin pm (at the end we only have final pm,
        # but for online we'd need pm at each k; simplest: use stored trellis and
        # do traceback from the best state at time k by reconstructing "best state"
        # using a backward metric isn't stored. So instead we approximate online by
        # using the globally best final state AFTER processing all samples for early
        # decisions is not correct.
        #
        # Correct finite-TB requires best state at time k. We can get it by re-running
        # and storing pm history, but that's cheap for N=1e5.
        break

    # Re-run forward but store pm history to get best-state at each time
    pm = np.full(nstates, INF)
    pm[:] = 0.0
    pm_hist = np.zeros((N, nstates))
    for k in range(N):
        new_pm = np.full(nstates, INF)
        for ps in range(nstates):
            p1, p2 = states[ps]
            x1 = sym_vals[p1]
            x2 = sym_vals[p2]
            base = f1 * x1 + f2 * x2
            for a in range(M):
                ns = next_state[ps, a]
                x0 = sym_vals[a]
                s_hat = f0 * x0 + base
                bm = (r[k] - s_hat) ** 2
                cand = pm[ps] + bm
                if cand < new_pm[ns]:
                    new_pm[ns] = cand
        pm = new_pm
        pm_hist[k, :] = pm

    # Now do finite-TB decisions properly
    for k in range(delta, N):
        best_state_k = int(np.argmin(pm_hist[k, :]))
        st = best_state_k
        # traceback delta steps to land at time k-delta and output its symbol
        for t in range(k, k - delta, -1):
            a = pred_sym[t, st]
            st = pred_state[t, st]
        det[k - delta] = pred_sym[k - delta, best_state_k]

    # Final full traceback from forced final state to fill remaining undecided symbols
    st = final_state
    for k in range(N - 1, -1, -1):
        a = pred_sym[k, st]
        if det[k] == -1:
            det[k] = a
        st = pred_state[k, st]

    return det

def run_sim():
    np.random.seed(0)

    # Channel taps
    f = np.array([0.8, -1.0, 0.6], dtype=float) / np.sqrt(2.0)

    # 4-PAM unit power
    sym_vals = pam4_unit_power()
    M = len(sym_vals)

    N_unknown = 100_000
    tail_len = 2
    N = N_unknown + tail_len

    # Choose a known tail symbol index (e.g., the +1 level -> index 2 in [-3,-1,1,3])
    tail_idx = 2
    tail_idx_pair = (tail_idx, tail_idx)

    # Generate data indices
    data_idx = np.random.randint(0, M, size=N_unknown)
    idx = np.concatenate([data_idx, np.array([tail_idx, tail_idx], dtype=int)])
    I = sym_vals[idx]

    # Build received noiseless signal (assume I(k<0)=0 for simulation start)
    # This start transient affects only first couple symbols; fine for SER over 1e5.
    s = np.zeros(N)
    for k in range(N):
        x0 = I[k]
        x1 = I[k-1] if k-1 >= 0 else 0.0
        x2 = I[k-2] if k-2 >= 0 else 0.0
        s[k] = f[0]*x0 + f[1]*x1 + f[2]*x2

    # Since symbols are unit power and sum |f|^2 = 1, signal power on s is ~1.
    # Use SNR = 1/sigma_v^2 => sigma_v^2 = 1/SNR_linear
    snr_db_list = np.arange(0, 18, 2)
    deltas = [5, 10, 20, 60]

    ser_results = {d: [] for d in deltas}

    for snr_db in snr_db_list:
        snr_lin = 10 ** (snr_db / 10.0)
        sigma_v2 = 1.0 / snr_lin
        v = np.sqrt(sigma_v2) * np.random.randn(N)
        r = s + v

        for d in deltas:
            det_idx = viterbi_mlse_finite_tb(r, f, sym_vals, tail_idx_pair, d)
            # SER over unknown 100,000 symbols only
            errs = np.sum(det_idx[:N_unknown] != data_idx)
            ser = errs / N_unknown
            ser_results[d].append(ser)

        print(f"SNR={snr_db:2d} dB done.")

    # Plot log10(SER) vs SNR(dB)
    plt.figure()
    for d in deltas:
        ser = np.array(ser_results[d])
        # Avoid log10(0) if any
        ser = np.maximum(ser, 1e-12)
        plt.plot(snr_db_list, np.log10(ser), marker='o', label=f"δ={d}")

    plt.xlabel("10log10(SNR) [dB]")
    plt.ylabel("log10(SER)")
    plt.grid(True, which='both')
    plt.legend()
    plt.title("MLSE (Viterbi) over 3-tap ISI channel, 4-PAM")
    plt.savefig("figures/figure1_log10SER_vs_SNR.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_sim()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed/60:.2f} minutes")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


