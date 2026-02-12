# 📡 ISI Channel Equalisation & Sequence Estimation  
### MLSE • Reduced-State Viterbi • DDFSE • LE • DFE (LMMSE)

> A comprehensive simulation study of optimal and sub-optimal detection techniques over FIR ISI channels using Python.

---

## 🚀 Overview

This project implements and compares classical and advanced receiver designs for digital communication systems affected by **Intersymbol Interference (ISI)**.

It includes:

- ✅ Full Maximum Likelihood Sequence Estimation (MLSE – Viterbi)
- ✅ Reduced-State Viterbi Algorithm
- ✅ Delayed Decision Feedback Sequence Estimation (DDFSE)
- ✅ Statistical Wiener Filter Linear Equaliser (LE)
- ✅ Statistical Wiener Filter Decision Feedback Equaliser (DFE)

All methods are evaluated using:

- Symbol Error Rate (SER) vs SNR  
- Convergence analysis  
- Equaliser order & delay tuning  
- Complexity vs performance trade-offs  

---

## 📡 System Model

Discrete-time FIR ISI channel:

r(k) = Σ h_l I(k − l) + v(k)

Where:

- I(k) → 2-PAM / 4-PAM symbols (unit power)
- h_l → channel taps
- v(k) → AWGN with variance σ_v²
- SNR = 1 / σ_v²

---

# 🔹 Channel 1 – 3-Tap ISI (4-PAM)

F(z) = (1/√2)(0.8 − z⁻¹ + 0.6z⁻²)

### Experiments
- 16-state MLSE  
- Traceback study: δ = 5, 10, 20, 60  
- SER vs SNR (0–16 dB)  
- 100,000 symbols per SNR  
- SWF vs TAWF comparison  

### Insights
- Proper delay selection significantly reduces MSE  
- Increasing equaliser order improves ISI mitigation  
- TAWF converges to SWF with increasing snapshots  

---

# 🔹 Channel 2 – 6-Tap ISI (2-PAM)

G(z) = (1/C)(1 − 0.95z⁻¹ + 0.5z⁻² + 0.15z⁻³ − 0.2z⁻⁴ − 0.1z⁻⁵)

C = √2.225 ≈ 1.4916  
Σ h_l² = 1  

### Configuration
- 2-PAM (±1)  
- 100,000 symbols per SNR  
- SNR: 0–20 dB  

---

# 🧠 Implemented Methods

## 🔵 1. Full MLSE (32-State Viterbi)

- Channel memory: 5  
- States: 2⁵ = 32  
- Squared Euclidean metric  
- Finite traceback (δ = 30)  
- Tail termination  

✔ Optimal performance  
❌ Highest complexity  

---

## 🟡 2. Reduced-State Viterbi (4-State)

- Channel truncated to first 3 taps  
- States: 2² = 4  

✔ Lower complexity  
❌ Residual ISI degradation  

---

## 🟢 3. DDFSE (Delayed Decision Feedback)

- 4-state trellis  
- Survivor-based feedback cancellation  
- Corrects ignored taps dynamically  

✔ Near-MLSE performance at high SNR  
✔ Major improvement over truncated VA  

---

# 📊 Equalisation Study (LE vs DFE vs MLSE)

### 🔹 SWF Linear Equaliser
- Order: N = 20  
- Delay: Δ = 9  

### 🔹 SWF Decision Feedback Equaliser
- Feedforward: N₁ = 15  
- Feedback: N₂ = 5  
- Delay: Δ = 1  

### 🔹 MLSE (Viterbi)
- 32 states  
- δ = 30  

---

# 📈 Results

Example output:

figures/fig2_ser_vs_snr_part_b.png

### Observations

- DFE outperforms LE due to better post-cursor cancellation  
- MLSE provides optimal detection  
- Reduced-State VA suffers from model mismatch  
- DDFSE significantly bridges the performance gap  
- Performance gap widens at moderate SNR  

---

# 📂 Project Structure

.
├── part_a_equalisation/  
├── part_b_sequence_detection/  
├── figures/  
├── utils/  
└── README.md  

---

# 🧮 Core Concepts Used

Wiener Filter solution:  
w_opt = R⁻¹p  

Minimum MSE:  
J_min = E[d²] − pᵀ w_opt  

Other techniques:
- Augmented correlation modelling (DFE)  
- Dynamic programming (Viterbi)  
- Survivor traceback techniques  

---

# 🛠️ Tech Stack

- Python  
- NumPy  
- Matplotlib  

No external communication libraries used.

---


