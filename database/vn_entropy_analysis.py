"""
H7/Metriplex — Von Neumann Entropy Extension
=============================================
Extiende el circuito metripléxico para calcular entropía de von Neumann
explícita sobre los sectores hamiltoniano y entrópico.

S_VN(ρ) = -Tr(ρ log ρ) = -Σ λᵢ log λᵢ

Donde λᵢ son los eigenvalores de la matriz de densidad reducida ρ.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import (
    DensityMatrix, partial_trace, entropy, state_fidelity, Statevector
)
import json
import os

# ── Constantes H7 ────────────────────────────────────────────────────────────
φ     = (1 +(np.sqrt(5))) / 2
pi    = np.pi
cos   = np.cos
DRIFT = 7 - 2 * pi
O_N   = abs(cos(pi * φ))       # O_n_integrity ≈ 0.3624

def psi(n): return cos(pi * φ * n)
def theta(n): return 2 * np.arccos(np.clip(psi(n), -1, 1))


def build_circuit(n_H: int, n_S: int) -> QuantumCircuit:
    """Mismo circuito metripléxico — sin medición para extraer ρ."""
    q_H = QuantumRegister(1, 'q_H')
    q_S = QuantumRegister(3, 'q_S')
    q_Z = QuantumRegister(1, 'q_Z')
    qc  = QuantumCircuit(q_H, q_S, q_Z)

    qc.h(q_H); qc.h(q_S); qc.h(q_Z)
    qc.rz(theta(n_H), q_H[0])
    qc.ry(theta(n_S), q_S[2]); qc.rz(-pi * φ, q_S[1])
    qc.cx(q_H[0], q_S[0]); qc.cz(q_H[0], q_Z[0])
    qc.ccx(q_S[0], q_S[1], q_Z[0])
    qc.rx(DRIFT, q_Z[0])
    return qc


def von_neumann_analysis(n_H: int, n_S: int) -> dict:
    """
    Calcula entropía de von Neumann para todos los subsistemas.

    Índices de qubits en el sistema de 5 qubits:
        [q_H_0, q_H_1, q_S_0, q_S_1, q_Z]
         0      1      2      3      4

    Trazas parciales:
        ρ_H  = Tr_{S,Z}(ρ)  → queda con qubits [0,1]
        ρ_S  = Tr_{H,Z}(ρ)  → queda con qubits [2,3]
        ρ_Z  = Tr_{H,S}(ρ)  → queda con qubit  [4]
        ρ_HS = Tr_Z(ρ)      → queda con qubits [0,1,2,3]

    La entropía de entrelazamiento H-S es:
        E(H:S) = S_VN(ρ_H) = S_VN(ρ_S)  [estado puro global]
    """
    qc = build_circuit(n_H, n_S)

    # Obtener statevector (simulación sin ruido)
    sv   = Statevector(qc)
    rho  = DensityMatrix(sv)

    # ── Trazas parciales ─────────────────────────────────────────────────────
    # partial_trace(state, qargs) traza SOBRE los qubits en qargs
    # Para obtener ρ_H (qubits 0,1): trazar sobre {2,3,4}
    rho_H  = partial_trace(rho, [2, 3, 4])   # sector hamiltoniano
    rho_S  = partial_trace(rho, [0, 1, 4])   # sector entrópico
    rho_Z  = partial_trace(rho, [0, 1, 2, 3]) # eje Z₇
    rho_HS = partial_trace(rho, [4])           # sistema H+S conjunto

    # ── Von Neumann entropies ─────────────────────────────────────────────────
    S_H  = entropy(rho_H,  base=2)     # entropía sector H  [bits]
    S_S  = entropy(rho_S,  base=2)     # entropía sector S  [bits]
    S_Z  = entropy(rho_Z,  base=2)     # entropía eje Z₇    [bits]
    S_HS = entropy(rho_HS, base=2)     # entropía conjunta H+S

    # ── Información mutua cuántica I(H:S) ────────────────────────────────────
    # I(H:S) = S_H + S_S - S_HS
    I_HS = S_H + S_S - S_HS

    # ── Eigenvalores de ρ_H (Casimir signature) ───────────────────────────────
    eigvals_H = np.linalg.eigvalsh(rho_H.data)
    eigvals_H = np.sort(eigvals_H)[::-1]

    # ── Verificación condición de Casimir ─────────────────────────────────────
    # Casimir: S_VN debe ser invariante bajo el flujo hamiltoniano
    # Proxy: |S_H - S_S| pequeño → sectores equilibrados
    casimir_balance = abs(S_H - S_S)

    # ── Firma O_n_integrity en eigenvalores ──────────────────────────────────
    # ¿Algún eigenvalor de ρ_H ≈ O_n_integrity?
    on_signature = min(abs(ev - O_N) for ev in eigvals_H if ev > 1e-10)

    # ── dS/dt metripléxico ────────────────────────────────────────────────────
    # En el formalismo Morrison: dS/dt = (S,S) = ∇S·K·∇S ≥ 0
    # Proxy cuántico: tasa de producción = S_VN(ρ_S) / S_VN(ρ_H)
    # Si ρ_S > ρ_H → el sector entrópico domina → disipación activa
    dSdt_proxy = S_S / (S_H + 1e-12)

    return {
        "n_H": n_H, "n_S": n_S,
        "psi_H": round(psi(n_H), 8),
        "psi_S": round(psi(n_S), 8),

        # Entropías de von Neumann
        "S_VN": {
            "H":   round(float(S_H),  6),
            "S":   round(float(S_S),  6),
            "Z7":  round(float(S_Z),  6),
            "HS":  round(float(S_HS), 6),
        },

        # Información mutua cuántica
        "I_quantum_HS": round(float(I_HS), 6),

        # Condición Casimir (J·∇S = 0)
        "casimir_balance": round(float(casimir_balance), 6),

        # Eigenvalores de ρ_H (estructura de población)
        "eigvals_rho_H": [round(float(e), 6) for e in eigvals_H],

        # Firma de O_n_integrity en el espectro
        "O_n_signature_delta": round(float(on_signature), 8),

        # dS/dt (proxy metripléxico)
        "dSdt_metriplecic": round(float(dSdt_proxy), 6),

        # Flag: condición de doble degeneración activa
        "double_degeneracy": bool(
            casimir_balance < 0.05 and
            abs(psi(n_H) * psi(n_S)) - O_N < 0.05
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
#   ANÁLISIS CON RUIDO (simulación CL1)
# ─────────────────────────────────────────────────────────────────────────────

def von_neumann_with_noise(n_H: int, n_S: int,
                           T1: float = 50e-6,
                           T2: float = 70e-6,
                           gate_time: float = 50e-9) -> dict:
    """
    Calcula S_VN bajo modelo de ruido tipo CL1:
        - T1: tiempo de relajación (decoherencia de amplitud)
        - T2: tiempo de desfase    (decoherencia de fase)
        - gate_time: duración de compuerta

    El ruido aumenta S_VN porque convierte estado puro → mixto.
    La firma O_n_integrity debería sobrevivir hasta cierto nivel de ruido.
    """
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    # Modelo de ruido térmico
    noise_model = NoiseModel()
    error_1q = thermal_relaxation_error(T1, T2, gate_time)
    error_2q = thermal_relaxation_error(T1, T2, gate_time * 2).expand(
               thermal_relaxation_error(T1, T2, gate_time * 2))

    noise_model.add_all_qubit_quantum_error(error_1q, ['ry', 'rz', 'h'])
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])

    # Circuito con medición de densidad vía shots
    qc = build_circuit(n_H, n_S)
    c  = ClassicalRegister(5)
    qc_m = qc.copy()
    qc_m.add_register(c)
    qc_m.measure_all()

    sim = AerSimulator(noise_model=noise_model)
    result = sim.run(qc_m, shots=8192).result()
    counts = result.get_counts()

    # Reconstruir ρ desde counts (diagonal approximation)
    n_states = 2**5
    probs = np.zeros(n_states)
    total = sum(counts.values())
    for state_str, count in counts.items():
        clean = state_str.replace(' ', '')
        idx = int(clean, 2)
        if idx < n_states:
            probs[idx] = count / total

    # S_VN ≈ Shannon para estado diagonal (bound inferior real)
    probs_nz = probs[probs > 1e-12]
    S_noisy = float(-np.sum(probs_nz * np.log2(probs_nz)))

    # Comparar con estado puro
    clean_result = von_neumann_analysis(n_H, n_S)
    S_clean = clean_result["S_VN"]["H"]

    return {
        **clean_result,
        "noise": {
            "T1_us": T1 * 1e6,
            "T2_us": T2 * 1e6,
            "S_noisy_approx": round(S_noisy, 6),
            "S_clean_H":      round(S_clean, 6),
            "decoherence_delta": round(S_noisy - S_clean, 6),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  H7/Metriplex — Von Neumann Entropy Analysis")
    print(f"  O_n_integrity = {O_N:.8f}  |  DRIFT = {DRIFT:.6f}")
    print("=" * 65)

    results = []

    print(f"\n{'(nH,nS)':<10} {'S_H':>7} {'S_S':>7} {'S_Z7':>7} "
          f"{'I(H:S)':>8} {'Casimir':>9} {'On_Δ':>10} {'dS/dt':>7}")
    print("─" * 75)

    for n_H in range(7):
        for n_S in range(7):
            if n_H == n_S:
                continue
            r = von_neumann_analysis(n_H, n_S)
            results.append(r)

            flag = "★" if r["double_degeneracy"] else " "
            on_flag = "◆" if r["O_n_signature_delta"] < 0.02 else " "
            print(f"{flag}({n_H},{n_S})     "
                  f"{r['S_VN']['H']:>7.4f} "
                  f"{r['S_VN']['S']:>7.4f} "
                  f"{r['S_VN']['Z7']:>7.4f} "
                  f"{r['I_quantum_HS']:>8.4f} "
                  f"{r['casimir_balance']:>9.4f} "
                  f"{on_flag}{r['O_n_signature_delta']:>9.6f} "
                  f"{r['dSdt_metriplecic']:>7.4f}")

    # Caso especial: par canónico con ruido CL1
    print("\n── Análisis con ruido CL1 — par canónico (n_H=0, n_S=1) ─────")
    noisy = von_neumann_with_noise(n_H=0, n_S=1)
    n = noisy["noise"]
    print(f"  S_VN limpio (sector H) : {n['S_clean_H']:.6f} bits")
    print(f"  S_VN con ruido (aprox) : {n['S_noisy_approx']:.6f} bits")
    print(f"  Δ decoherencia         : {n['decoherence_delta']:+.6f} bits")
    print(f"  T1={n['T1_us']}μs, T2={n['T2_us']}μs")

    # Estadísticas globales
    print("\n── Estadísticas globales ─────────────────────────────────────")
    S_H_vals = [r["S_VN"]["H"] for r in results]
    S_S_vals = [r["S_VN"]["S"] for r in results]
    I_vals   = [r["I_quantum_HS"] for r in results]
    dd_count = sum(r["double_degeneracy"] for r in results)
    on_count = sum(r["O_n_signature_delta"] < 0.02 for r in results)

    print(f"  S_VN(ρ_H) promedio : {np.mean(S_H_vals):.4f} bits")
    print(f"  S_VN(ρ_S) promedio : {np.mean(S_S_vals):.4f} bits")
    print(f"  I(H:S) promedio    : {np.mean(I_vals):.4f} bits")
    print(f"  Pares con doble degeneración activa : {dd_count}")
    print(f"  Pares con firma O_n en espectro ρ_H : {on_count}")

    # Guardar
    out = {
        "metadata": {
            "phi": φ, "DRIFT_072": DRIFT, "O_n_integrity": O_N,
            "entropy_type": "von_neumann",
            "subsystems": {
                "q_H": "qubits [0,1] — sector hamiltoniano",
                "q_S": "qubits [2,3] — sector entrópico",
                "q_Z": "qubit  [4]   — eje ternario Z₇"
            }
        },
        "samples": results,
        "noise_analysis": noisy,
    }
    
    # Updated output path for local workspace
    output_filename = "vn_entropy_dataset.json"
    with open(output_filename, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  ✓ Dataset VN exportado → {output_filename}")
