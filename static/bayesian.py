import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis, euclidean
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

# =============================================================================
# 1. GOLDEN OPERATOR (O_n) - Spacetime Modulation
# =============================================================================
class GoldenOperator:
    """Modulates the simulation space using the Golden Ratio (phi)."""
    PHI = (1 + np.sqrt(5)) / 2

    @staticmethod
    def modulate(n: float) -> float:
        """O_n = cos(pi * n) * cos(pi * phi * n)"""
        return np.cos(np.pi * n) * np.cos(np.pi * GoldenOperator.PHI * n)

# =============================================================================
# 2. INFORMATION TOPOLOGY (Isomorphism Level 3)
# =============================================================================
class InformationTopology:
    """Rigorous physical mapping of information metrics."""

    @staticmethod
    def shannon_entropy(rho: np.ndarray) -> float:
        """Shannon entropy as a measure of information density (rho)."""
        probabilities, counts = np.unique(rho, return_counts=True)
        p = counts / len(rho)
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    @staticmethod
    def calculate_directional_cosines(entropy: float, energy: float) -> Tuple[float, float, float]:
        """Normalized directional cosines for state visualization."""
        eps = 1e-6
        ent = max(entropy, eps)
        eng = max(energy, eps)
        magnitude = np.sqrt(ent**2 + eng**2 + 1)
        return ent / magnitude, eng / magnitude, 1 / magnitude

    @staticmethod
    def informational_reynolds(v: float, rho: float, mu: float = 1e-3) -> float:
        """
        Re_psi = (rho * v) / mu
        Inertia (information volume * flow) / Viscosity (latency/friction).
        """
        return (rho * v) / (mu + 1e-9)

# =============================================================================
# 3. METRIPLECTIC DYNAMICS ({u, H} + [u, S])
# =============================================================================
class MetriplecticSystem:
    """
    Core Metriplectic engine: Conservative (Symplectic) + Dissipative (Metric).
    Standard Nomenclature:
    psi (field), rho (density), v (flow velocity).
    """
    def __init__(self, psi_init: float, h_const: float, s_const: float):
        self.psi = psi_init
        self.h = h_const
        self.s = s_const
        self.history = {"psi": [], "symp": [], "metr": []}

    def compute_lagrangian(self) -> Tuple[float, float]:
        """Returns L_symp (conservative) and L_metr (dissipative)."""
        l_symp = 0.5 * self.psi**2 - self.h
        l_metr = -self.s * np.log(self.psi + 1e-9)
        return l_symp, l_metr

    def step(self, dt: float = 0.01):
        """d_psi = {psi, H} + [psi, S]"""
        # Symplectic component: d_symp = {psi, H} (Oscillatory/Reversible)
        d_symp = -np.sin(self.psi * np.pi) * self.h
        
        # Metric component: d_metr = [psi, S] (Dissipative/Irreversible)
        d_metr = -self.s * (self.psi - 0.5) 
        
        self.psi += (d_symp + d_metr) * dt
        
        self.history["psi"].append(self.psi)
        self.history["symp"].append(d_symp)
        self.history["metr"].append(d_metr)

    def plot_diagnostics(self):
        """Visualizes the competition between conservative and dissipative terms."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["symp"], label="Symplectic {u, H} (Conservative)", alpha=0.7)
        plt.plot(self.history["metr"], label="Metric [u, S] (Dissipative)", alpha=0.7)
        plt.title("Metriplectic Diagnostics: Conservative vs. Dissipative Forces")
        plt.xlabel("Step")
        plt.ylabel("Force Magnitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# =============================================================================
# 4. ENERGY ORACLE & BAYESIAN INFERENCE
# =============================================================================
@dataclass
class BayesConfig:
    epsilon: float = 1e-6
    entropy_threshold: float = 0.8
    coherence_threshold: float = 0.6
    action_threshold: float = 0.5

class BayesianInference:
    def __init__(self, config: Optional[BayesConfig] = None):
        self.config = config or BayesConfig()

    def update_posterior(self, prior: float, likelihood: float, evidence: float) -> float:
        evidence = max(evidence, self.config.epsilon)
        return (likelihood * prior) / evidence

    def analyze_generalization(self, entropy: float, coherence: float, energy: float) -> Dict:
        """Evaluates Memorization vs Generalization via Bayesian logic."""
        # Prior for high entropy (generalization tendency)
        prior_ent = 0.1 + 0.4 * (1 / (1 + np.exp(12 * (entropy - 0.75))))
        
        # Posterior calculation
        cond_prob = 0.3 + 0.5 * energy if entropy > self.config.entropy_threshold else 0.15
        posterior = self.update_posterior(prior_ent, cond_prob, coherence)
        
        decision = 1 if posterior > self.config.action_threshold else 0
        return {
            "is_generalizing": bool(decision),
            "posterior": float(posterior),
            "prior_entropy": float(prior_ent)
        }

class EnergyOracle:
    """Momentum-Energy mapping under H7 Conservation."""
    H7_CONSTANT = 7

    def __init__(self, target_norm: float = 0.1024):
        self.target_norm = target_norm

    def get_energy(self, momentum: int) -> float:
        """Maps momentum to energy modulated by Golden Operator."""
        raw_energy = self.target_norm * (momentum / 6.0)**1.2
        return raw_energy * (1.0 + 0.1 * GoldenOperator.modulate(momentum))

    @staticmethod
    def get_partner(state: int) -> int:
        """H7 Symmetry: state ^ 7"""
        return EnergyOracle.H7_CONSTANT ^ state

# =============================================================================
# 5. UNIFIED METRIPLECTIC EVALUATOR
# =============================================================================
class MetriplecticEvaluator:
    """Unified framework for LLM response evaluation."""

    def __init__(self):
        self.topology = InformationTopology()
        self.inference = BayesianInference()
        self.oracle = EnergyOracle()

    def evaluate(self, generated_p: List[int], train_p: Optional[List[int]] = None) -> Dict:
        if train_p is None:
            train_p = list(range(1, 7))

        # 1. State Space Construction (psi)
        train_states = []
        gen_states = []
        for p in train_p:
            energy = self.oracle.get_energy(p)
            ent = self.topology.shannon_entropy(np.array([p]))
            cx, cy, _ = self.topology.calculate_directional_cosines(ent, energy)
            train_states.append([energy, cx, cy])
        
        for p in generated_p:
            energy = self.oracle.get_energy(p)
            ent = self.topology.shannon_entropy(np.array([p]))
            cx, cy, _ = self.topology.calculate_directional_cosines(ent, energy)
            gen_states.append([energy, cx, cy])

        train_states = np.array(train_states)
        gen_states = np.array(gen_states)

        # 2. Distance Analysis (Dual Distance)
        mean_train = np.mean(train_states, axis=0)
        cov_train = np.cov(train_states.T) + 1e-6 * np.eye(3)
        inv_cov = np.linalg.inv(cov_train)
        
        d_maha = mahalanobis(gen_states[0], mean_train, inv_cov)
        d_euc = euclidean(gen_states[0], train_states[0])

        # 3. Bayesian Decision
        entropy_val = self.topology.shannon_entropy(np.array(generated_p))
        coherence = 1.0 - np.clip(d_maha / 5.0, 0.0, 1.0)
        avg_energy = np.mean([self.oracle.get_energy(p) for p in generated_p])
        
        decision = self.inference.analyze_generalization(entropy_val, coherence, avg_energy)

        # 4. Metriplectic Evolution (Simulation)
        system = MetriplecticSystem(psi_init=float(avg_energy), h_const=0.1, s_const=0.05)
        for _ in range(100):
            system.step()

        return {
            "status": "GENERALIZATION" if decision["is_generalizing"] else "MEMORIZATION",
            "metrics": {
                "mahalanobis": float(d_maha),
                "euclidean": float(d_euc),
                "shannon_entropy": float(entropy_val),
                "reynolds": self.topology.informational_reynolds(v=float(d_euc), rho=float(avg_energy))
            },
            "system_state": {
                "final_psi": float(system.psi),
                "lagrangian": system.compute_lagrangian()
            },
            "h7_pairs": {i: self.oracle.get_partner(i) for i in range(8)}
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    evaluator = MetriplecticEvaluator()
    
    print("--- Evaluating Generalization Case ---")
    results = evaluator.evaluate([2, 5, 1, 6])
    print(f"Status: {results['status']}")
    print(f"Reynolds Number: {results['metrics']['reynolds']:.4f}")
    print(f"Final Psi: {results['system_state']['final_psi']:.4f}")
    
    # Simulate and show plots
    sys = MetriplecticSystem(psi_init=0.1, h_const=0.1, s_const=0.02)
    for _ in range(500):
        sys.step()
    # sys.plot_diagnostics() # Uncomment to see plot