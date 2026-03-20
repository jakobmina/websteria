from qiskit_aer import Aer
import qiskit
from qiskit.visualization import plot_histogram
from templates.script import qc_transpiled
from matplotlib.pyplot import plot as plt
import numpy as np
from typing import Callable
from math import log 
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import minkowski
from qiskit.quantum_info import Statevector
from scipy.spatial.distance import mahalanobis
# Crear un circuito cuántico con 3 qubits (2 qubits de datos y 1 qubit auxiliar)
qc = qiskit.QuantumCircuit(3, 3)
# angulos de rotacion
phi = (1+np.sqrt(5))/2 # azimutal
pi = np.pi # axioma vertical 
cos = np.cos # axioma horizontal
n=np.arange(1,6)
# Inicializar el circuito
qc.h(0)
qc.h(1)
qc.h(2)

qc.ccx(1, 2, 0)  # Estado inicial del primer qubit
qc.y(2)  # Estado inicial del segundo qubit
qc.rz(2*pi*phi**2, 0)
qc.ry(2*pi*phi**3, 1)
qc.rx(2*pi*phi**4, 2)

# Aplicar puertas Toffoli y X en un bucle
for i in n:  # Número de iteraciones
    # Aplicar la puerta Toffoli (control: qubit 0 y 1, objetivo: qubit 2)
    qc.ccx(0, 2, 1)  # Toffoli
    # Aplicar la puerta X al qubit auxiliar
    qc.cswap(0, 1, 2)  # Invertir el estado del qubit auxiliar

# Medir todos los qubits
qc.measure_all()

## Crear un sampler para ejecutar el circuito
sampler = Aer.get_backend('qasm_simulator')
compiled_circuit = qiskit.transpile(qc, sampler)
result = sampler.run(compiled_circuit, shots=1024).result()
counts = result.get_counts(compiled_circuit)
# Ejemplo de integración de la función de onda (usando la probabilidad del estado '000')
prob_000 = counts.get('000', 0) / sum(counts.values())  # Probabilidad de medir '000'

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) * prob_000  # Modulamos la onda con la probabilidad
compiled_circuit = qiskit.transpile(qc, sampler)
result = sampler.run(compiled_circuit, shots=1024).result()


# Obtener y mostrar los resultados
counts = result.get_counts(qc)
# Visualizar los resultados cuánticos con un histograma

print("Resultados de la simulación:")
print(qc_transpiled)
print(counts)
plot_histogram(counts)

"""representacion basica del codigo del circuito base"""
 
# --- 1. Qubit State Representation ---

def initialize_node():
    """
    Initializes a node with random complex amplitudes for the qubit states.
    Returns a 3x3 matrix representing the qubit states.
    """
    node = np.zeros((3, 3), dtype=complex)
    # Exposed states (top-left 2x2 submatrix)
    for i in range(2):
        for j in range(2):
            node[i, j] = complex(np.random.random(), np.random.random())
    node[2, 2] = complex(np.random.random(), np.random.random())  # Hidden state
    return node


# --- 2. Activation Logic ---

def is_active(node, threshold=0.5):
    """
    Checks if a node is active based on the probability of exposed states.
    Args:
        node: The node's state matrix.
        threshold: The activation threshold.
    Returns: True if the node is active, False otherwise.
    """
    prob_exposed = sum(abs(node[i, j]) ** 2 for i in range(2) for j in range(2))
    return prob_exposed.real + node[2, 2].real > threshold


def activate_node_with_ccx(node):
    """Transforms the node's matrix when activated using CCX (Toffoli) gate logic."""
    new_node = node.copy()

    # Example: Use CCX to activate a hidden state based on exposed states
    # Assuming qubit 0 and 1 are the exposed states, and qubit 2 is the hidden state
    qc.ccx(0, 1, 2)  # CCX gate: if qubit 0 and 1 are active, activate qubit 2
    #  (In a real implementation, you'd update new_node based on the CCX gate's effect)
    #  This is a placeholder;  the CCX gate's effect on the node matrix needs to be defined.

    return new_node


# use Mahalanobis distance

# define the average
data = np.random.rand(10, 2)
# Calcular la matriz de covarianza
covariance_matrix = np.cov(data.T)

# Calcular la inversa de la matriz de covarianza (necesaria para la distancia de Mahalanobis)
try:
    inverse_covariance = np.linalg.inv(covariance_matrix)
except np.linalg.LinAlgError:
    inverse_covariance = np.eye(2)  # Handle singular matrix

# Nueva acción (representada como un tensor)
action = np.array([2.0, 3.0])

# Calcular la distancia de Mahalanobis (simplificado)
mahalanobis_distance = np.sqrt(np.dot(np.dot((action[:2] - np.pi/21), inverse_covariance), (action[:2] - np.pi/21)))

def neighbor_activation(node, active_neighbors):
    if is_active(node):  # Existing state is active
        if active_neighbors < 2 or active_neighbors > 3:
            return node  # Deactivate (in this example, do nothing)
        else:
            return node  # Remains active (no change needed in this simple example)
    else:  # Existing state is not active
        if active_neighbors == 3:
            return activate_node_with_ccx(node)  # Birth/Activation using CCX
        else:
            return node  # Remains Inactive


# --- 3. Neighborhood Logic ---

def calculate_neighbors(network, layer_index, node_index, p=2):
    """
    Calculates the number of active neighbors for a given node.
    Args:
        network: The network of nodes.
        layer_index: The index of the layer containing the node.
        node_index: The index of the node within its layer.
        p: The order of the Minkowski distance (default is 2 for Euclidean distance).
    Returns: The number of active neighbors.
    """

    active_neighbors = 0
    current_node = network[layer_index][node_index]

    # Iterate through previous and next layers (adjust as needed for your connectivity rules)
    for i in [layer_index - 1, layer_index + 1]:
        if 0 <= i < len(network):  # Check layer bounds
            for neighbor_node in network[i]:
                distance = minkowski(current_node.flatten(), neighbor_node.flatten(), p=p)
                if distance < 0.7:  # Example distance threshold for neighborhood – needs tuning.
                    if is_active(neighbor_node):
                        active_neighbors += 1

    return active_neighbors


 
class MetriplecticSystem4Bracket:
    """
    Morrison's 4-bracket metriplectic structure.
    
    Ecuación de movimiento:
        ẋ = J·∇H + K·∇S
    
    Donde:
        J = tensor de Poisson (antisimétrico)
        K = tensor métrico (simétrico, semi-def. positivo)
    
    Condiciones de degeneración:
        J·∇S = 0   (S es Casimir de Poisson)
        K·∇H = 0   (H en kernel métrico)
    """

    def __init__(self, dim: int, J: np.ndarray, K: np.ndarray):
        self.dim = dim
        self.J = J  # Poisson tensor [antisimétrico]
        self.K = K  # Metric tensor  [simétrico, ≥0]
        self._validate_tensors()

    def _validate_tensors(self):
        assert np.allclose(self.J, -self.J.T), "J debe ser antisimétrico"
        assert np.allclose(self.K, self.K.T),  "K debe ser simétrico"
        eigvals = np.linalg.eigvalsh(self.K)
        assert np.all(eigvals >= -1e-10),       "K debe ser semi-definido positivo"

    # ── Brackets fundamentales ──────────────────────────────────────────────

    def poisson_bracket(self, F: Callable, G: Callable, x: np.ndarray,
                        eps: float = 1e-6) -> float:
        """
        {F, G} = ∇F · J · ∇G   [antisimétrico]
        """
        grad_F = self._gradient(F, x, eps)
        grad_G = self._gradient(G, x, eps)
        return float(grad_F @ self.J @ grad_G)

    def metric_bracket(self, F: Callable, G: Callable, x: np.ndarray,
                       eps: float = 1e-6) -> float:
        """
        (F, G) = ∇F · K · ∇G   [simétrico]
        """
        grad_F = self._gradient(F, x, eps)
        grad_G = self._gradient(G, x, eps)
        return float(grad_F @ self.K @ grad_G)

    def four_bracket(self, F: Callable, H: Callable, S: Callable,
                     x: np.ndarray, eps: float = 1e-6) -> float:
        """
        4-bracket metripléxico [F; H, S]:
            [F; H, S] = {F, H} + (F, S)
        
        Este es el bracket de Morrison que unifica la dinámica
        reversible (Poisson) con la irreversible (métrica/entropía).
        """
        pb = self.poisson_bracket(F, H, x, eps)
        mb = self.metric_bracket(F, S, x, eps)
        return pb + mb

    # ── Evolución temporal ──────────────────────────────────────────────────

    def equations_of_motion(self, H: Callable, S: Callable,
                             x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        ẋ = J·∇H + K·∇S
        """
        grad_H = self._gradient(H, x, eps)
        grad_S = self._gradient(S, x, eps)
        return self.J @ grad_H + self.K @ grad_S

    def integrate(self, H: Callable, S: Callable, x0: np.ndarray,
                  dt: float = 0.01, steps: int = 1000) -> np.ndarray:
        """
        Integrador RK4 con monitoreo de degeneración doble.
        """
        trajectory = np.zeros((steps + 1, self.dim))
        trajectory[0] = x0.copy()
        x = x0.copy()

        for i in range(steps):
            k1 = self.equations_of_motion(H, S, x, dt * 0.1)
            k2 = self.equations_of_motion(H, S, x + 0.5 * dt * k1, dt * 0.1)
            k3 = self.equations_of_motion(H, S, x + 0.5 * dt * k2, dt * 0.1)
            k4 = self.equations_of_motion(H, S, x + dt * k3, dt * 0.1)
            x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            trajectory[i + 1] = x

        return trajectory

    # ── Verificación de condiciones de degeneración ─────────────────────────

    def verify_degeneracy(self, H: Callable, S: Callable,
                          x: np.ndarray, eps: float = 1e-6) -> dict:
        """
        Verifica las condiciones de doble degeneración de Morrison:
            J·∇S ≈ 0   (S Casimir de Poisson)
            K·∇H ≈ 0   (H en kernel métrico)
        """
        grad_H = self._gradient(H, x, eps)
        grad_S = self._gradient(S, x, eps)

        J_grad_S = self.J @ grad_S   # debe → 0
        K_grad_H = self.K @ grad_H   # debe → 0

        return {
            "J·∇S":        J_grad_S,
            "K·∇H":        K_grad_H,
            "|J·∇S|":      np.linalg.norm(J_grad_S),
            "|K·∇H|":      np.linalg.norm(K_grad_H),
            "degeneracy_ok": (np.linalg.norm(J_grad_S) < 1e-8 and
                               np.linalg.norm(K_grad_H) < 1e-8)
        }

    def entropy_production_rate(self, S: Callable, x: np.ndarray,
                                eps: float = 1e-6) -> float:
        """
        dS/dt = (S, S) = ∇S · K · ∇S  ≥ 0  [segunda ley]
        """
        return self.metric_bracket(S, S, x, eps)

    # ── Utilidades ──────────────────────────────────────────────────────────

    @staticmethod
    def _gradient(F: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Gradiente numérico por diferencias centradas."""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            grad[i] = (F(xp) - F(xm)) / (2 * eps)
        return grad
    

def shannon_entropy(data, average = 'p'):
    """
    Calculates the Shannon entropy of a dataset.
    Args:
        average: The dataset (can be a list, numpy array, or torch tensor).
    """
    isinstance(data, type)
    if isinstance(data, torch.Tensor):
        data = data.tolist()

    if not isinstance(data, list):
        data = [data]

    value_counts = {}
    for value in data:
        value_counts[value] = value_counts.get(value, 0) + 1
    total_values = len(data)
    probabilities = [count / total_values for count in value_counts.values()]
    entropy = 0
    for p in probabilities:
        if p > 0:
            entropy -= p * log(p, 2)
    return entropy


print(f"Entropía de Shannon:, {shannon_entropy(data.flatten().tolist())}")


def calculate_cosines(entropy, env_value):
    """
    Calculates the direction cosines (x, y, z) for a 3D vector.
    Args:
        entropy: The x-component of the vector.
        env_value: The y-component of the vector.
    Returns: A tuple containing the direction cosines (cos_x, cos_y, cos_z).
    Note: The z-component is implicitly set to 1.
    """
    if entropy == 0:
        entropy = 1e-6
    if env_value == 0:
        env_value = 1e-6

    magnitude = np.sqrt(entropy ** 2 + env_value ** 2 + 1)
    cos_x = entropy / magnitude
    cos_y = env_value / magnitude
    cos_z = 1 / magnitude

    return cos_x, cos_y, cos_z


def wave_function(x, t, amplitude=1.0, frequency=1.0, phase=0.0):
    """Defines a simple sinusoidal wave function."""
    return amplitude * np.sin(2 * np.pi * frequency * x - phase * t)  # added time variable


def visualize_wave_and_network(network, iteration, t):  # Added time parameter
    """Visualizes both the network state and the wave function."""

    # 1. Wave Function Plot
    x_wave = np.linspace(0, 10, 500)  # Adjust x range as needed
    y_wave = wave_function(x_wave, t)  # Evaluate wave function

    plt.figure(figsize=(12, 6))  # Larger figure for both plots
    plt.subplot(1, 2, 1)  # Subplot for the wave

    plt.plot(x_wave, y_wave, color='blue', label=f"Wave at t={t:.2f}")  # label added
    plt.xlabel("x")
    plt.ylabel("ψ(x)")
    plt.title("Wave Function")
    plt.grid()
    plt.legend()  # show label

    # 2. Network State Plot
    plt.subplot(1, 2, 2)  # Subplot for the network
    for layer_index, layer in enumerate(network):
        for node_index, node in enumerate(layer):
            if is_active(node):
                plt.scatter(layer_index, node_index, color='red', marker='o')
            # Optionally visualize node state (e.g. color based on activation level)
            # prob_exposed = sum(abs(node[i, j]) ** 2 for i in range(2) for j in range(2))
            # plt.scatter(layer_index, node_index, color=plt.cm.viridis(prob_exposed), marker='o')

    plt.title(f"Network State at Iteration {iteration}")
    plt.xlabel("Layer Index")
    plt.ylabel("Node Index")
    plt.xlim(-1, len(network))
    plt.ylim(-1, max(len(layer) for layer in network))
    plt.grid()

    plt.tight_layout()  # Adjust subplot params for a tight layout
    plt.show()


# --- Simulation and Visualization ---

def visualize_network(network, iteration):
    """
    Visualizes the network state using matplotlib.
    Args:
        network: The network to visualize.
    """
    plt.figure(figsize=(8, 6))
    for layer_index, layer in enumerate(network):
        for node_index, node in enumerate(layer):
            if is_active(node):
                plt.scatter(layer_index, node_index, color='red', marker='o')

    plt.title(f"Network State at Iteration {iteration}")
    plt.xlabel("Layer Index")
    plt.ylabel("Node Index")
    plt.xlim(-1, len(network))
    plt.ylim(-1, max(len(layer) for layer in network))
    plt.grid()
    plt.show()


# Run simulation
network = [[initialize_node() for _ in range(1000)] for i in [2, 3, 2, 2]]
for iteration in range(10):  # Simulate for 10 iterations
    t = iteration + 0.1  # Time variable for wave function
visualize_network(network, iteration=10)  # Visualize the final network state
visualize_wave_and_network(network, t=10, iteration=10)  # added time t.

print(network)
print(qc)

# Inicializar el circuito (puedes personalizar esta parte)
qc.x(1)  # Estado inicial del primer qubit
qc.x(0)  # Estado inicial del segundo qubit

# Aplicar puertas Toffoli y X en un bucle
for _ in range(50):  # Número de iteraciones
    # Aplicar la puerta Toffoli (control: qubit 0 y 1, objetivo: qubit 2)
    qc.ccx(0, 1, 2)  # Toffoli
    # Aplicar la puerta X al qubit auxiliar
    qc.x(2)  # Invertir el estado del qubit auxiliar

# Medir todos los qubits
qc.measure_all()

## Crear un sampler para ejecutar el circuito
sampler = Aer.get_backend('qasm_simulator')

compiled_circuit = qiskit.transpile(qc, sampler)
result = sampler.run(compiled_circuit, shots=1024).result()

# Obtener y mostrar los resultados
counts = result.get_counts(qc)
print("Resultados de la simulación:")
print(qc_transpiled)
print(counts)
"""representacion basica del codigo del circuito base"""
