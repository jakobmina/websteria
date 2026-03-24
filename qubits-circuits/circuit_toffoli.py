import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import minkowski

from static.model import qc


# --- 1. Normalization
def normalize_node(node):
    """
    Normalizes the amplitudes of a node.
    Args:
        node: The node's state matrix.
    Returns: The normalized node.
    """
    # Calcular la norma (suma de los cuadrados de las magnitudes)
    norm = np.linalg.norm(node)  #Efficiently computes the norm
    normalized_node = node / norm if norm > 0 else node
    return normalized_node


def initialize_node():
    """
    Initializes a node with random complex amplitudes for the qubit states.
    Returns a 3x3 matrix representing the qubit states.
    """
    node = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            node[i, j] = complex(np.random.rand(), np.random.rand()) #Initialize with random complex numbers
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
    prob_exposed = np.sum(np.abs(node[:2, :2])**2)
    return prob_exposed > threshold

# Run simulation
def activate_node_with_ccx(node):
    """Applies CCX gate logic."""
    qc.ccx(0, 1, 2)
    return node

#Simplified Mahalanobis distance calculation (assuming 2D for demonstration)
def mahalanobis_distance(x, mean, cov):
    """Computes the Mahalanobis distance."""
    try:
        inv_cov = np.linalg.inv(cov)
        return np.sqrt((x - mean) @ inv_cov @ (x - mean))
    except np.linalg.LinAlgError:
        warnings.warn("Singular covariance matrix. Returning Euclidean distance.")
        return np.linalg.norm(x - mean)

def action_rules(node, active_neighbors):
    """
    Applies  Action rules to determine the next state of a node.
    Args:
        node: The current state of the node.
        active_neighbors: The number of active neighbors.
    Returns: The next state of the node.
    """
    if is_active(node):  # Existing state is active
        if active_neighbors < 2 or active_neighbors > 3:
            return node
    else:  # Existing state is not active
        if active_neighbors == 3:
            return activate_node_with_ccx(node)
        return node


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
                if distance < 1.0:  # Example distance threshold for neighborhood – needs tuning.
                    if is_active(neighbor_node):
                        active_neighbors += 1

    return active_neighbors



def shannon_entropy(data):
    """
    Calculates the Shannon entropy of a dataset.
    Args:
        data: The dataset (list or numpy array).
    """
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = 0
    for p in probabilities:
        entropy -= p * np.log2(p) if p > 0 else 0
    return entropy

#Calculate direction cosines
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
# Aplicar logica bayesiana
class BayesLogic:
    EPSILON = 1e-6
    HIGH_ENTROPY_THRESHOLD = 0.8
    HIGH_COHERENCE_THRESHOLD = 0.6
    ACTION_THRESHOLD = 0.5

def __init__(self):
    self.EPSILON = 1e-6
    self.HIGH_ENTROPY_THRESHOLD = 0.8
    self.HIGH_COHERENCE_THRESHOLD = 0.6
    self.ACTION_THRESHOLD = 0.5

def calculate_posterior_probability(self, prior_a: float, prior_b: float, conditional_b_given_a: float) -> float:
    """Calculate the posterior probability using Bayes' theorem."""
    prior_b = prior_b if prior_b != 0 else self.EPSILON
    return (conditional_b_given_a * prior_a) / prior_b

def calculate_conditional_probability(self, joint_probability: float, prior: float) -> float:
    """Calculate a conditional probability given joint probability and prior."""
    prior = prior if prior != 0 else self.EPSILON
    return joint_probability / prior

def calculate_high_entropy_prior(self, entropy: float) -> float:
    """Get prior based on entropy value."""
    return 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.1

def calculate_high_coherence_prior(self, coherence: float) -> float:
    """Get prior based on coherence value."""
    return 0.6 if coherence > self.HIGH_COHERENCE_THRESHOLD else 0.2

def calculate_joint_probability(self, coherence: float, action: int, prn_influence: float) -> float:
    """Calculate the joint probability of A and B based on coherence and action."""
    if coherence > self.HIGH_COHERENCE_THRESHOLD:
        if action == 1:
            return prn_influence * 0.8 + (1 - prn_influence) * 0.2
        else:
            return prn_influence * 0.1 + (1 - prn_influence) * 0.7
    return 0.3

def calculate_probabilities_and_select_action(self, entropy: float, coherence: float, prn_influence: float, action: int) -> dict:
    """Calculate probabilities and select an action based on entropy, coherence, PRN, and action."""
    high_entropy_prior = self.calculate_high_entropy_prior(entropy)
    high_coherence_prior = self.calculate_high_coherence_prior(coherence)

    conditional_b_given_a = prn_influence * 0.7 + (1 - prn_influence) * 0.3 if entropy > self.HIGH_ENTROPY_THRESHOLD else 0.2
    posterior_a_given_b = self.calculate_posterior_probability(high_entropy_prior, high_coherence_prior, conditional_b_given_a)

    joint_probability_ab = self.calculate_joint_probability(coherence, action, prn_influence)
    conditional_action_given_b = self.calculate_conditional_probability(joint_probability_ab, high_coherence_prior)

    action_to_take = 1 if conditional_action_given_b > self.ACTION_THRESHOLD else 0

    return {
        "action_to_take": action_to_take,
        "high_entropy_prior": high_entropy_prior,
        "high_coherence_prior": high_coherence_prior,
        "posterior_a_given_b": posterior_a_given_b,
        "conditional_action_given_b": conditional_action_given_b,
    }
#Define a simple wave function
def wave_function(x, t, amplitude=1.0, frequency=1.0, phase=0.5):
    """Defines a simple sinusoidal wave function."""
    return amplitude * np.sin(2 * np.pi * frequency * x - phase * t)


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
             plt.scatter(layer_index, node_index)

    plt.title(f"Network State at Iteration {iteration}")
    plt.xlabel("Layer Index")
    plt.ylabel("Node Index")
    plt.xlim(-1, len(network))
    plt.ylim(-1, max(len(layer) for layer in network))
    plt.grid()

    plt.tight_layout()  # Adjust subplot params for a tight layout
    plt.show()


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
    visualize_wave_and_network(network, iteration, t)  # added time t.


# Run simulation
network = [[initialize_node() for _ in range(n)] for n in [2, 3, 2, 2]]
average = np.array([1.0, 0.1]) #Example average for Mahalanobis
covariance = np.cov(np.random.rand(10,2)) #Example covariance matrix
for iteration in range(10):
    t = iteration + 0.1
    visualize_network(network, iteration)
    visualize_wave_and_network(network, iteration, t)

for layer in network:
    for node in layer:
        print("Node Norm:", np.linalg.norm(node))

print(network) #Uncomment for debugging
print(qc.measure_active()) #Uncomment for debugging
