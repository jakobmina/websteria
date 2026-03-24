import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski

class QuantumNetwork:
    def __init__(self, layer_sizes):
        self.network = self.initialize_network(layer_sizes)
        self.bayes_logic = BayesLogic()

    def initialize_network(self, layer_sizes):
        return [[self.initialize_node() for _ in range(n)] for n in layer_sizes]

    @staticmethod
    def initialize_node():
        """Inicializa un nodo con amplitudes complejas aleatorias."""
        return np.array([[complex(np.random.rand(), np.random.rand()) for _ in range(3)] for _ in range(3)])

    @staticmethod
    def normalize_node(node):
        """Normaliza las amplitudes de un nodo."""
        norm = np.linalg.norm(node)
        return node / norm if norm > 0 else node

    def is_active(self, node, threshold=0.5):
        """Verifica si un nodo está activo."""
        prob_exposed = np.sum(np.abs(node[:2, :2])**2)
        return prob_exposed > threshold

    def simulate(self, iterations):
        """Ejecuta la simulación de la red cuántica."""
        for iteration in range(iterations):
            self.update_network(iteration)
            self.visualize_network(iteration)

    def update_network(self, iteration):
        """Actualiza el estado de la red."""
        for layer_index, layer in enumerate(self.network):
            for node_index, node in enumerate(layer):
                active_neighbors = self.calculate_neighbors(layer_index, node_index)
                updated_node = self.action_rules(node, active_neighbors)
                self.network[layer_index][node_index] = updated_node

    def calculate_neighbors(self, layer_index, node_index, p=2):
        """Calcula el número de vecinos activos."""
        active_neighbors = 0
        current_node = self.network[layer_index][node_index]

        for i in [layer_index - 1, layer_index + 1]:
            if 0 <= i < len(self.network):
                for neighbor_node in self.network[i]:
                    distance = minkowski(current_node.flatten(), neighbor_node.flatten(), p=p)
                    if distance < 1.0 and self.is_active(neighbor_node):
                        active_neighbors += 1

        return active_neighbors

    def action_rules(self, node, active_neighbors):
        """Aplica reglas de acción para determinar el siguiente estado."""
        if self.is_active(node):
            return node if 2 <= active_neighbors <= 3 else node
        else:
            return self.activate_node_with_ccx(node) if active_neighbors == 3 else node

    @staticmethod
    def activate_node_with_ccx(node):
        """Aplica la lógica de la puerta CCX."""
        # Implementación de la lógica de activación
        return node

    def visualize_network(self, iteration):
        """Visualiza el estado de la red."""
        plt.figure(figsize=(10, 6))
        for layer_index, layer in enumerate(self.network):
            for node_index, node in enumerate(layer):
                if self.is_active(node):
                    plt.scatter(layer_index, node_index, color='red')

        plt.title(f"Red Cuántica - Iteración {iteration}")
        plt.xlabel("Índice de Capa")
        plt.ylabel("Índice de Nodo")
        plt.grid(True)
        plt.show()

# Ejemplo de uso
def main():
    layer_sizes = [2, 3, 2, 2]
    quantum_network = QuantumNetwork(layer_sizes)
    quantum_network.simulate(10)

if __name__ == "__main__":
    main()

"""
Funciones principales:
1. Encapsulación: Toda la lógica se agrupa en una clase `QuantumNetwork`
2. Mejor modularidad y separación de responsabilidades
3. Métodos más concisos y enfocados
4. Uso de métodos estáticos para funciones independientes del estado
5. Simplificación de la lógica de simulación
6. Manejo de dependencias más limpio

Recomendaciones adicionales:
- Añadir manejo de errores
- Documentación de métodos
- Configuración de hiperparámetros
- Logging para seguimiento de simulación
- Pruebas unitarias

"""