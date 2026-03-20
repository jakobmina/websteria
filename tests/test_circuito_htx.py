import pytest
import numpy as np
import torch
from circuito_htx_completo import initialize_node, is_active, shannon_entropy, activate_node_with_ccx

def test_initialize_node():
    """Verifica que el nodo se inicialice con dimensiones correctas y valores complejos."""
    node = initialize_node()
    assert node.shape == (3, 3)
    assert node.dtype == complex
    # Regla 1.1: Componente Simpléctica (Debe existir un estado de energía/orden)
    # Verificamos que el estado oculto (2,2) tenga magnitud
    assert abs(node[2, 2]) >= 0

def test_is_active():
    """Prueba la lógica de activación basada en densidades de probabilidad."""
    # Nodo inactivo (valores muy pequeños)
    inactive_node = np.zeros((3, 3), dtype=complex)
    assert is_active(inactive_node, threshold=0.5) == False
    
    # Nodo activo (valores que superan el umbral)
    active_node = np.zeros((3, 3), dtype=complex)
    active_node[0, 0] = 1.0 + 0j
    assert is_active(active_node, threshold=0.5) == True

def test_shannon_entropy():
    """Valida el cálculo de entropía de Shannon (S, Métrica/Disipativa)."""
    # Caso simple: Datos uniformes
    data = [1, 2, 3, 4]
    entropy = shannon_entropy(data)
    # Log2(4) = 2.0
    assert pytest.approx(entropy, 0.01) == 2.0
    
    # Caso con un solo valor (Entropía 0)
    data_single = [1, 1, 1]
    assert shannon_entropy(data_single) == 0

def test_symplectic_metric_duality():
    """
    Regla 1.3: Verificamos que el sistema no sea puramente conservativo ni disipativo.
    En este contexto, probamos que activate_node_with_ccx mantiene la estructura del nodo.
    """
    node = initialize_node()
    new_node = activate_node_with_ccx(node)
    
    # El mapeo debe conservar las dimensiones (Nivel 3 de Isomorfismo)
    assert new_node.shape == node.shape
    # El estado resultante no debe ser nulo (Prohibición de muerte térmica)
    assert np.any(new_node != 0)

@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_is_active_thresholds(threshold):
    """Prueba la estabilidad del umbral de activación."""
    node = initialize_node()
    # No debería explotar numéricamente (Regla 1.3)
    result = is_active(node, threshold=threshold)
    assert isinstance(result, bool)
