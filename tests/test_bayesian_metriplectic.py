import pytest
import numpy as np
import sys
import os

# Supplement path to import from static
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../static')))

from bayesian import (
    GoldenOperator, 
    InformationTopology, 
    MetriplecticSystem, 
    EnergyOracle, 
    MetriplecticEvaluator
)

def test_golden_operator():
    """Verify Golden Operator modulation."""
    val = GoldenOperator.modulate(1.0)
    # n=1: cos(pi) * cos(pi * phi) = -1 * cos(pi * phi)
    expected = -np.cos(np.pi * GoldenOperator.PHI)
    assert np.isclose(val, expected)

def test_information_topology_entropy():
    """Verify Shannon entropy calculation."""
    data = np.array([1, 1, 2, 2])
    entropy = InformationTopology.shannon_entropy(data)
    assert np.isclose(entropy, 1.0)  # Uniform distribution of 2 elements

def test_metriplectic_dynamics():
    """Verify conservative vs dissipative components."""
    system = MetriplecticSystem(psi_init=0.5, h_const=0.1, s_const=0.05)
    
    # Test Lagrangian
    l_symp, l_metr = system.compute_lagrangian()
    # L_symp = 0.5 * 0.5^2 - 0.1 = 0.125 - 0.1 = 0.025
    assert np.isclose(l_symp, 0.025)
    
    # Initial step
    system.step()
    assert len(system.history["psi"]) == 1
    assert len(system.history["symp"]) == 1
    assert len(system.history["metr"]) == 1

def test_energy_oracle_h7():
    """Verify H7 partner state."""
    assert EnergyOracle.get_partner(0) == 7
    assert EnergyOracle.get_partner(7) == 0
    assert EnergyOracle.get_partner(3) == 4

def test_evaluator_generalization():
    """Verify unified evaluator logic."""
    evaluator = MetriplecticEvaluator()
    # High entropy/new momentums should favor generalization
    result = evaluator.evaluate([1, 2, 5, 6])
    assert "status" in result
    assert "reynolds" in result["metrics"]
    assert "h7_pairs" in result

def test_informational_reynolds():
    """Verify Reynolds number calculation."""
    # Re = (rho * v) / mu
    re = InformationTopology.informational_reynolds(v=10, rho=0.1, mu=0.001)
    # (0.1 * 10) / 0.001 = 1 / 0.001 = 1000
    assert np.isclose(re, 1000)
