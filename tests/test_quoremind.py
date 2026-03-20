import pytest
import numpy as np
from quoremind_engine import MetriplecticDynamics, compute_golden_operator, QuoreMindProcessor

def test_golden_operator():
    """Verifica que el operador áureo no sea trivial."""
    o1 = compute_golden_operator(1)
    o2 = compute_golden_operator(2)
    assert isinstance(o1, float)
    assert o1 != o2

def test_poisson_bracket_energy_conservation():
    """En un sistema puramente conservativo, {H, H} debe ser 0."""
    q, p = 1.0, 1.0
    val = MetriplecticDynamics.poisson_bracket(
        MetriplecticDynamics.H, MetriplecticDynamics.H, q, p
    )
    assert abs(val) < 1e-4

def test_metric_bracket_dissipation():
    """La métrica debe generar un valor no nulo para [S, S]."""
    q, p = 1.0, 1.0
    M = np.eye(2) * 0.1
    val = MetriplecticDynamics.metric_bracket(
        MetriplecticDynamics.S, MetriplecticDynamics.S, q, p, M
    )
    # [S, S]_M = grad(S)^T * M * grad(S) >= 0
    assert val > 0

def test_lagrangian_splitting():
    """Verifica que el lagrangiano se divida en componentes."""
    l_s, l_m = MetriplecticDynamics.compute_lagrangian(1.0, 1.0, 0.5, 0.5)
    assert isinstance(l_s, float)
    assert isinstance(l_m, float)

def test_engine_processing():
    """Verifica el flujo básico del procesador."""
    engine = QuoreMindProcessor()
    mock_input = np.array([0.5, 0.5, 0.5, 0.5])
    res = engine.process_information(mock_input)
    assert "v_flow" in res
    assert "L_symp" in res
    assert res["new_n"] == 2
