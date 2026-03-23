import json
import os
import numpy as np

def get_notebook_code():
    nb_path = os.path.join(os.path.dirname(__file__), "..", "souldierAI", "neuralQ", "neural_network_training.ipynb")
    with open(nb_path, 'r') as f:
        nb = json.load(f)
    
    code_blocks = []
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            code_blocks.append("".join(cell["source"]))
    return "\n".join(code_blocks)

def test_metriplectic_neural_network_compiles_and_runs():
    code = get_notebook_code()
    
    # Create an empty namespace for the notebook execution
    namespace = {}
    
    # Execute the code from the notebook
    exec(code, namespace)
    
    # Check that the MetriplecticNeuralNetwork class was defined
    assert "MetriplecticNeuralNetwork" in namespace, "Neural network class not found in notebook."
    MetriplecticNeuralNetwork = namespace["MetriplecticNeuralNetwork"]
    
    # Test initialization
    model = MetriplecticNeuralNetwork(layers=[2, 4, 1], activation='relu')
    assert len(model.weights) == 2
    assert len(model.momentum_w) == 2
    
    # Test forward pass
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    activations, zs = model.forward(X)
    assert len(activations) == 3 # Input + 2 hidden/output layers
    
    # Test lagrangian
    L_symp, L_metr = model.compute_lagrangian(X, y)
    assert isinstance(L_symp, float)
    assert isinstance(L_metr, float)
    
    # Test step (training loop)
    loss, grads = model.step(X, y)
    assert isinstance(loss, float)
    assert len(grads) == 2
