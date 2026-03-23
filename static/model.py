import qiskit
from qiskit.circuit.library import n_local
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import numpy as np
from qiskit.transpiler import CouplingMap

# 1. Create a Quantum Circuit with 5 qubits
qc = qiskit.QuantumCircuit(5)

# 2. Define the coupling map for connectivity
coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])

# 3. Create the NLocal ansatz using the functional approach (replaces TwoLocal)
# Following Qiskit 2.x+ migration path
ansatz = n_local(
    num_qubits=5,
    rotation_blocks=["rx", "ry", "rz"],
    entanglement_blocks="cx",
    entanglement="linear",
    reps=2,
    insert_barriers=True
)

qc.compose(ansatz, inplace=True)

# 4. Measure all qubits
qc.measure_all()

# 5. Simulation and Transpilation
simulator = Aer.get_backend('qasm_simulator')
transpiled_circuit = qiskit.transpile(qc, simulator)

# Assign random parameters to the circuit before running
params = np.random.rand(transpiled_circuit.num_parameters)
bound_circuit = transpiled_circuit.assign_parameters(params)

job = simulator.run(bound_circuit, shots=1024)
result = job.result()
counts = result.get_counts()

# --- Visualization Section ---

# 1. Prepare data for plotting (handle empty counts case)
if counts:
    x = list(counts.keys())  # Resulting states (binary strings)
    y = list(counts.values())  # Frequencies of each state

    # 2. Create the Bar Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, y)
    plt.xlabel("Resulting Quantum State")
    plt.ylabel("Frequency")
    plt.title("Quantum Simulation Results")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # 3. Display the plot
    plt.show()
else:
    print("No simulation results obtained.")

# 6. Visualize the circuit structure
qc.draw()

