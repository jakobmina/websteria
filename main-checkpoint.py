import numpy as np
import qiskit.primitives
from qiskit import transpile
from qiskit.primitives import StatevectorSampler
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

from souldierAI.neuralQ.circuito_htx_completo import data

# 1. Define the observable to be measured
operator = SparsePauliOp.from_list([("XXY", 1), ("XYX", 1), ("YXX", 1), ("YYY", -1)])
qc = qiskit.QuantumCircuit(3,3)
qc.h(0)
qc.h(1)
qc.h(2)  # generate superposition
qc.p(np.pi * 1.618,  0)
  # add quantum phase
qc.rz(np.pi * 1.618, 1)  # 0th-qubit-Controlled-NOT gate on 1st qubit
qc.ccx(0, 1, 2)  # 0th-qubit-Controlled-NOT gate on 2nd qubit
qc.ry(np.pi * 1.618, 2)  # 1st-qubit-Controlled-NOT gate on 2nd qubit
qc.rx(np.pi * 1.618, 2) 
qc.cswap(0,1,2) # 2nd-qubit-Controlled-NOT gate on 3rd qubit

qc_transpiled = transpile(qc)

# 3. Add the classical output in the form of measurement of all qubits.
qc_measured = qc.measure_all(inplace=False)

# 4. Execute using the Sampler primitive and Estimator primitive.
sampler = StatevectorSampler().run([qc_measured])

estimator = StatevectorEstimator()
job = estimator.run([(qc, operator)], precision=1e-3)
result = [sampler.result(),job.result()]
expectation_value = result[0]["evs"].real
print(f"Expectation values: {expectation_value}")
print(f" > Counts: {result[0]("evs").binary_probabilities()}")
print(qc, operator)
print(qc_measured, qc_transpiled)

""""
Análisis del Código y su Salida
Este código en Python utiliza Qiskit para simular un circuito cuántico y calcular valores de expectación.

Detalles del Código.
Importación de Librerías:
 Se importan las librerías necesarias: 
 numpy para operaciones numéricas,
 qiskit para funcionalidades de computación cuántica,
 y módulos específicos para manipulación de vectores de estado y operadores.

Definición del Observable.
Se define el observable como un SparsePauliOp.
 El observable está representado por las cadenas de Pauli "XXY", "XYX", "YXX" e "YYY"
con coeficientes correspondientes 1, 1, 1 y -1.
 Este observable es del que queremos medir el valor de expectación.

Creación del Circuito Cuántico:
 Se crea un circuito cuántico de 3 cúbicos.
qc.h(0): Aplica una puerta de Hadamard al cúbito 0, creando una superposición.
qc.p(np.pi / 2, 0): Aplica una puerta de fase con ángulo π/2 al cúbito 0. Esto introduce una fase compleja.
qc.cx(0, 1) y qc.cx(0, 2): Aplican puertas CNOT (controlado-NO) desde el cúbito 0 a los cúbicos 1 y 2.
 Esto entrelaza los cúbicos. El circuito prepara el estado (|000⟩ + i|111⟩)/√2.

Transpilación del Circuito:
Se transpile el circuito para un backend específico (simulado aquí).
 La transpilación optimiza el circuito para el hardware objetivo (o simulador) descomponiendo las puertas en las puertas
base del backend (cz, sx, rz) y considerando el mapa de acoplamiento (conexiones entre cúbicos).
 optimization_level=3 solicita un alto nivel de optimización.

Medición de Todos los Cúbicos: Se añaden mediciones a todos los cúbicos.
 Esto es crucial para obtener los conteos en la parte del sampler del código.
inplace=False crea un nuevo circuito con mediciones en lugar de modificar el original.

Statevector Sampler (Muestreador de Vector de Estado):
 Se crea un StatevectorSampler.
  Esta primitiva simula el circuito y devuelve el vector de estado.

sampler.run([qc_measured], shots=1000): Ejecuta el circuito medido 1000 veces (simulaciones).
result[0].data["meas"].get_counts(): Extrae los conteos de las mediciones.
 Dado que el estado es (|000⟩ + i|111⟩)/√2,
esperamos probabilidades aproximadamente iguales para '000' y '111', que es lo que muestra la salida.

Statevector Estimator (Estimador de Vector de Estado):
 Se crea un StatevectorEstimator. 

Esta primitiva estima el valor de expectación de un observable dado un estado.
estimator.run([(qc, operator)], precision=1e-3): Ejecuta la estimación.

Toma una lista de tuplas, donde cada tupla contiene el circuito y el observable.
precision especifica la precisión deseada de la estimación.
result[0].data.evs: Extrae el valor de expectación estimado.

Análisis de la Salida:
> Counts: {'111': 514, '000': 486}: Como se esperaba, los conteos son cercanos a 500 para tanto '000' como '111',
 confirmando la creación del estado de superposición.
  La ligera diferencia se debe a la naturaleza probabilística de la medición cuántica (incluso en simulación).

Expectation values: 4.000639326085698: Este es el valor de expectación estimado del observable operator
para el estado cuántico preparado. El valor esperado debería ser 4.

Por qué el valor de expectación es 4:
El observable es XXY + XYX + YXX - YYY.
 Analicemos su acción sobre el estado (|000⟩ + i|111⟩)/√2.

XXY actuando sobre |000⟩ da 0 y sobre |111⟩ da -i|000⟩.
XYX actuando sobre |000⟩ da 0 y sobre |111⟩ da -i|000⟩.
YXX actuando sobre |000⟩ da 0 y sobre |111⟩ da -i|000⟩.
YYY actuando sobre |000⟩ da 0 y sobre |111⟩ da -|111⟩.

Por lo tanto, el observable actuando sobre el estado da (-3i|000⟩ - |111⟩)/√2. 
 El valor de expectación se calcula como el producto interno del estado y el estado transformado, lo que resulta en 4.

Circuito Transpilado: La salida también muestra el circuito transpilado. 
Es más complejo porque el transpilador ha descompuesto las puertas originales en las puertas base 
(cz, sx, rz) y las ha reorganizado para optimizar para el mapa de acoplamiento.

 En resumen, el código prepara un estado cuántico específico, lo mide para confirmar la preparación del estado
y luego estima el valor de expectación de un observable dado.
  Los resultados son consistentes con las predicciones teóricas.
"""""
