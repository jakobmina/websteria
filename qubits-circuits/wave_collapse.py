import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from typing import List, Tuple, Optional
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

class QuantumWave:
    """
    Clase para modelar ondas cuánticas con propiedades avanzadas
    """
    def __init__(
        self, 
        amplitude: float, 
        frequency: float, 
        phase: float, 
        quantum_number: Optional[int] = None
    ):
        """
        Inicializa una onda cuántica
        
        Args:
            amplitude (float): Amplitud de la onda
            frequency (float): Frecuencia de la onda
            phase (float): Fase inicial de la onda
            quantum_number (int, optional): Número cuántico asociado
        """
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.quantum_number = quantum_number

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evalúa la onda en un conjunto de puntos
        
        Args:
            x (np.ndarray): Puntos de evaluación
        
        Returns:
            np.ndarray: Valores de la onda
        """
        return self.amplitude * np.sin(2 * np.pi * self.frequency * x + self.phase)

    def collapse(
        self, 
        superposed_wave: np.ndarray, 
        y: np.ndarray
    ) -> float:
        """
        Simula el colapso de la función de onda
        
        Args:
            superposed_wave (np.ndarray): Onda superpuesta
            y (np.ndarray): Onda original
        
        Returns:
            float: Estado colapsado
        """
        normalization = self.amplitude / np.sum(np.abs(y))
        return np.random.choice(superposed_wave, p=np.abs(superposed_wave) / np.sum(np.abs(superposed_wave)))

class QuantumSimulation:
    """
    Clase para simulación de sistemas cuánticos
    """
    def __init__(
        self, 
        waves: List[QuantumWave], 
        x_range: Tuple[float, float] = (0, 10), 
        num_points: int = 500
    ):
        """
        Inicializa la simulación cuántica
        
        Args:
            waves (List[QuantumWave]): Lista de ondas cuánticas
            x_range (Tuple[float, float]): Rango de x
            num_points (int): Número de puntos de muestreo
        """
        self.waves = waves
        self.x = np.linspace(x_range[0], x_range[1], num_points)
        self.superposed_wave = self._superpose_waves()

    def _superpose_waves(self) -> np.ndarray:
        """
        Superpone las ondas cuánticas
        
        Returns:
            np.ndarray: Onda superpuesta
        """
        return np.sum([wave.evaluate(self.x) for wave in self.waves], axis=0)

    def perform_fft(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza la Transformada Rápida de Fourier
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Frecuencias y magnitudes espectrales
        """
        N = len(self.x)
        T = self.x[1] - self.x[0]
        
        yf = fft(self.superposed_wave)
        xf = fftfreq(N, T)[:N//2]
        
        return xf, 2.0/N * np.abs(yf[0:N//2])

    def visualize(self):
        """
        Visualiza la simulación cuántica
        """
        try:
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            # Ondas individuales
            for wave in self.waves:
                axs[0, 0].plot(
                    self.x, 
                    wave.evaluate(self.x), 
                    label=f"Onda (A={wave.amplitude}, f={wave.frequency})"
                )
            
            # Onda superpuesta
            axs[0, 0].plot(
                self.x, 
                self.superposed_wave, 
                label="Onda Superpuesta", 
                color="green", 
                linewidth=2
            )
            
            axs[0, 0].set_title("Ondas Cuánticas")
            axs[0, 0].set_xlabel("x")
            axs[0, 0].set_ylabel("ψ(x)")
            axs[0, 0].legend()
            axs[0, 0].grid(True)

            # Espectro de frecuencia
            xf, yf = self.perform_fft()
            axs[0, 1].plot(xf, yf, color="red")
            axs[0, 1].set_title("Espectro de Frecuencia")
            axs[0, 1].set_xlabel("Frecuencia")
            axs[0, 1].set_ylabel("Amplitud Espectral")
            axs[0, 1].grid(True)

            # Distribución de probabilidad
            probabilities = np.abs(self.superposed_wave) / np.sum(np.abs(self.superposed_wave))
            axs[1, 0].hist(self.superposed_wave, weights=probabilities, bins=30, color="purple")
            axs[1, 0].set_title("Distribución de Probabilidad")
            axs[1, 0].set_xlabel("Valor de Onda")
            axs[1, 0].set_ylabel("Probabilidad")

            # Colapso de onda
            collapsed_state = np.random.choice(
                self.superposed_wave, 
                p=probabilities
            )
            axs[1, 1].axhline(
                y=collapsed_state, 
                color='red', 
                linestyle='--', 
                label=f"Estado Colapsado: {collapsed_state:.4f}"
            )
            axs[1, 1].set_title("Colapso de Onda")
            axs[1, 1].set_xlabel("x")
            axs[1, 1].set_ylabel("Valor Colapsado")
            axs[1, 1].legend()

            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.error(f"Error en visualización: {e}")

def main():
    # Crear ondas cuánticas
    waves = [
        QuantumWave(amplitude=0.5, frequency=1.5, phase=-np.pi/21, quantum_number=1),
        QuantumWave(amplitude=0.3, frequency=2.0, phase=np.pi/4, quantum_number=2)
    ]

    # Iniciar simulación
    simulation = QuantumSimulation(waves)
    simulation.visualize()

if __name__ == "__main__":
    main()