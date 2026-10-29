
"""
Pregunta 3: definición de una red neuronal convolucional (CNN).

La arquitectura recibe imágenes CIFAR-10 con forma
(lote, 3, 32, 32) y produce diez puntuaciones por imagen.
"""

import torch
import torch.nn as nn


class CNNClasificador(nn.Module):
    """Red CNN para clasificar las diez clases de CIFAR-10."""

    def __init__(self, numero_clases=10):
        super().__init__()

        # Bloque 1: (lote, 3, 32, 32) -> (lote, 32, 16, 16)
        self.bloque_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Bloque 2: (lote, 32, 16, 16) -> (lote, 64, 8, 8)
        self.bloque_conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Aplanamiento: (lote, 64, 8, 8) -> (lote, 4096)
        self.aplanar = nn.Flatten()

        # Clasificador: 4096 -> 256 -> 10 clases
        self.clasificador = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(p=0.30),
            nn.Linear(256, numero_clases),
        )

    def forward(self, x):
        """Define el flujo de los tensores por la red."""

        if x.ndim != 4:
            raise ValueError(
                "La entrada debe ser un tensor 4D con forma "
                "(lote, canales, alto, ancho)."
            )

        x = self.bloque_conv1(x)
        x = self.bloque_conv2(x)
        x = self.aplanar(x)
        x = self.clasificador(x)

        return x


def comprobar_dimensiones():
    """Comprueba las dimensiones intermedias de la CNN."""

    print("=== VALIDACIÓN DEL FLUJO DE TENSORES 4D ===")

    modelo = CNNClasificador()
    entrada = torch.randn(4, 3, 32, 32)

    modelo.eval()

    print("Entrada:", tuple(entrada.shape))

    with torch.no_grad():
        salida_bloque1 = modelo.bloque_conv1(entrada)
        print("Después del bloque 1:",
              tuple(salida_bloque1.shape))

        salida_bloque2 = modelo.bloque_conv2(salida_bloque1)
        print("Después del bloque 2:",
              tuple(salida_bloque2.shape))

        salida_aplanada = modelo.aplanar(salida_bloque2)
        print("Después de Flatten:",
              tuple(salida_aplanada.shape))

        salida_final = modelo.clasificador(salida_aplanada)
        print("Salida final:", tuple(salida_final.shape))

        # Comprobación de todas las dimensiones.
        assert salida_bloque1.shape == (4, 32, 16, 16)
        assert salida_bloque2.shape == (4, 64, 8, 8)
        assert salida_aplanada.shape == (4, 4096)
        assert salida_final.shape == (4, 10)

    print("Todas las comprobaciones se superaron correctamente.")


if __name__ == "__main__":
    comprobar_dimensiones()
