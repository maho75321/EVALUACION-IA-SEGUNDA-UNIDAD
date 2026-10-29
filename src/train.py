
"""
Pregunta 4: entrenamiento, evaluación y generación de métricas.

Entrena la CNN con Adam y CrossEntropyLoss, registra las métricas
por época y guarda las curvas de aprendizaje y la matriz de confusión.
"""

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.optim import Adam

# Permitir importar los módulos del directorio src.
DIRECTORIO_SRC = Path(__file__).resolve().parent
DIRECTORIO_RAIZ = DIRECTORIO_SRC.parent
sys.path.insert(0, str(DIRECTORIO_SRC))

from dataset import cargar_datos
from model import CNNClasificador


# ---------------------------------------------------------
# 1. Configuración del experimento
# ---------------------------------------------------------

SEMILLA = 15
EPOCHS = 10
TASA_APRENDIZAJE = 0.001

DIRECTORIO_SALIDAS = DIRECTORIO_RAIZ / "outputs"
DIRECTORIO_SALIDAS.mkdir(parents=True, exist_ok=True)

DISPOSITIVO = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def establecer_semilla():
    """Controla las fuentes principales de aleatoriedad."""

    random.seed(SEMILLA)
    np.random.seed(SEMILLA)
    torch.manual_seed(SEMILLA)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEMILLA)


# ---------------------------------------------------------
# 2. Funciones de entrenamiento y evaluación
# ---------------------------------------------------------

def entrenar_epoca(modelo, cargador, funcion_perdida, optimizador):
    """Entrena el modelo durante una época."""

    modelo.train()

    perdida_acumulada = 0.0
    aciertos = 0
    total = 0

    for imagenes, etiquetas in cargador:
        imagenes = imagenes.to(DISPOSITIVO)
        etiquetas = etiquetas.to(DISPOSITIVO)

        # Eliminar gradientes anteriores.
        optimizador.zero_grad()

        # Propagación hacia adelante.
        predicciones = modelo(imagenes)

        # Calcular la pérdida de clasificación.
        perdida = funcion_perdida(predicciones, etiquetas)

        # Retropropagación y actualización de pesos.
        perdida.backward()
        optimizador.step()

        cantidad = etiquetas.size(0)
        perdida_acumulada += perdida.item() * cantidad

        clases_predichas = predicciones.argmax(dim=1)
        aciertos += (clases_predichas == etiquetas).sum().item()
        total += cantidad

    perdida_media = perdida_acumulada / total
    exactitud = 100.0 * aciertos / total

    return perdida_media, exactitud


def evaluar_modelo(modelo, cargador, funcion_perdida):
    """Evalúa el modelo sin actualizar sus parámetros."""

    modelo.eval()

    perdida_acumulada = 0.0
    aciertos = 0
    total = 0

    etiquetas_reales = []
    etiquetas_predichas = []

    with torch.no_grad():
        for imagenes, etiquetas in cargador:
            imagenes = imagenes.to(DISPOSITIVO)
            etiquetas = etiquetas.to(DISPOSITIVO)

            predicciones = modelo(imagenes)
            perdida = funcion_perdida(predicciones, etiquetas)

            cantidad = etiquetas.size(0)
            perdida_acumulada += perdida.item() * cantidad

            clases_predichas = predicciones.argmax(dim=1)

            aciertos += (
                clases_predichas == etiquetas
            ).sum().item()
            total += cantidad

            etiquetas_reales.extend(etiquetas.cpu().tolist())
            etiquetas_predichas.extend(
                clases_predichas.cpu().tolist()
            )

    perdida_media = perdida_acumulada / total
    exactitud = 100.0 * aciertos / total

    return (
        perdida_media,
        exactitud,
        etiquetas_reales,
        etiquetas_predichas,
    )


# ---------------------------------------------------------
# 3. Curvas de aprendizaje
# ---------------------------------------------------------

def guardar_curvas(historial):
    """Guarda las curvas de pérdida y exactitud."""

    epocas = range(1, len(historial["perdida_entrenamiento"]) + 1)

    figura, ejes = plt.subplots(1, 2, figsize=(13, 5))

    # Curvas de pérdida.
    ejes[0].plot(
        epocas,
        historial["perdida_entrenamiento"],
        marker="o",
        label="Entrenamiento",
    )
    ejes[0].plot(
        epocas,
        historial["perdida_validacion"],
        marker="o",
        label="Validación",
    )
    ejes[0].set_title("Pérdida por época")
    ejes[0].set_xlabel("Época")
    ejes[0].set_ylabel("CrossEntropyLoss")
    ejes[0].legend()
    ejes[0].grid(True, alpha=0.3)

    # Curvas de exactitud.
    ejes[1].plot(
        epocas,
        historial["exactitud_entrenamiento"],
        marker="o",
        label="Entrenamiento",
    )
    ejes[1].plot(
        epocas,
        historial["exactitud_validacion"],
        marker="o",
        label="Validación",
    )
    ejes[1].set_title("Exactitud por época")
    ejes[1].set_xlabel("Época")
    ejes[1].set_ylabel("Accuracy (%)")
    ejes[1].legend()
    ejes[1].grid(True, alpha=0.3)

    figura.suptitle("Curvas de aprendizaje de la CNN")
    figura.tight_layout()

    ruta = DIRECTORIO_SALIDAS / "loss_accuracy.png"
    figura.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(figura)

    print(f"Curvas guardadas en: {ruta}")


# ---------------------------------------------------------
# 4. Matriz de confusión
# ---------------------------------------------------------

def guardar_matriz_confusion(reales, predichas, nombres_clases):
    """Genera la matriz de confusión del conjunto de prueba."""

    matriz = confusion_matrix(
        reales,
        predichas,
        labels=list(range(len(nombres_clases))),
    )

    figura, eje = plt.subplots(figsize=(10, 9))

    visualizacion = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=nombres_clases,
    )

    visualizacion.plot(
        ax=eje,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
        values_format="d",
    )

    eje.set_title("Matriz de confusión - CIFAR-10")
    figura.tight_layout()

    ruta = DIRECTORIO_SALIDAS / "confusion_matrix.png"
    figura.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(figura)

    print(f"Matriz de confusión guardada en: {ruta}")


# ---------------------------------------------------------
# 5. Ejecución del entrenamiento
# ---------------------------------------------------------

def main():
    establecer_semilla()

    print("=== ENTRENAMIENTO DE LA CNN PARA CIFAR-10 ===")
    print(f"Semilla individual: {SEMILLA}")
    print(f"Dispositivo utilizado: {DISPOSITIVO}")
    print(f"Número de épocas: {EPOCHS}")

    # Cargar entrenamiento, validación y prueba.
    (
        cargador_entrenamiento,
        cargador_validacion,
        cargador_prueba,
        nombres_clases,
    ) = cargar_datos(mostrar_imagenes=False)

    # Inicializar el modelo y sus componentes de optimización.
    modelo = CNNClasificador(
        numero_clases=len(nombres_clases)
    ).to(DISPOSITIVO)

    funcion_perdida = nn.CrossEntropyLoss()
    optimizador = Adam(
        modelo.parameters(),
        lr=TASA_APRENDIZAJE,
    )

    historial = {
        "perdida_entrenamiento": [],
        "perdida_validacion": [],
        "exactitud_entrenamiento": [],
        "exactitud_validacion": [],
    }

    mejor_exactitud_validacion = -1.0
    ruta_mejor_modelo = DIRECTORIO_SALIDAS / "mejor_modelo.pth"

    # Entrenamiento iterativo por épocas.
    for epoca in range(EPOCHS):
        perdida_train, exactitud_train = entrenar_epoca(
            modelo,
            cargador_entrenamiento,
            funcion_perdida,
            optimizador,
        )

        (
            perdida_val,
            exactitud_val,
            _,
            _,
        ) = evaluar_modelo(
            modelo,
            cargador_validacion,
            funcion_perdida,
        )

        historial["perdida_entrenamiento"].append(perdida_train)
        historial["perdida_validacion"].append(perdida_val)
        historial["exactitud_entrenamiento"].append(exactitud_train)
        historial["exactitud_validacion"].append(exactitud_val)

        print(
            f"Época {epoca + 1:02d}/{EPOCHS} | "
            f"Pérdida entrenamiento: {perdida_train:.4f} | "
            f"Exactitud entrenamiento: {exactitud_train:.2f}% | "
            f"Pérdida validación: {perdida_val:.4f} | "
            f"Exactitud validación: {exactitud_val:.2f}%"
        )

        # Guardar los pesos con mejor resultado de validación.
        if exactitud_val > mejor_exactitud_validacion:
            mejor_exactitud_validacion = exactitud_val

            torch.save(
                modelo.state_dict(),
                ruta_mejor_modelo,
            )

    # Generar las curvas con el historial completo.
    guardar_curvas(historial)

    # Cargar el mejor modelo antes de la evaluación final.
    modelo.load_state_dict(
        torch.load(
            ruta_mejor_modelo,
            map_location=DISPOSITIVO,
            weights_only=True,
        )
    )

    # Evaluar una sola vez sobre el conjunto de prueba independiente.
    (
        perdida_prueba,
        exactitud_prueba,
        etiquetas_reales,
        etiquetas_predichas,
    ) = evaluar_modelo(
        modelo,
        cargador_prueba,
        funcion_perdida,
    )

    guardar_matriz_confusion(
        etiquetas_reales,
        etiquetas_predichas,
        nombres_clases,
    )

    print("\n=== RESULTADOS FINALES EN EL CONJUNTO DE PRUEBA ===")
    print(f"Pérdida de prueba: {perdida_prueba:.4f}")
    print(f"Exactitud de prueba: {exactitud_prueba:.2f}%")
    print("\nReporte de clasificación por clase:")

    print(
        classification_report(
            etiquetas_reales,
            etiquetas_predichas,
            labels=list(range(len(nombres_clases))),
            target_names=nombres_clases,
            zero_division=0,
            digits=4,
        )
    )

    print(f"Mejor modelo guardado en: {ruta_mejor_modelo}")
    print("Entrenamiento y evaluación finalizados.")


if __name__ == "__main__":
    main()
