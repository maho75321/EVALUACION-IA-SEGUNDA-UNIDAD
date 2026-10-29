
"""
PREGUNTA 1: Carga, visualización y preparación del conjunto CIFAR-10.

Este módulo descarga los datos, muestra imágenes, aplica transformaciones,
realiza una partición estratificada y comprueba las dimensiones de los
tensores utilizados por la red neuronal convolucional.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# =========================================================
# 1. CONFIGURACIÓN GENERAL
# =========================================================

SEMILLA = 15
TAMANO_LOTE = 64
PROPORCION_VALIDACION = 0.20

# Obtener las rutas principales del proyecto.
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
DIRECTORIO_DATOS = DIRECTORIO_RAIZ / "data"
DIRECTORIO_SALIDAS = DIRECTORIO_RAIZ / "outputs"

# Crear las carpetas si todavía no existen.
DIRECTORIO_DATOS.mkdir(parents=True, exist_ok=True)
DIRECTORIO_SALIDAS.mkdir(parents=True, exist_ok=True)

# Nombres de las diez clases de CIFAR-10 en español.
CLASES = [
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
]

# Medias y desviaciones estándar de los canales RGB.
MEDIA_RGB = (0.4914, 0.4822, 0.4465)
DESVIACION_RGB = (0.2470, 0.2435, 0.2616)


def establecer_semilla(semilla=SEMILLA):
    """Fija las semillas para favorecer la reproducibilidad."""

    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)

    # Solicitar operaciones deterministas cuando sean compatibles.
    torch.use_deterministic_algorithms(True)

    print(f"Semilla individual configurada: {semilla}")


# =========================================================
# 2. TRANSFORMACIONES DE LAS IMÁGENES
# =========================================================

# Transformaciones utilizadas únicamente durante el entrenamiento.
transformaciones_entrenamiento = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_RGB, DESVIACION_RGB),
])

# En validación y prueba no se aplican aumentos aleatorios.
transformaciones_evaluacion = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEDIA_RGB, DESVIACION_RGB),
])


# =========================================================
# 3. VISUALIZACIÓN DE 16 IMÁGENES
# =========================================================

def mostrar_muestras(dataset, cantidad=16):
    """
    Muestra imágenes con sus etiquetas reales y guarda la figura.

    Matplotlib requiere imágenes con forma (alto, ancho, canales).
    PyTorch almacena los tensores como (canales, alto, ancho).
    Por ello, se cambia el orden de los ejes antes de visualizarlos.
    """

    cantidad = min(cantidad, len(dataset))

    # Seleccionar imágenes sin repetir y usando una semilla fija.
    generador = np.random.default_rng(SEMILLA)
    indices = generador.choice(
        len(dataset),
        size=cantidad,
        replace=False,
    )

    columnas = 4
    filas = int(np.ceil(cantidad / columnas))

    figura, ejes = plt.subplots(
        filas,
        columnas,
        figsize=(12, 3 * filas),
    )

    # Convertir los ejes en un arreglo unidimensional.
    ejes = np.asarray(ejes).reshape(-1)

    for posicion, indice in enumerate(indices):
        imagen, etiqueta = dataset[int(indice)]

        # CORRECCIÓN:
        # Cambiar (canales, alto, ancho) a (alto, ancho, canales).
        imagen = imagen.permute(1, 2, 0).cpu().numpy()

        # Mostrar la imagen RGB y su categoría real.
        ejes[posicion].imshow(imagen)
        ejes[posicion].set_title(
            f"Etiqueta: {CLASES[int(etiqueta)]}"
        )
        ejes[posicion].axis("off")

    # Ocultar los espacios que no se utilicen.
    for eje in ejes[cantidad:]:
        eje.axis("off")

    figura.suptitle(
        "Muestras del conjunto de datos CIFAR-10",
        fontsize=15,
    )
    figura.tight_layout()

    # Guardar la evidencia visual del dataset.
    ruta_imagen = DIRECTORIO_SALIDAS / "muestras_dataset.png"
    figura.savefig(
        ruta_imagen,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Visualización guardada en: {ruta_imagen}")

    # Mostrar la figura y liberar memoria.
    plt.show()
    plt.close(figura)


# =========================================================
# 4. CARGA Y PARTICIÓN ESTRATIFICADA
# =========================================================

def cargar_datos(mostrar_imagenes=True):
    """
    Carga CIFAR-10 y devuelve los cargadores de entrenamiento,
    validación y prueba, además de los nombres de las clases.
    """

    establecer_semilla()

    print("\nDescargando o cargando CIFAR-10...")

    # Cargar las imágenes originales para visualización y etiquetas.
    dataset_visualizacion = datasets.CIFAR10(
        root=str(DIRECTORIO_DATOS),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Mostrar las 16 imágenes solicitadas en la evaluación.
    if mostrar_imagenes:
        mostrar_muestras(
            dataset_visualizacion,
            cantidad=16,
        )

    # Obtener las etiquetas y los índices del conjunto original.
    etiquetas = np.asarray(dataset_visualizacion.targets)
    indices = np.arange(len(etiquetas))

    # Separar entrenamiento y validación conservando la proporción
    # de las diez clases en ambos subconjuntos.
    indices_entrenamiento, indices_validacion = train_test_split(
        indices,
        test_size=PROPORCION_VALIDACION,
        random_state=SEMILLA,
        stratify=etiquetas,
    )

    # Dataset con aumento de datos para el entrenamiento.
    dataset_entrenamiento_completo = datasets.CIFAR10(
        root=str(DIRECTORIO_DATOS),
        train=True,
        download=False,
        transform=transformaciones_entrenamiento,
    )

    # Dataset sin aumento aleatorio para la validación.
    dataset_validacion_completo = datasets.CIFAR10(
        root=str(DIRECTORIO_DATOS),
        train=True,
        download=False,
        transform=transformaciones_evaluacion,
    )

    # Seleccionar los índices de cada subconjunto.
    dataset_entrenamiento = Subset(
        dataset_entrenamiento_completo,
        indices_entrenamiento.tolist(),
    )

    dataset_validacion = Subset(
        dataset_validacion_completo,
        indices_validacion.tolist(),
    )

    # Cargar el conjunto oficial de prueba de CIFAR-10.
    dataset_prueba = datasets.CIFAR10(
        root=str(DIRECTORIO_DATOS),
        train=False,
        download=True,
        transform=transformaciones_evaluacion,
    )

    # Generador con semilla fija para el orden de los lotes.
    generador = torch.Generator().manual_seed(SEMILLA)

    # Crear el cargador de entrenamiento con lotes aleatorios.
    cargador_entrenamiento = DataLoader(
        dataset_entrenamiento,
        batch_size=TAMANO_LOTE,
        shuffle=True,
        generator=generador,
        num_workers=0,
    )

    # En validación y prueba se conserva el orden de los datos.
    cargador_validacion = DataLoader(
        dataset_validacion,
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
    )

    cargador_prueba = DataLoader(
        dataset_prueba,
        batch_size=TAMANO_LOTE,
        shuffle=False,
        num_workers=0,
    )

    # Mostrar el tamaño de cada partición.
    print("\n--- RESUMEN DE LOS DATOS ---")
    print(
        f"Imágenes de entrenamiento: {len(dataset_entrenamiento)}"
    )
    print(
        f"Imágenes de validación:    {len(dataset_validacion)}"
    )
    print(
        f"Imágenes de prueba:        {len(dataset_prueba)}"
    )
    print(f"Número de clases:          {len(CLASES)}")

    # Obtener un lote para comprobar las dimensiones.
    imagenes, etiquetas_lote = next(iter(cargador_entrenamiento))

    print("\n--- COMPROBACIÓN DE TENSORES ---")
    print(f"Forma de las imágenes: {tuple(imagenes.shape)}")
    print(f"Forma de las etiquetas: {tuple(etiquetas_lote.shape)}")
    print(f"Tipo de dato: {imagenes.dtype}")

    # Comprobar que los datos tienen el formato esperado por la CNN.
    assert imagenes.ndim == 4, (
        "Error: las imágenes deben formar un tensor 4D."
    )

    assert imagenes.shape[1:] == (3, 32, 32), (
        "Error: cada imagen debe tener tres canales RGB y tamaño 32x32."
    )

    assert etiquetas_lote.ndim == 1, (
        "Error: las etiquetas deben formar un tensor unidimensional."
    )

    print("Comprobación de tensores superada correctamente.")

    return (
        cargador_entrenamiento,
        cargador_validacion,
        cargador_prueba,
        CLASES,
    )


# =========================================================
# 5. PRUEBA INDEPENDIENTE DEL MÓDULO
# =========================================================

if __name__ == "__main__":
    cargar_datos(mostrar_imagenes=True)
