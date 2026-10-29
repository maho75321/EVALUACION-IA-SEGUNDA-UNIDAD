# Clasificación de imágenes con redes neuronales convolucionales (CNN)

**Curso:** Inteligencia Artificial\
**Código de matrícula:** `017200915e`\
**Semilla individual:** `15`\
**Dataset:** CIFAR-10\
**Framework:** PyTorch

## 1. Objetivo del proyecto

Desarrollar una red neuronal convolucional (CNN) capaz de clasificar
imágenes del conjunto CIFAR-10 en diez categorías: avión, automóvil,
pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

El proyecto comprende la preparación de los datos, el diseño de la
arquitectura, la comprobación de tensores 4D, el entrenamiento y la
evaluación del modelo mediante métricas y visualizaciones.

## 2. Pregunta 1: dataset y preprocesamiento

### Dataset utilizado

CIFAR-10 contiene 60 000 imágenes RGB de 32 × 32 píxeles: 50 000
imágenes oficiales de entrenamiento y 10 000 imágenes oficiales de
prueba.

En `src/dataset.py` se carga el dataset y se genera una visualización de
16 imágenes con sus etiquetas para comprobar que las imágenes y sus
categorías se leen correctamente.

### Preprocesamiento y aumento de datos

-   `ToTensor()` convierte las imágenes en tensores y escala los píxeles
    al intervalo de 0 a 1.
-   `Normalize()` normaliza los canales RGB con las medias y
    desviaciones estándar definidas en el código.
-   `RandomCrop` genera variaciones mediante recortes aleatorios con
    relleno durante el entrenamiento.
-   `RandomHorizontalFlip` voltea horizontalmente imágenes con una
    probabilidad del 50 % durante el entrenamiento.

Los aumentos aleatorios no se aplican a validación ni a prueba.

### División de los datos

Las 50 000 imágenes oficiales de entrenamiento se dividieron en un
conjunto de entrenamiento y uno de validación. El conjunto oficial de
prueba se mantuvo independiente.

  Subconjunto        Cantidad de imágenes
  ---------------- ----------------------
  Entrenamiento                    40 000
  Validación                       10 000
  Prueba oficial                   10 000
  **Total**                    **60 000**

La división entre entrenamiento y validación es estratificada. La
semilla individual utilizada es `15`, correspondiente a los dos últimos
dígitos numéricos de la matrícula indicada para el trabajo.

## 3. Pregunta 2: arquitectura y justificación de la CNN

### ¿Por qué una CNN y no un MLP?

Un MLP convencional suele aplanar la imagen desde el inicio y procesarla
como un vector de 3072 valores. Una CNN utiliza filtros que recorren la
imagen y aprenden características locales, como bordes, texturas y
formas, conservando la estructura espacial durante las primeras etapas.
Además, comparte los pesos de sus filtros entre posiciones.

Por estas razones, una CNN es apropiada para la clasificación de
imágenes.

### Arquitectura implementada

La arquitectura se encuentra en `src/model.py`.

  -----------------------------------------------------------------------
  Etapa                   Operación               Forma de salida
  ----------------------- ----------------------- -----------------------
  Entrada                 Imagen RGB de 32 × 32   `(N, 3, 32, 32)`

  Bloque convolucional 1  `Conv2d(3, 32)`, ReLU y `(N, 32, 16, 16)`
                          MaxPool2d               

  Bloque convolucional 2  `Conv2d(32, 64)`, ReLU  `(N, 64, 8, 8)`
                          y MaxPool2d             

  Aplanamiento            `Flatten`               `(N, 4096)`

  Capa densa              `Linear(4096, 256)`,    `(N, 256)`
                          ReLU y Dropout 0.30     

  Salida                  `Linear(256, 10)`       `(N, 10)`
  -----------------------------------------------------------------------

`N` representa el número de imágenes del lote. La salida contiene diez
puntuaciones por imagen, una por clase. Estas puntuaciones se utilizan
directamente con `CrossEntropyLoss`, sin aplicar Softmax antes de
calcular la pérdida.

### Función de las capas

-   **Conv2d:** extrae características visuales mediante filtros
    aprendibles.
-   **ReLU:** introduce no linealidad.
-   **MaxPool2d:** reduce las dimensiones espaciales.
-   **Flatten:** transforma los mapas de características en un vector.
-   **Linear:** combina las características para producir puntuaciones
    de clase.
-   **Dropout:** desactiva aleatoriamente algunas activaciones durante
    el entrenamiento para ayudar a reducir el sobreajuste.

## 4. Pregunta 3: comprobación de tensores 4D

PyTorch representa un lote de imágenes con el formato `(N, C, H, W)`:

-   `N`: cantidad de imágenes.
-   `C`: canales de color.
-   `H`: altura.
-   `W`: anchura.

La comprobación se realizó con cuatro imágenes simuladas y produjo las
siguientes dimensiones:

  Etapa                  Forma comprobada
  ---------------------- -------------------
  Entrada                `(4, 3, 32, 32)`
  Después del bloque 1   `(4, 32, 16, 16)`
  Después del bloque 2   `(4, 64, 8, 8)`
  Después de Flatten     `(4, 4096)`
  Salida final           `(4, 10)`

Todas las aserciones de dimensiones se superaron correctamente. También
se verificó un lote real del `DataLoader` con forma `(64, 3, 32, 32)` y
sus etiquetas con forma `(64,)`.

Para repetir la comprobación, ejecutar:

``` powershell
.\.venv\Scripts\python.exe src\model.py
```

## 5. Pregunta 4: entrenamiento y evaluación

### Configuración

  Parámetro             Valor utilizado
  --------------------- --------------------
  Optimizador           Adam
  Función de pérdida    `CrossEntropyLoss`
  Tasa de aprendizaje   `0.001`
  Épocas                10
  Tamaño del lote       64
  Semilla               15
  Dispositivo           CPU

En cada lote, la red genera puntuaciones, se calcula la pérdida y
`backward()` obtiene los gradientes. El optimizador Adam actualiza los
parámetros. Al final de cada época se registran pérdida y exactitud de
entrenamiento y validación.

El mejor modelo se selecciona según la exactitud de validación. Después
se cargan sus pesos y se evalúa una vez sobre el conjunto oficial de
prueba, que no se utiliza para actualizar los parámetros.

### Resultados experimentales obtenidos

Los siguientes resultados corresponden a la ejecución real de diez
épocas.

  Métrica                                     Resultado
  ----------------------------------------- -----------
  Mejor exactitud de validación                 73.63 %
  Época del mejor resultado de validación            10
  Pérdida de prueba                              0.7668
  Exactitud de prueba                           73.13 %
  Imágenes del conjunto de prueba                10 000
  Promedio macro de precisión                    0.7298
  Promedio macro de recall                       0.7313
  Promedio macro de F1-score                     0.7272

La exactitud de prueba de **73.13 %** significa que el modelo clasificó
correctamente aproximadamente 7 313 de las 10 000 imágenes evaluadas. La
cantidad exacta de aciertos es la suma de la diagonal de la matriz de
confusión.

### Evolución por época

  ----------------------------------------------------------------------------
           Época       Exactitud      Exactitud         Pérdida        Pérdida
                   entrenamiento     validación   entrenamiento     validación
  -------------- --------------- -------------- --------------- --------------
               1         40.16 %        55.17 %          1.6340         1.2505

               2         52.97 %        62.33 %          1.3013         1.0482

               3         58.34 %        65.55 %          1.1654         0.9814

               4         61.75 %        68.47 %          1.0802         0.8955

               5         63.75 %        69.12 %          1.0175         0.8686

               6         65.28 %        70.20 %          0.9862         0.8452

               7         66.42 %        72.43 %          0.9528         0.7906

               8         67.35 %        73.18 %          0.9256         0.7831

               9         68.30 %        73.57 %          0.9014         0.7564

              10         68.80 %        73.63 %          0.8871         0.7581
  ----------------------------------------------------------------------------

La exactitud de validación aumentó del 55.17 % al 73.63 %. La pérdida
disminuyó a lo largo del entrenamiento, tanto en entrenamiento como en
validación, aunque la pérdida de validación subió ligeramente en la
última época respecto de la novena. En las diez épocas observadas no se
aprecia un aumento sostenido de la pérdida de validación.

### Métricas por clase

  Clase         Precisión   Recall   F1-score
  ----------- ----------- -------- ----------
  Avión            0.6833   0.8240     0.7471
  Automóvil        0.7764   0.9030     0.8350
  Pájaro           0.6278   0.5820     0.6040
  Gato             0.6096   0.4450     0.5145
  Ciervo           0.7612   0.6310     0.6900
  Perro            0.6060   0.6860     0.6435
  Rana             0.7752   0.8140     0.7941
  Caballo          0.7879   0.7840     0.7860
  Barco            0.8547   0.8350     0.8447
  Camión           0.8163   0.8090     0.8127

La clase automóvil obtuvo el mayor recall, con 0.9030. La clase gato
obtuvo el menor recall, con 0.4450, por lo que el modelo no identificó
correctamente una parte importante de los gatos.

### Interpretación de la matriz de confusión

La matriz de confusión muestra los aciertos en su diagonal y las
confusiones fuera de ella. Entre las confusiones más destacadas de esta
ejecución están:

-   **Gato confundido con perro:** 226 imágenes.
-   **Camión confundido con automóvil:** 122 imágenes.
-   **Pájaro confundido con avión:** 113 imágenes.
-   **Perro confundido con gato:** 114 imágenes.

Estas confusiones son compatibles con las dificultades de distinguir
algunas categorías visualmente similares en imágenes pequeñas de 32 × 32
píxeles.

### Archivos de resultados

El entrenamiento guardó los siguientes archivos en `outputs/`:

-   `loss_accuracy.png`: curvas de pérdida y exactitud de entrenamiento
    y validación.
-   `confusion_matrix.png`: matriz de confusión del conjunto de prueba.
-   `mejor_modelo.pth`: pesos del modelo con mejor exactitud de
    validación.
-   `muestras_dataset.png`: visualización de 16 imágenes del dataset.

También se comprobó que `mejor_modelo.pth` puede cargarse en PyTorch y
que el modelo produce una salida con forma `(1, 10)` para una imagen de
entrada.

## 6. Estructura del proyecto

``` text
evaluacion/
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── notebooks/
│   └── experiment.ipynb
├── outputs/
│   ├── muestras_dataset.png
│   ├── loss_accuracy.png
│   ├── confusion_matrix.png
│   └── mejor_modelo.pth
├── requirements.txt
├── .gitignore
└── README.md
```

-   `src/dataset.py`: carga, preprocesa y visualiza los datos.
-   `src/model.py`: define la arquitectura CNN y comprueba las
    dimensiones de los tensores.
-   `src/train.py`: entrena y evalúa el modelo, y guarda las métricas y
    visualizaciones.
-   `notebooks/experiment.ipynb`: notebook de experimentación.
-   `outputs/`: almacena las imágenes generadas y los pesos del modelo.
-   `requirements.txt`: enumera las dependencias de Python.

## 7. Instalación y ejecución

Ejecutar los comandos desde PowerShell, en la carpeta raíz del proyecto
`evaluacion`.

### Instalar dependencias

Si el entorno virtual ya está creado:

``` powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Comprobar el dataset y la arquitectura

``` powershell
.\.venv\Scripts\python.exe src\dataset.py
.\.venv\Scripts\python.exe src\model.py
```

### Entrenar y evaluar

``` powershell
.\.venv\Scripts\python.exe src\train.py
```

La primera ejecución necesita conexión a Internet para descargar
CIFAR-10 si no está disponible localmente. El tiempo de entrenamiento
depende del hardware; la ejecución documentada en este informe se
realizó en CPU.

## 8. Conclusiones

Se implementó y evaluó una CNN para clasificar las diez categorías de
CIFAR-10. Se verificaron las dimensiones de los tensores en cada etapa
de la arquitectura, se completaron diez épocas de entrenamiento y se
generaron las curvas de aprendizaje, el reporte de clasificación y la
matriz de confusión.

El modelo alcanzó una exactitud de **73.13 % en el conjunto de prueba**,
con un F1-score macro de **0.7272**. Los resultados muestran un
desempeño desigual entre clases: automóvil y barco obtuvieron resultados
relativamente altos, mientras que gato fue la clase con mayores
dificultades de identificación.

Como mejora futura, se podría probar más épocas, ajustar la arquitectura
y comparar configuraciones de hiperparámetros usando únicamente el
conjunto de validación. La evaluación final debe seguir realizándose
sobre el conjunto de prueba independiente.

## 9. Requisitos y reproducibilidad

-   Mantener `src/dataset.py`, `src/model.py` y `src/train.py` junto con
    este README.
-   Mantener los archivos de evidencia de `outputs/` en la entrega.
-   No incluir el entorno virtual `.venv/` ni el conjunto de datos
    descargado en el repositorio.
-   Los resultados descritos corresponden a la ejecución documentada;
    una nueva ejecución puede producir pequeñas diferencias según el
    entorno y las operaciones aleatorias.
