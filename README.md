# Taller 6 — Diseño y Optimización de una Red Neuronal Convolucional (CNN)

**Asignatura:** Técnicas de Inteligencia Artificial  
**Profesor:** Flavio Prieto — faprietoo@unal.edu.co  
**Programa:** Ingeniería Mecatrónica  
**Universidad:** Universidad Nacional de Colombia — Sede Bogotá  
**Fecha:** Abril 30, 2026  
**Framework:** PyTorch | **Dataset:** CIFAR-10

---

## Tabla de contenidos

1. [Descripción del problema](#1-descripción-del-problema)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Resultados obtenidos](#3-resultados-obtenidos)
4. [Arquitectura diseñada](#4-arquitectura-diseñada)
5. [Cómo ejecutar el notebook](#5-cómo-ejecutar-el-notebook)
6. [Requisitos del sistema](#6-requisitos-del-sistema)
7. [Preguntas de análisis](#7-preguntas-de-análisis)
8. [Referencias](#8-referencias)

---

## 1. Descripción del problema

Se diseña, entrena y optimiza una **Red Neuronal Convolucional (CNN)** para clasificar imágenes a color del dataset **CIFAR-10** en 10 categorías:

| Índice | Clase (ES) | Clase (EN) |
|--------|-----------|-----------|
| 0 | Avión | Airplane |
| 1 | Automóvil | Automobile |
| 2 | Pájaro | Bird |
| 3 | Gato | Cat |
| 4 | Ciervo | Deer |
| 5 | Perro | Dog |
| 6 | Rana | Frog |
| 7 | Caballo | Horse |
| 8 | Barco | Ship |
| 9 | Camión | Truck |

**Características del dataset:**
- 60 000 imágenes a color de 32×32×3 píxeles
- 50 000 para entrenamiento / 10 000 para prueba
- Clases perfectamente balanceadas (6 000 imágenes por clase)

---

## 2. Estructura del repositorio

```
Taller6_CNN/
├── Taller6_CNN_CIFAR10.ipynb   ← Notebook principal (informe técnico completo)
├── requirements.txt             ← Dependencias con versiones exactas
├── README.md                    ← Este archivo
├── setup_env.bat                ← Script de configuración automática (Windows)
└── .gitignore                   ← Archivos excluidos del repositorio
```

El notebook está organizado en **6 partes** que cubren todo el ciclo de vida del modelo:

| Parte | Contenido |
|-------|-----------|
| **1** | Preprocesamiento: carga, split estratificado, EDA, normalización, Data Augmentation |
| **2** | Diseño de la CNN: arquitectura configurable + justificación de cada decisión |
| **3** | Búsqueda de hiperparámetros: Random Search sobre 7 hiperparámetros |
| **4** | Selección de los 3 mejores modelos: tabla comparativa + curvas de entrenamiento |
| **5** | Evaluación final: accuracy, F1 macro/micro, matrices de confusión, análisis de sobreajuste |
| **6** | Discusión: 5 preguntas de análisis respondidas + comparación CNN vs MLP |

---

## 3. Resultados obtenidos

> Los valores exactos dependen de la configuración sorteada en el Random Search y del hardware.  
> Los rangos típicos para esta arquitectura en CIFAR-10 son:

| Métrica | Valor típico |
|---------|-------------|
| Test Accuracy | 72 – 82 % |
| F1-score Macro | 0.72 – 0.82 |
| F1-score Micro | 0.72 – 0.82 |
| Tiempo de entrenamiento (CPU) | 20 – 60 min |
| Tiempo de entrenamiento (GPU) | 3 – 8 min |

**Clases con mayor dificultad de clasificación** (confusión frecuente):
- Gato ↔ Perro (similitud visual alta)
- Pájaro ↔ Avión (formas similares en 32×32)
- Automóvil ↔ Camión (categorías vehiculares)

---

## 4. Arquitectura diseñada

La CNN sigue un diseño **VGG-like** adaptado a imágenes de 32×32:

```
Input: (B, 3, 32, 32)
│
├── Bloque 1: Conv(3→F₁, k×k) → BN → ReLU → Conv(F₁→F₁, k×k) → BN → ReLU → MaxPool(2×2)
│   Salida: (B, F₁, 16, 16)
│
├── Bloque 2: Conv(F₁→F₂, k×k) → BN → ReLU → Conv(F₂→F₂, k×k) → BN → ReLU → MaxPool(2×2)
│   Salida: (B, F₂, 8, 8)
│
├── [Bloque 3 opcional]: Conv(F₂→F₃, k×k) → BN → ReLU → Conv(F₃→F₃, k×k) → BN → ReLU → MaxPool(2×2)
│   Salida: (B, F₃, 4, 4)
│
├── Flatten → (B, F_last × spatial²)
├── Linear(→256) → ReLU → BN → Dropout(p)
├── Linear(256→128) → ReLU → Dropout(p)
└── Linear(128→10)  ← logits (Softmax implícito en CrossEntropyLoss)
```

**Hiperparámetros buscados:**

| Hiperparámetro | Espacio de búsqueda |
|---|---|
| Bloques convolucionales | {2, 3} |
| Filtros por bloque | {[32,64], [32,64,128], [64,128,256]} |
| Tamaño de kernel | {3, 5} |
| Tasa de aprendizaje | {1e-3, 5e-4, 1e-4} |
| Batch size | {64, 128} |
| Dropout rate | {0.3, 0.4, 0.5} |
| Batch Normalization | {True, False} |

---

## 5. Cómo ejecutar el notebook

### Opción A — Google Colab (recomendado, sin instalar nada)

1. Hacer clic en el badge:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmendozar2001/Taller6_CNN/blob/main/Taller6_CNN_CIFAR10.ipynb)

2. En Colab: `Entorno de ejecución → Cambiar tipo → GPU T4` (opcional pero recomendado)
3. `Ctrl + F9` para ejecutar todas las celdas
4. La primera celda instala las dependencias automáticamente

---

### Opción B — Ejecución local (Windows, Linux o macOS)

#### Paso 1 — Clonar el repositorio

```bash
git clone https://github.com/lmendozar2001/Taller6_CNN.git
cd Taller6_CNN
```

#### Paso 2 — Instalar dependencias

**Windows (cualquier versión de Python 3.9–3.14):**

```cmd
pip install torch==2.9.0+cpu torchvision==0.24.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn notebook ipykernel
```

**Linux / macOS:**

```bash
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas scikit-learn matplotlib seaborn notebook ipykernel
```

**Con GPU NVIDIA (CUDA 12.1):**

```bash
pip install torch==2.9.0+cu121 torchvision==0.24.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn matplotlib seaborn notebook ipykernel
```

#### Paso 3 — Abrir el notebook

```bash
jupyter notebook Taller6_CNN_CIFAR10.ipynb
```

#### Paso 4 — Ejecutar

- Ejecutar todas las celdas en orden: `Kernel → Restart & Run All`
- CIFAR-10 se descarga automáticamente (~170 MB) en la carpeta `./data/`

---

### Opción C — Script automático (solo Windows)

Doble clic en `setup_env.bat`. El script:
1. Verifica que Python 3.11 esté instalado
2. Crea un entorno virtual aislado en `./venv/`
3. Instala todas las dependencias
4. Registra el kernel de Jupyter

Luego:
```cmd
venv\Scripts\activate
jupyter notebook Taller6_CNN_CIFAR10.ipynb
```

---

## 6. Requisitos del sistema

| Componente | Mínimo | Recomendado |
|---|---|---|
| Python | 3.9 | 3.11 – 3.14 |
| RAM | 4 GB | 8 GB |
| Almacenamiento | 500 MB | 1 GB |
| GPU | No requerida | NVIDIA con CUDA 12.x |
| SO | Windows 10 / Ubuntu 20.04 / macOS 12 | Cualquiera |

**Versiones exactas probadas:**

| Librería | Versión |
|---|---|
| torch | 2.9.0+cpu |
| torchvision | 0.24.0+cpu |
| numpy | ≥ 1.26.0 |
| pandas | ≥ 2.2.0 |
| scikit-learn | ≥ 1.4.0 |
| matplotlib | ≥ 3.8.0 |
| seaborn | ≥ 0.13.0 |

---

## 7. Preguntas de análisis

El notebook responde en detalle las siguientes preguntas (Parte 6):

1. **¿Cuál fue el impacto de la profundidad de la red?**  
   Más bloques = mayor campo receptivo y capacidad representacional, pero con imágenes de 32×32 el límite práctico es 3 bloques (mapa 4×4 tras MaxPool).

2. **¿Qué efecto tuvo el tamaño del kernel?**  
   Kernel 3×3 es más eficiente (9 parámetros vs 25 del 5×5). Dos capas 3×3 apiladas tienen el mismo campo receptivo que una 5×5 con más no-linealidad.

3. **¿Cuál fue el rol del Data Augmentation?**  
   Regularizador implícito: aumenta la diversidad efectiva del dataset, introduce invarianzas y reduce la memorización. Mejora típica de 3–8 puntos porcentuales.

4. **¿Se logró reducir el sobreajuste?**  
   Sí. La combinación de Dropout + Batch Normalization + Data Augmentation + Early Stopping + Weight Decay mantuvo la brecha train/test por debajo del 10%.

5. **¿Cómo se compara la CNN con un MLP?**  
   La CNN supera al MLP en ~20–30 puntos porcentuales gracias a la compartición de pesos (convolución) y la invarianza espacial (pooling). El MLP trata cada píxel como característica independiente, perdiendo la estructura espacial.

---

## 8. Referencias

1. LeCun, Y., et al. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep CNNs. *NeurIPS*, 25.
3. Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks. *arXiv:1409.1556*.
4. Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*.
5. Srivastava, N., et al. (2014). Dropout. *JMLR*, 15(1), 1929–1958.
6. He, K., et al. (2015). Delving deep into rectifiers. *ICCV*.
7. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. *Technical Report, U. Toronto*.
8. Paszke, A., et al. (2019). PyTorch: An imperative style deep learning library. *NeurIPS*, 32.

---

*Repositorio: [github.com/lmendozar2001/Taller6_CNN](https://github.com/lmendozar2001/Taller6_CNN)*
