# Taller 6 — Diseño y Optimización de una CNN
### Técnicas de Inteligencia Artificial · Universidad Nacional de Colombia

Clasificación multiclase sobre **CIFAR-10** (10 clases, 60 000 imágenes 32×32×3)  
usando una Red Neuronal Convolucional construida con **TensorFlow / Keras**.

---

## Contenido del repositorio

```
Taller6_CNN/
├── Taller6_CNN_CIFAR10.ipynb   ← Notebook principal (informe técnico completo)
├── requirements.txt             ← Dependencias exactas
├── README.md                    ← Este archivo
└── .gitignore
```

## Estructura del notebook

| Parte | Descripción |
|-------|-------------|
| **1** | Preprocesamiento: carga, EDA, normalización, Data Augmentation |
| **2** | Diseño de la CNN: arquitectura base + justificación de decisiones |
| **3** | Búsqueda de hiperparámetros (Random Search) |
| **4** | Selección de los 3 mejores modelos + curvas de entrenamiento |
| **5** | Evaluación en test: accuracy, matriz de confusión, F1 macro/micro |
| **6** | Discusión y análisis (solo Markdown) |

---

## Ejecución rápida

### Opción A — Google Colab (recomendado, sin instalación)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lmendozar2001/Taller6_CNN/blob/main/Taller6_CNN_CIFAR10.ipynb)

1. Clic en el badge de arriba.  
2. En Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.  
3. Ejecutar todas las celdas: **Entorno de ejecución → Ejecutar todo** (`Ctrl+F9`).

> CIFAR-10 se descarga automáticamente (~170 MB). No se necesita ningún archivo adicional.

---

### Opción B — Entorno local

#### Requisitos previos
- Python 3.10 o 3.11
- Git

#### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/lmendozar2001/Taller6_CNN.git
cd Taller6_CNN

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Lanzar Jupyter
jupyter notebook Taller6_CNN_CIFAR10.ipynb
```

#### Con GPU (opcional pero recomendado)
Si tienes una GPU NVIDIA con CUDA 12.x instalado, TensorFlow 2.15 la detectará automáticamente. Verifica con:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Resultados esperados

| Métrica | Valor típico |
|---------|-------------|
| Test Accuracy | 78 – 85 % |
| F1-score Macro | 0.78 – 0.84 |
| Tiempo entrenamiento (CPU) | 15 – 40 min |
| Tiempo entrenamiento (GPU) | 3 – 8 min |

> Los resultados exactos varían según el hardware y la configuración sorteada en el Random Search.  
> La semilla `SEED = 42` garantiza reproducibilidad en el mismo hardware.

---

## Dependencias principales

| Librería | Versión |
|----------|---------|
| TensorFlow | 2.15.0 |
| NumPy | 1.26.4 |
| Pandas | 2.2.2 |
| Matplotlib | 3.8.4 |
| Seaborn | 0.13.2 |
| Scikit-learn | 1.4.2 |

---

## Autor
**lmendozar2001** · Universidad Nacional de Colombia  
Curso: Técnicas de Inteligencia Artificial · Prof. Flavio Prieto
