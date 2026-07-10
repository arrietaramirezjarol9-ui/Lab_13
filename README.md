# Laboratorio N.º 13: Redes Neuronales y NLP (Sistemas Inteligentes)

**UNIVERSIDAD CÉSAR VALLEJO**  
**Escuela Profesional de Ingeniería de Sistemas**  

* **Estudiante:** Jermain Jarol Arrieta Ramirez  
* **Semestre:** 2026-I  
* **Curso:** Sistemas Inteligentes  
* **Tema:** Aplicación Práctica de Redes Neuronales (MLP, RNN, CNN) y NLP  

---

## 📌 Presentación del Repositorio

Este repositorio contiene la resolución e informes de las cuatro secciones prácticas correspondientes al **Laboratorio 13** del curso de Sistemas Inteligentes. Se han desarrollado modelos predictivos y de procesamiento secuencial utilizando arquitecturas avanzadas de aprendizaje profundo en Python.

---

## 📂 Estructura del Proyecto

El repositorio está organizado en las siguientes carpetas:

```text
Lab-13/
├── 1_MLP_CampanasBancarias/      # Perceptrón Multicapa (MLP)
│   ├── train_mlp.py              # Script de entrenamiento y evaluación
│   ├── informe_laboratorio.md    # Cuestionario y análisis conceptual
│   ├── mlp_bank_marketing_model.joblib # Modelo serializado
│   └── images/                   # Gráficos y matrices de confusión
│
├── 2_RNN_Traductor/              # Redes Recurrentes (Seq2Seq LSTM)
│   ├── translator.py             # Traductor Ruso -> Español con Keras
│   ├── informe_laboratorio.md    # Respuestas teóricas
│   ├── russian_spanish_seq2seq_lstm.keras # Modelo guardado
│   └── images/                   # Curvas de pérdida y exactitud
│
├── 3_CNN_Digitos/                # Redes Convolucionales (CNN)
│   ├── cnn_digits.py             # Reconocedor de dígitos manuscritos
│   ├── informe_laboratorio.md    # Cuestionario de convolución y pooling
│   ├── cnn_digits_uci.keras      # Modelo CNN serializado
│   └── images/                   # Aciertos, errores y muestras de entrada
│
└── 4_NLP_Sentimientos/            # NLP y MLP
    ├── sentiment_analysis.py     # Análisis de sentimientos (TF-IDF + MLP)
    ├── informe_laboratorio.md    # Cuestionario e informe ético
    ├── sentiment_mlp_pipeline.joblib # Pipeline guardado
    └── images/                   # Métricas comparativas
```

---

## 🛠️ Requisitos e Instalación

Para ejecutar cualquiera de los laboratorios en tu entorno local, asegúrate de tener instalado Python 3.10 o superior y las siguientes dependencias:

```bash
pip install ucimlrepo tensorflow pandas scikit-learn matplotlib joblib
```

---

## 🚀 Ejecución de los Proyectos

Puedes entrenar, evaluar y generar los gráficos de cada laboratorio ejecutando su respectivo archivo principal desde la terminal:

### 1. Campañas Bancarias (MLP)
```bash
cd 1_MLP_CampanasBancarias
python train_mlp.py
```

### 2. Traductor Ruso a Español (RNN/LSTM)
```bash
cd ../2_RNN_Traductor
python translator.py
```

### 3. Clasificación de Dígitos (CNN)
```bash
cd ../3_CNN_Digitos
python cnn_digits.py
```

### 4. Análisis de Sentimientos (TF-IDF + MLP)
```bash
cd ../4_NLP_Sentimientos
python sentiment_analysis.py
```

---

## 📊 Resumen de Resultados y Métricas

A continuación se consolidan las métricas clave obtenidas en la evaluación de los mejores modelos en los conjuntos de prueba:

| Laboratorio / Modelo | Métrica Clave | Exactitud (Accuracy) | Pérdida (Loss) | Entregables Generados |
| :--- | :---: | :---: | :---: | :--- |
| **MLP Campañas Bancarias** | F1-Score | **~90.52%** | N/A | Matriz de confusión, Modelo `.joblib` |
| **RNN Traductor** | Inferencia Seq2Seq | **100%** (en dataset) | ~0.02 | Modelo `.keras`, Curva Loss |
| **CNN Dígitos Manuscritos** | Accuracy | **98.70%** | 0.0514 | Muestra de errores, Modelo `.keras` |
| **NLP Análisis de Sentimientos**| F1-Score | **78.91%** | N/A | Pipeline unificado, Gráficos |

*Los análisis individuales, justificaciones de negocio, limitaciones y respuestas a los cuestionarios teóricos correspondientes a cada tema se encuentran detallados en el archivo `informe_laboratorio.md` de cada subcarpeta.*
