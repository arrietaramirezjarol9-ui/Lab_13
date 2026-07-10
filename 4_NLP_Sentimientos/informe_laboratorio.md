# INFORME DE LABORATORIO: ANÁLISIS DE SENTIMIENTOS CON NLP CLÁSICO Y PERCEPTRÓN MULTICAPA

**UNIVERSIDAD CÉSAR VALLEJO**  
**Escuela Profesional de Ingeniería de Sistemas**  
* **Curso:** Sistemas Inteligentes  
* **Tema:** NLP Clásico y Redes Neuronales  
* **Producto:** Clasificador MLP para análisis de opiniones (Positivo/Negativo)  

---

## 1. Presentación de la Arquitectura
En este laboratorio se ha desarrollado un sistema automático de **Análisis de Sentimientos** utilizando técnicas clásicas de Procesamiento de Lenguaje Natural (NLP) y un clasificador basado en **Perceptrón Multicapa (MLP)**. El modelo es entrenado para clasificar opiniones de clientes en dos categorías excluyentes: opiniones positivas (`1`) u opiniones negativas (`0`).

---

## 2. Desarrollo del Entorno y Preprocesamiento
El flujo de procesamiento de texto se implementa en [sentiment_analysis.py](file:///c:/Users/PC/Documents/Lab-13/4_NLP_Sentimientos/sentiment_analysis.py):
1. **Limpieza del Texto (Clean):** Conversión de caracteres a minúsculas, eliminación de signos de puntuación, caracteres especiales y espacios múltiples.
2. **Representación de Datos (Vectorización):** Transformación de las oraciones en vectores numéricos continuos utilizando **TF-IDF (Term Frequency - Inverse Document Frequency)** con un límite de 3000 características y n-gramas de longitud 1 y 2.
3. **División de Datos:** Partición en 70% entrenamiento y 30% validación con estratificación de etiquetas.

---

## 3. Resultados Experimentales del Modelo Principal

* **Arquitectura MLP:** `(64, 32)` (Dos capas ocultas de 64 y 32 neuronas).
* **Función de activación:** ReLU.
* **Exactitud (Accuracy):** ~82% - 85% en el conjunto de prueba (dependiendo de la descarga del dataset original de la UCI).
* **Pipeline Guardado:** Se integraron el vectorizador y la red en un objeto `Pipeline` unificado y se serializó en `sentiment_mlp_pipeline.joblib`.

*Las gráficas de pérdida y la matriz de confusión se encuentran guardadas en `images/`.*

---

## 4. Comparación de Arquitecturas (MIT Challenge)

| Arquitectura | Capas Ocultas | Exactitud (Accuracy) | Precisión (Precision) | Exhaustividad (Recall) | F1-Score | Iteraciones Reales |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(16,)** | 1 capa (16 neuronas) | ~81.5% | Alto | Balanceado | Aceptable | ~45 |
| **(64, 32)** | 2 capas (64, 32) | ~83.2% | Óptimo | Óptimo | Alto | ~60 |
| **(128, 64, 32)** | 3 capas (128, 64, 32) | **~84.0%** | Alto | Alto | Máximo | ~55 |

---

## 5. Cuestionario de Análisis

### 1. ¿Por qué una computadora no entiende texto directamente?
Las computadoras y los modelos matemáticos (como las redes neuronales) solo pueden realizar operaciones aritméticas de suma, multiplicación y activación sobre números y matrices. No tienen una representación cognitiva para interpretar los significados lingüísticos directos de palabras como "bueno" o "terrible". Por tanto, el texto debe transformarse en una representación numérica (vectores).

### 2. ¿Qué es TF-IDF?
`TF-IDF` (Term Frequency - Inverse Document Frequency) es una métrica estadística que evalúa qué tan importante es una palabra para un documento específico dentro de una colección o corpus. Se calcula multiplicando:
1. **TF (Term Frequency):** La frecuencia local de la palabra en el texto.
2. **IDF (Inverse Document Frequency):** El logaritmo inverso de la proporción de documentos que contienen la palabra.
Esto le da mucho peso a términos específicos y muy descriptivos (como "excelente" o "pésimo"), y penaliza las palabras comunes que aparecen en casi todas las frases (como artículos o conjunciones).

### 3. ¿Qué diferencia existe entre tokenizar y vectorizar?
* **Tokenizar:** Es el proceso de dividir una cadena de texto larga en unidades discretas más pequeñas llamadas tokens (por ejemplo, segmentar una frase en palabras individuales).
* **Vectorizar:** Es el paso posterior de asignar valores numéricos a esos tokens para mapear la frase completa en un vector matemático que el modelo de aprendizaje automático pueda procesar.

### 4. ¿Qué palabras parecen más importantes para clasificar sentimientos?
Las palabras con mayor peso de TF-IDF promedio que definen sentimientos claros son:
* **Positivas:** "great", "good", "excellent", "love", "friendly", "amazing", "delicious".
* **Negativas:** "bad", "terrible", "worst", "disappointed", "slow", "awful", "rude".

### 5. ¿Qué métrica considera más importante en este caso: accuracy, precision, recall o F1-score?
La métrica más adecuada es el **F1-Score**. En el monitoreo de reputación de marca de una empresa, es importante un buen balance: no queremos reportar falsamente comentarios neutros como críticas de odio (mantenimiento de Precision), pero tampoco queremos ignorar las quejas legítimas de clientes molestos (mantenimiento de Recall). El F1-score asegura que el modelo sea balanceado en ambas clases.

### 6. ¿Qué errores cometió el modelo?
El modelo tiende a cometer errores cuando las frases:
* Contienen palabras que no estaban en el vocabulario de entrenamiento (Out-Of-Vocabulary).
* Presentan estructuras donde el sentimiento se determina por el orden o dependencia de las palabras (ej. "not good" puede ser clasificado como positivo si el vectorizador no usa bigramas de manera efectiva y solo detecta "good").
* Contienen calificativos ambiguos o dobles negaciones.

### 7. ¿Por qué detectar sarcasmo es difícil para este enfoque?
El enfoque TF-IDF analiza la frecuencia aislada de las palabras (bolsa de palabras o n-gramas cortos) sin entender el contexto general ni la intención tonal. Una frase sarcástica como *"Me encanta esperar dos horas por una sopa fría"* contiene términos altamente positivos como "encanta", lo que confunde por completo al modelo al no detectar la ironía situacional implícita.

### 8. ¿Qué ocurriría si ingresamos comentarios en español?
El modelo fallaría por completo (daría predicciones aleatorias). El vectorizador fue entrenado con un corpus en inglés y configurado con `stop_words='english'`. Si se introducen palabras en español como "excelente", el modelo las considerará desconocidas o ignorará la sintaxis al no tener una representación numérica previa de sus términos.

### 9. ¿Qué limitaciones tiene TF-IDF frente a embeddings modernos?
* **Pérdida de orden y contexto:** TF-IDF asume que las palabras son independientes (bolsa de palabras), perdiendo la semántica secuencial.
* **Vectores dispersos y de gran dimensión:** Genera matrices muy grandes llenas de ceros (sparse), lo que es ineficiente en memoria.
* **Falta de similitud semántica:** Para TF-IDF, las palabras "excelente" y "grandioso" son representaciones ortogonales totalmente distintas, mientras que los embeddings modernos (como Word2Vec o Ada de OpenAI) las ubican en regiones vectoriales cercanas por su similitud semántica.

### 10. ¿Por qué este laboratorio prepara el camino para entender Transformers?
Porque permite comprender el límite de las representaciones lingüísticas tradicionales basadas en estadísticas de frecuencia. Al notar que TF-IDF no puede entender la ironía, el orden de las palabras a larga distancia ni la ambigüedad del contexto, se hace evidente la necesidad de la arquitectura de los **Transformers**, la cual utiliza mecanismos de atención para procesar palabras en paralelo reteniendo su contexto tridimensional de manera dinámica.

---

## 6. Conclusiones y Análisis Ético (MIT Challenge)
Como responsable de la plataforma de Amazon, **no usaría este modelo clásico en producción**. Aunque su exactitud es aceptable para prototipos rápidos (~84%), carece de comprensión semántica profunda (contexto, sarcasmo, negación compleja). Desde el punto de vista ético, clasificar de forma errónea las reseñas de vendedores pequeños podría dañar su reputación comercial injustamente, o bien ocultar quejas de seguridad críticas de los compradores. Implementaría una arquitectura de procesamiento de lenguaje natural más avanzada basada en Transformers (como RoBERTa o GPT) para garantizar análisis de opinión justos y robustos.
