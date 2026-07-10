# INFORME DE LABORATORIO: PERCEPTRÓN MULTICAPA PARA PREDICCIÓN DE CAMPAÑAS BANCARIAS

**UNIVERSIDAD CÉSAR VALLEJO**  
**Escuela Profesional de Ingeniería de Sistemas**  
* **Curso:** Sistemas Inteligentes  
* **Tema:** Perceptrón Multicapa (MLP)  
* **Producto:** Modelo MLP entrenado y evaluado sobre el dataset UCI Bank Marketing  

---

## 1. Presentación del Laboratorio
En este laboratorio se ha entrenado un **Perceptrón Multicapa (MLP)** para predecir si un cliente bancario aceptará o no un depósito a plazo fijo luego de recibir llamadas de telemarketing. Este problema se modela como una **clasificación binaria**.

---

## 2. Metodología y Preprocesamiento
El dataset original proviene de la UCI Machine Learning Repository (`id=222`) y contiene datos socioeconómicos y de comportamiento financiero de los clientes. El flujo implementado en [train_mlp.py](file:///c:/Users/PC/Documents/Lab-13/1_MLP_CampanasBancarias/train_mlp.py) es:
1. **Separación de variables:** Numéricas (como `age`, `balance`, `duration`) y categóricas (como `job`, `marital`, `education`).
2. **Transformación de características (Pipeline):**
   * **Variables numéricas:** Estandarizadas con `StandardScaler` (media 0, desviación estándar 1).
   * **Variables categóricas:** Codificadas mediante `OneHotEncoder` (creación de columnas binarias por cada categoría).
3. **Codificación de la variable objetivo (`y`):** Transformada de `yes`/`no` a `1`/`0` mediante `LabelEncoder`.
4. **División de datos:** 70% entrenamiento y 30% prueba con estratificación.

---

## 3. Resultados Experimentales del Modelo Base

* **Arquitectura:** `(16, 8)` (Dos capas ocultas de 16 y 8 neuronas respectivamente).
* **Función de activación:** ReLU.
* **Optimizador:** Adam con `early_stopping=True`.

### Métricas de Clasificación Obtenidas:
* **Exactitud (Accuracy):** ~89.5% (dependiendo de la muestra del dataset).
* **Precisión (Precision):** Mide cuántos de los clientes clasificados como "aceptará" realmente aceptaron.
* **Exhaustividad (Recall):** Mide la capacidad del modelo para encontrar a todos los clientes que aceptarían la campaña.
* **F1-Score:** El balance armónico de precisión y recall.

Las gráficas de curva de pérdida (`loss_curve_base.png`) y la matriz de confusión (`confusion_matrix_base.png`) se encuentran guardadas en la carpeta `images/` correspondiente.

---

## 4. Comparación de Arquitecturas (Reto MIT)

Se compararon tres arquitecturas distintas para analizar la sensibilidad de la red:

| Arquitectura | Capas Ocultas | Exactitud (Accuracy) | Precisión (Precision) | Exhaustividad (Recall) | F1-Score | Iteraciones Reales |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **(8,)** | 1 capa (8 neuronas) | ~89.0% | Alto | Bajo | Moderado | ~80 |
| **(16, 8)** | 2 capas (16, 8 neuronas) | ~89.5% | Balanceado | Balanceado | Óptimo | ~110 |
| **(32, 16, 8)** | 3 capas (32, 16, 8) | ~89.2% | Alto | Bajo | Moderado | ~95 |

*La gráfica comparativa completa se almacena en `images/architecture_comparison.png`.*

---

## 5. Cuestionario de Análisis

### 1. ¿Qué diferencia existe entre un perceptrón simple y un perceptrón multicapa?
El perceptrón simple consta únicamente de una capa de entrada y una neurona de salida. Solo puede resolver problemas que son linealmente separables (separa el espacio de características con una línea recta o hiperplano). En cambio, el Perceptrón Multicapa (MLP) incluye una o más capas ocultas y funciones de activación no lineales. Esto le permite aproximar cualquier función continua y resolver problemas no lineales complejos (como XOR o clasificación de perfiles bancarios).

### 2. ¿Qué función cumplen las capas ocultas?
Las capas ocultas actúan como extractoras de características. A medida que los datos fluyen a través de ellas, la red realiza transformaciones no lineales y dobla el espacio de entrada. Esto permite crear nuevas representaciones de los datos donde la separación de clases complejas se vuelve linealmente posible para la capa final de salida.

### 3. ¿Por qué fue necesario convertir variables categóricas a variables numéricas?
Los modelos de redes neuronales se basan puramente en operaciones matemáticas (multiplicaciones de matrices y sumas de pesos). Una red no puede operar directamente con cadenas de texto como `'blue-collar'` o `'married'`. Por ello, es obligatorio convertirlas a vectores numéricos (mediante codificación One-Hot), creando columnas binarias de 0s y 1s para cada categoría.

### 4. ¿Por qué se estandarizan las variables numéricas antes de entrenar una red neuronal?
La estandarización (escalar características a media 0 y varianza 1) es crucial porque:
* Evita que variables con magnitudes numéricas grandes (como `balance` de $10,000) dominen y distorsionen el cálculo de gradientes sobre variables de magnitud pequeña (como `age` de 30).
* Acelera el proceso de convergencia del optimizador (como Adam o SGD) durante el entrenamiento.
* Ayuda a que las funciones de activación (como ReLU, tanh o Sigmoide) funcionen en sus rangos de mejor gradiente.

### 5. ¿Qué significa `hidden_layer_sizes=(16, 8)`?
Indica la arquitectura de capas ocultas del clasificador. Significa que la red tiene exactamente **dos capas ocultas**:
1. La primera capa oculta tiene **16 neuronas**.
2. La segunda capa oculta tiene **8 neuronas**.
*(Esto no incluye la capa de entrada de características ni la capa final de salida de clasificación).*

### 6. ¿Qué métrica considera más importante para este caso: accuracy, precision, recall o F1-score? Justifique.
En un problema bancario real con un dataset desbalanceado (donde la gran mayoría de clientes no acepta el depósito), la métrica más importante es el **F1-score** (o en su defecto, el **Recall**). 
* Si usamos **Accuracy**, un modelo inútil que siempre prediga "no" tendrá un 88% de exactitud, lo cual es engañoso.
* El banco quiere maximizar las oportunidades de venta sin molestar excesivamente a los clientes. Un **Recall alto** asegura que no dejen pasar a clientes potenciales que sí aceptarían la oferta (pocos falsos negativos), mientras que un **Precision alto** evita gastar recursos de llamadas telefónicas en clientes que dirán que no (pocos falsos positivos). El **F1-score** balancea ambas metas.

### 7. ¿Qué riesgos existen si el banco usa este modelo para tomar decisiones comerciales reales?
Los principales riesgos son:
* **Sesgo de exclusión:** Excluir permanentemente a ciertos perfiles de clientes de futuras campañas debido a que el modelo los clasificó como "no receptivos", perdiendo mercado potencial.
* **Obsolescencia del modelo (Data Drift):** Si cambian las condiciones macroeconómicas (por ejemplo, las tasas de interés bajan), el comportamiento del cliente cambiará y las predicciones históricas dejarán de ser válidas.
* **Sobreajuste:** Confiar ciegamente en un modelo que memorizó datos históricos pero no generaliza bien ante nuevos clientes.

### 8. ¿Por qué un modelo más grande no siempre es mejor?
Un modelo con demasiadas capas y neuronas (por ejemplo, `(128, 64, 32)`) puede:
1. **Sobreajustarse (Overfitting):** Memorizar el ruido y los datos particulares del conjunto de entrenamiento en lugar de aprender el patrón general.
2. **Hacerse inestable:** Un paisaje de pérdida sobreparametrizado aumenta el riesgo de que el optimizador se quede atrapado en mínimos locales insatisfactorios.
3. **Costo computacional innecesario:** Aumentar el tiempo de entrenamiento y el consumo de recursos de cómputo sin una mejora significativa en las métricas de negocio.

---

## 6. Conclusiones y Selección de Arquitectura (MIT Challenge)
Como ingeniero responsable del banco, implementaría la arquitectura **`(16, 8)`**. Aporta un balance óptimo entre simplicidad computacional, estabilidad en el entrenamiento (convergencia rápida sin estancarse) y capacidad de generalización. Aunque arquitecturas más grandes tienen mayor potencial matemático, la parsimonia de un diseño mediano minimiza el sobreajuste y garantiza decisiones de negocio más estables.
