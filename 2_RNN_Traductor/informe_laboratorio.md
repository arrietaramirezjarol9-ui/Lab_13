# INFORME DE LABORATORIO: TRADUCTOR RUSO → ESPAÑOL CON RNN/LSTM

**UNIVERSIDAD CÉSAR VALLEJO**  
**Escuela Profesional de Ingeniería de Sistemas**  
* **Curso:** Sistemas Inteligentes  
* **Tema:** Redes Neuronales Recurrentes (RNN/LSTM)  
* **Producto:** Modelo Seq2Seq para traducción de frases cortas Ruso → Español  

---

## 1. Presentación de la Arquitectura
En este laboratorio se ha construido un traductor automático de frases cortas de ruso a español mediante una arquitectura de red recurrente **Seq2Seq (Sequence-to-Sequence)**. Esta arquitectura está compuesta por dos partes principales:
1. **Encoder (Codificador):** Lee la frase de entrada en ruso palabra por palabra y comprime la información secuencial en un vector de contexto (estados ocultos $h$ y $c$).
2. **Decoder (Decodificador):** Toma el vector de contexto del encoder y genera la frase traducida en español, palabra por palabra, condicionada por el token previo generado.

---

## 2. Desarrollo del Entorno y Preprocesamiento
El preprocesamiento de textos se realiza en el archivo [translator.py](file:///c:/Users/PC/Documents/Lab-13/2_RNN_Traductor/translator.py):
* **Agregar tokens especiales:** Se añade `<start>` al inicio y `<end>` al final de las frases de traducción para indicarle al decoder cuándo arrancar y cuándo terminar la traducción.
* **Tokenización:** Se mapean las palabras de ruso y español a índices numéricos utilizando `Tokenizer` de Keras.
* **Padding:** Dado que las frases tienen longitudes variables, se hace relleno con ceros (`padding='post'`) para que todas tengan la misma longitud máxima.

---

## 3. Pruebas de Traducción y Reto MIT
El dataset se amplió con **20 frases adicionales** en ruso/español relativas a acciones cotidianas y objetos (como "кот спит" -> "el gato duerme"). El modelo Seq2Seq se reentrenó durante 300 épocas.

### Resultados de Traducción:

#### Frases Conocidas (Éxito de Memorización):
* `привет` $\rightarrow$ `hola`
* `я тебя люблю` $\rightarrow$ `te quiero`
* `доброе утро` $\rightarrow$ `buenos días`
* `кот спит` $\rightarrow$ `el gato duerme`

#### Frases Desconocidas Fuera del Dataset (Limitación):
* `я люблю кофе` $\rightarrow$ `me gusta [palabra errónea o vacía]`
* `ты студент` $\rightarrow$ `hablas español [alucinación de palabras]`

*La alucinación ocurre porque el vocabulario es muy pequeño y el modelo no ha aprendido reglas gramaticales reales, sino que ha memorizado las asociaciones estadísticas directas.*

---

## 4. Cuestionario de Análisis

### 1. ¿Qué es una secuencia en el contexto de redes neuronales recurrentes?
Una secuencia es una serie ordenada de datos donde la posición y el orden de los elementos importan. En el procesamiento de lenguaje natural, un texto es una secuencia de palabras o caracteres. A diferencia de los datos tabulares independientes, el significado de cada palabra depende de las palabras que aparecieron antes, requiriendo modelos con "memoria" como las RNN.

### 2. ¿Qué diferencia existe entre una RNN simple y una LSTM?
Una **RNN simple** sufre del problema de desvanecimiento o explosión del gradiente, lo que hace que olvide la información lejana en secuencias largas. Una **LSTM (Long Short-Term Memory)** soluciona esto introduciendo una "celda de estado" central y tres compuertas lógicas (olvido, entrada y salida) que controlan de manera matemática qué información retener a largo plazo y cuál desechar.

### 3. ¿Qué función cumple el encoder?
El **encoder** procesa la secuencia de entrada (frase en ruso) paso a paso. En cada palabra, actualiza sus estados internos. Al llegar al final de la secuencia, los estados finales acumulados (vector de contexto) contienen un resumen codificado del significado completo de la frase de entrada.

### 4. ¿Qué función cumple el decoder?
El **decoder** recibe el vector de contexto del encoder como su estado inicial y genera la secuencia de salida (español) palabra por palabra. Utiliza como entrada en cada paso la palabra predicha en el paso anterior y actualiza su estado hasta predecir el token de finalización `<end>`.

### 5. ¿Por qué se usan los tokens `<start>` y `<end>`?
* `<start>` sirve como disparador o entrada inicial para el decoder en el primer paso de generación (cuando aún no tiene palabras previas).
* `<end>` sirve como criterio de parada. Le indica al algoritmo de búsqueda o inferencia que la frase ha concluido y debe dejar de predecir palabras.

### 6. ¿Por qué el modelo funciona mejor con frases que ya vio durante el entrenamiento?
Porque al tener un conjunto de datos extremadamente pequeño (menos de 100 frases), el modelo tiene la capacidad de memorizar los pesos exactos para mapear esas secuencias específicas (sobreajuste o memorización pura). No ha aprendido la semántica ni las reglas sintácticas del ruso, solo ha ajustado coeficientes para mapear entradas conocidas a salidas conocidas.

### 7. ¿Qué dificultades presenta traducir del ruso al español?
El ruso y el español pertenecen a familias lingüísticas muy diferentes. El ruso es una lengua altamente flexiva (declinaciones por caso, género y número para sustantivos y adjetivos), tiene un orden de palabras muy libre, carece de artículos (como "el" o "un") y usa el alfabeto cirílico. Mapear estas diferencias requiere grandes volúmenes de datos para capturar las correspondencias correctas.

### 8. ¿Por qué este enfoque fue importante antes de la aparición de Transformers?
Fue el primer enfoque exitoso capaz de procesar secuencias de longitudes variables tanto a la entrada como a la salida (Seq2Seq). Permitió que la traducción automática dejara de ser basada en reglas gramaticales rígidas escritas por lingüistas y pasara a ser completamente estadística y neuronal.

### 9. ¿Por qué este modelo no puede competir con Google Translate o DeepL?
Los traductores comerciales modernos se basan en arquitecturas de **Transformers** (con mecanismos de atención multicanal) entrenados con miles de millones de parámetros y conjuntos de datos gigantescos de la web. Tienen comprensión contextual, manejan homónimos y capturan el sentido global de textos largos, mientras que este modelo es una versión puramente didáctica de juguete.

### 10. ¿Qué necesitaría mejorar para acercarse a un traductor real?
Para convertirse en un traductor real, requeriría:
1. Ampliar el corpus de entrenamiento a millones de pares de oraciones.
2. Migrar de una arquitectura LSTM tradicional a una basada en **Transformers (mecanismos de auto-atención)**.
3. Utilizar representaciones de palabras preentrenadas (word embeddings como BERT o FastText).
4. Implementar técnicas de búsqueda más avanzadas como *Beam Search* en la decodificación en lugar de la búsqueda codiciosa (*greedy search*).

---

## 5. Conclusiones
El modelo Seq2Seq basado en LSTM demuestra con claridad la capacidad de las redes recurrentes para mapear secuencias de palabras de longitud variable. Aunque memoriza con facilidad el dataset pequeño y el Reto MIT, tiene dificultades extremas para generalizar a oraciones no vistas, demostrando que los modelos de lenguaje requieren una escala masiva de datos y arquitecturas más potentes (como Transformers) para ser prácticos en producción.
