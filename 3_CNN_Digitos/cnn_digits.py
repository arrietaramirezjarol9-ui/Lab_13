import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("images", exist_ok=True)

print("==========================================================")
print("LABORATORIO: CLASIFICACIÓN DE DÍGITOS USANDO CNN")
print("==========================================================\n")

# 1. Cargar dataset real desde UCI Machine Learning Repository
print("Descargando dataset 'Optical Recognition of Handwritten Digits' (ID: 80)...")
try:
    digits = fetch_ucirepo(id=80)
    X = digits.data.features
    y = digits.data.targets.squeeze()
except Exception as e:
    print(f"Error al descargar desde UCI: {e}")
    print("Creando dataset artificial para demostración en caso de fallo de red...")
    # Dataset sintético de dígitos 8x8
    n_samples = 1500
    X = pd.DataFrame(np.random.randint(0, 17, size=(n_samples, 64)))
    y = pd.Series(np.random.randint(0, 10, size=n_samples))

print('Dimensión de X:', X.shape)
print('Dimensión de y:', y.shape)

# 2. Exploración y representación en 8x8
X_images = X.to_numpy().reshape(-1, 8, 8, 1)
y_values = y.to_numpy().astype('int')

print('\nNueva forma de X_images:', X_images.shape)
print('Forma de y_values:', y_values.shape)

# Visualizar algunos ejemplos del dataset y guardarlos
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_images[i].reshape(8, 8), cmap='gray')
    plt.title(f'Dígito: {y_values[i]}')
    plt.axis('off')
plt.suptitle('Ejemplos de dígitos escritos a mano')
plt.savefig('images/digitos_ejemplos.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Normalización (píxeles van de 0 a 16)
X_images = X_images / 16.0
print('\nValores de píxeles normalizados:')
print('Valor mínimo:', X_images.min())
print('Valor máximo:', X_images.max())

# 4. Codificación One-Hot de las etiquetas
num_classes = 10
y_categorical = to_categorical(y_values, num_classes=num_classes)

# 5. División entrenamiento / prueba
X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(
    X_images,
    y_categorical,
    y_values,
    test_size=0.30,
    random_state=42,
    stratify=y_values,
)
print(f"\nEntrenamiento: {X_train.shape}")
print(f"Prueba: {X_test.shape}")

# 6. Construcción del modelo CNN
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (2, 2), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Entrenamiento de la CNN
print("\nEntrenando la CNN...")
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)
print("Entrenamiento finalizado.")

# 8. Visualizar curvas de entrenamiento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Loss entrenamiento', color='blue')
plt.plot(history.history['val_loss'], label='Loss validación', color='orange')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Curva de Pérdida de la CNN')
plt.legend()
plt.savefig('images/loss_curve_cnn.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Accuracy entrenamiento', color='blue')
plt.plot(history.history['val_accuracy'], label='Accuracy validación', color='orange')
plt.xlabel('Época')
plt.ylabel('Exactitud')
plt.title('Curva de Exactitud (Accuracy) de la CNN')
plt.legend()
plt.savefig('images/accuracy_curve_cnn.png', dpi=150, bbox_inches='tight')
plt.close()

# 9. Evaluación en datos de prueba
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\nEvaluación en prueba (CNN):')
print(f'Test loss    : {test_loss:.4f}')
print(f'Test accuracy: {test_accuracy:.4f}')

# Reporte de clasificación y guardar matriz de confusión
y_pred_prob = model.predict(X_test, verbose=0)
y_pred_labels = np.argmax(y_pred_prob, axis=1)

print('\nReporte de clasificación de la CNN:')
print(classification_report(y_test_labels, y_pred_labels))

plt.figure(figsize=(8, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test_labels,
    y_pred_labels,
    display_labels=list(range(10)),
    cmap='Blues'
)
plt.title('Matriz de confusión - CNN')
plt.savefig('images/confusion_matrix_cnn.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualizar predicciones individuales
plt.figure(figsize=(12, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_test[i].reshape(8, 8), cmap='gray')
    plt.title(f'Real: {y_test_labels[i]} | Pred: {y_pred_labels[i]}')
    plt.axis('off')
plt.suptitle('Visualización de Predicciones Individuales')
plt.savefig('images/predictions_samples.png', dpi=150, bbox_inches='tight')
plt.close()

# Visualizar errores cometidos
wrong_indices = np.where(y_test_labels != y_pred_labels)[0]
print(f"\nCantidad de errores cometidos por la CNN en el test set: {len(wrong_indices)}")

if len(wrong_indices) > 0:
    plt.figure(figsize=(12, 6))
    for plot_index, data_index in enumerate(wrong_indices[:15]):
        plt.subplot(3, 5, plot_index + 1)
        plt.imshow(X_test[data_index].reshape(8, 8), cmap='gray')
        plt.title(f'Real: {y_test_labels[data_index]} | Pred: {y_pred_labels[data_index]}')
        plt.axis('off')
    plt.suptitle('Ejemplos de errores (dígitos mal clasificados)')
    plt.savefig('images/errors_samples.png', dpi=150, bbox_inches='tight')
    plt.close()

# 10. Comparación con Red Densa Simple
print("\nEntrenando Red Densa Simple (Multicapa)...")
dense_model = Sequential([
    Flatten(input_shape=(8, 8, 1)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

dense_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

dense_history = dense_model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

dense_loss, dense_accuracy = dense_model.evaluate(X_test, y_test, verbose=0)
print(f'\nExactitud comparativa:')
print(f'Accuracy CNN:          {test_accuracy:.4f}')
print(f'Accuracy Red Densa:    {dense_accuracy:.4f}')

# 11. RETO MIT: Comparación experimental de variantes CNN
print("\n[Reto MIT] Evaluando 3 variantes de CNN...")

# Variante 1: CNN con una sola capa convolucional
model_v1 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Variante 2: CNN con dos capas convolucionales
model_v2 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (2, 2), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Variante 3: CNN con Dropout y más neuronas densas
model_v3 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (2, 2), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])

variantes = {
    'CNN 1 Capa Convolucional': model_v1,
    'CNN 2 Capas Convolucionales': model_v2,
    'CNN con Dropout y más neuronas': model_v3
}

mit_results = []

for nombre, m in variantes.items():
    print(f"Entrenando variante: {nombre}...")
    m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    l, acc = m.evaluate(X_test, y_test, verbose=0)
    preds = np.argmax(m.predict(X_test, verbose=0), axis=1)
    errs = np.sum(preds != y_test_labels)
    
    mit_results.append({
        'Variante': nombre,
        'Loss': l,
        'Accuracy': acc,
        'Errores': errs
    })

mit_df = pd.DataFrame(mit_results)
print("\nResultados del Reto MIT:")
print(mit_df)

# Guardar el modelo CNN principal
model.save('cnn_digits_uci.keras')
print("\nModelo guardado como 'cnn_digits_uci.keras' correctamente.")
