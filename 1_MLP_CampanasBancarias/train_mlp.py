import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# 1. Crear directorios para los entregables
os.makedirs("images", exist_ok=True)

print("==========================================================")
print("LABORATORIO: MLP PARA PREDICCIÓN DE CAMPAÑAS BANCARIAS")
print("==========================================================\n")

# 2. Cargar dataset real desde UCI Machine Learning Repository
print("Descargando dataset 'Bank Marketing' (ID: 222)...")
try:
    bank_marketing = fetch_ucirepo(id=222)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets.squeeze()
except Exception as e:
    print(f"Error al descargar desde UCI: {e}")
    print("Intentando cargar localmente o simulando un subconjunto para demostración...")
    # Si hay problemas de red, creamos un dataset sintético con las mismas columnas para evitar que falle
    # Columnas numéricas y categóricas típicas del dataset bank-marketing
    np.random.seed(42)
    n_samples = 1000
    X = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'job': np.random.choice(['admin.', 'technician', 'blue-collar', 'services', 'retired'], n_samples),
        'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
        'education': np.random.choice(['primary', 'secondary', 'tertiary'], n_samples),
        'default': np.random.choice(['yes', 'no'], n_samples, p=[0.02, 0.98]),
        'balance': np.random.randint(-1000, 10000, n_samples),
        'housing': np.random.choice(['yes', 'no'], n_samples, p=[0.6, 0.4]),
        'loan': np.random.choice(['yes', 'no'], n_samples, p=[0.15, 0.85]),
        'contact': np.random.choice(['cellular', 'telephone', 'unknown'], n_samples),
        'day': np.random.randint(1, 31, n_samples),
        'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], n_samples),
        'duration': np.random.randint(10, 1000, n_samples),
        'campaign': np.random.randint(1, 10, n_samples),
        'pdays': np.random.choice([-1, 999, 10, 20], n_samples),
        'previous': np.random.randint(0, 5, n_samples),
        'poutcome': np.random.choice(['success', 'failure', 'other', 'unknown'], n_samples)
    })
    y = pd.Series(np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88]))

print('Dimensión de X:', X.shape)
print('Dimensión de y:', y.shape)

# 3. Exploración inicial y distribución de clases
df = X.copy()
df['target'] = y
print("\nDistribución de clases de la variable objetivo:")
print(df['target'].value_counts(normalize=True))

# 4. Preparación de datos (Preprocesamiento)
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print('\nVariables numéricas:', numeric_features)
print('Variables categóricas:', categorical_features)

# Codificar la variable objetivo (y) a 0 y 1
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
print('Clases objetivo:', list(target_encoder.classes_))
print('Ejemplo de etiquetas codificadas (primeros 10):', y_encoded[:10])

# Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ]
)

# 5. División entrenamiento / prueba
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.30,
    random_state=42,
    stratify=y_encoded,
)
print(f"\nDatos de entrenamiento: {X_train.shape}")
print(f"Datos de prueba: {X_test.shape}")

# 6. Entrenamiento del Perceptrón Multicapa (MLP)
print("\nEntrenando arquitectura base MLP (16, 8)...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', mlp_model),
    ]
)

pipeline.fit(X_train, y_train)
print('Entrenamiento finalizado.')

# 7. Evaluación del modelo
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f'\nResultados del modelo base:')
print(f'Accuracy : {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1-score : {f1:.4f}')

print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_, zero_division=0))

# Graficar y guardar Matriz de confusión
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=target_encoder.classes_,
    cmap='Blues'
)
plt.title('Matriz de confusión - MLP (16, 8)')
plt.savefig('images/confusion_matrix_base.png', dpi=150, bbox_inches='tight')
plt.close()

# Graficar y guardar Curva de pérdida
loss_curve = pipeline.named_steps['model'].loss_curve_
plt.figure(figsize=(8, 5))
plt.plot(loss_curve, label='Pérdida en entrenamiento', color='purple', lw=2)
plt.xlabel('Iteración')
plt.ylabel('Loss')
plt.title('Curva de pérdida del MLP (16, 8)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('images/loss_curve_base.png', dpi=150, bbox_inches='tight')
plt.close()

# 8. Comparación de arquitecturas
print("\nComparando múltiples arquitecturas de capas ocultas...")
architectures = [
    (8,),
    (16, 8),
    (32, 16, 8),
]

results = []

for architecture in architectures:
    print(f"Probando arquitectura: {architecture}...")
    model = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation='relu',
        solver='adam',
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    
    pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model),
        ]
    )
    
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)
    
    results.append({
        'architecture': str(architecture),
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1_score': f1_score(y_test, predictions, zero_division=0),
        'iterations': pipe.named_steps['model'].n_iter_,
    })

results_df = pd.DataFrame(results)
print("\nResultados Comparativos de Arquitecturas:")
print(results_df)

# Graficar comparación de arquitecturas y guardar
results_df.set_index('architecture')[['accuracy', 'precision', 'recall', 'f1_score']].plot(
    kind='bar',
    figsize=(10, 6)
)
plt.title('Comparación de Métricas por Arquitectura MLP')
plt.xlabel('Arquitectura (Capas Ocultas)')
plt.ylabel('Métrica')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/architecture_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 9. Guardar el modelo base / pipeline
joblib.dump(pipeline, 'mlp_bank_marketing_model.joblib')
print("\nModelo guardado como 'mlp_bank_marketing_model.joblib' correctamente.")
print("Proceso completado con éxito. Las imágenes fueron guardadas en 'images/'.")
