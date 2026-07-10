import os
import re
import zipfile
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

# Fijar semillas para reproducibilidad
np.random.seed(42)

os.makedirs("images", exist_ok=True)

print("==========================================================")
print("LABORATORIO: NLP CLÁSICO Y MLP PARA ANÁLISIS DE SENTIMIENTOS")
print("==========================================================\n")

# 1. Cargar dataset desde UCI o usar respaldo
def load_uci_sentiment_dataset():
    url = 'https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip'
    zip_path = Path('sentiment_labelled_sentences.zip')
    extract_dir = Path('sentiment_labelled_sentences')
    
    print("Intentando descargar dataset original de UCI...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    possible_files = [
        extract_dir / 'sentiment labelled sentences' / 'amazon_cells_labelled.txt',
        extract_dir / 'sentiment labelled sentences' / 'imdb_labelled.txt',
        extract_dir / 'sentiment labelled sentences' / 'yelp_labelled.txt',
    ]
    
    frames = []
    for file_path in possible_files:
        if file_path.exists():
            temp_df = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=['text', 'sentiment']
            )
            temp_df['source'] = file_path.stem.replace('_labelled', '')
            frames.append(temp_df)
            
    if not frames:
        raise FileNotFoundError('No se encontraron archivos en la extracción del ZIP de UCI.')
        
    return pd.concat(frames, ignore_index=True)

def load_fallback_dataset():
    print("Usando dataset de respaldo integrado...")
    examples = [
        ('I love this product', 1, 'fallback'),
        ('This is an excellent service', 1, 'fallback'),
        ('The movie was amazing', 1, 'fallback'),
        ('The food was delicious', 1, 'fallback'),
        ('I am very happy with the purchase', 1, 'fallback'),
        ('The staff was friendly', 1, 'fallback'),
        ('This phone works perfectly', 1, 'fallback'),
        ('The experience was wonderful', 1, 'fallback'),
        ('I would buy it again', 1, 'fallback'),
        ('Highly recommended', 1, 'fallback'),
        ('I hate this product', 0, 'fallback'),
        ('This is a terrible service', 0, 'fallback'),
        ('The movie was boring', 0, 'fallback'),
        ('The food was awful', 0, 'fallback'),
        ('I am very disappointed', 0, 'fallback'),
        ('The staff was rude', 0, 'fallback'),
        ('This phone does not work', 0, 'fallback'),
        ('The experience was horrible', 0, 'fallback'),
        ('I would never buy this again', 0, 'fallback'),
        ('Not recommended', 0, 'fallback'),
    ]
    # Duplicamos un poco para tener suficientes muestras para entrenar
    examples = examples * 10
    return pd.DataFrame(examples, columns=['text', 'sentiment', 'source'])

try:
    df = load_uci_sentiment_dataset()
    print("Dataset UCI cargado correctamente.")
except Exception as error:
    print('No se pudo cargar UCI. Detalle:', error)
    df = load_fallback_dataset()

print('Dimensión del dataset:', df.shape)
print("\nDistribución de sentimientos (0 = Negativo, 1 = Positivo):")
print(df['sentiment'].value_counts())

# 2. Limpieza de texto
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
print("\nEjemplo de texto limpio:")
print(df[['text', 'clean_text', 'sentiment']].head(5))

# 3. División entrenamiento / prueba
X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)
print(f"\nEntrenamiento: {X_train.shape}")
print(f"Prueba: {X_test.shape}")

# 4. Vectorización con TF-IDF
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words='english',
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print('\nMatriz de características TF-IDF:')
print('Entrenamiento:', X_train_tfidf.shape)
print('Prueba       :', X_test_tfidf.shape)

# 5. Entrenamiento del Perceptrón Multicapa (MLP)
print("\nEntrenando clasificador MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
)

mlp.fit(X_train_tfidf, y_train)
print('Entrenamiento finalizado.')

# 6. Evaluación del modelo
y_pred = mlp.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f'\nResultados en prueba:')
print(f'Accuracy : {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall   : {recall:.4f}')
print(f'F1-score : {f1:.4f}')

print('\nReporte de clasificación:')
print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

# Graficar Matriz de confusión
plt.figure(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=['Negativo', 'Positivo'],
    cmap='Blues'
)
plt.title('Matriz de confusión - Sentimientos')
plt.savefig('images/confusion_matrix_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

# Graficar Curva de pérdida
plt.figure(figsize=(8, 5))
plt.plot(mlp.loss_curve_, label='Loss', color='red')
plt.title('Curva de pérdida del MLP')
plt.xlabel('Iteración')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.savefig('images/loss_curve_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

# 7. Análisis de errores
error_df = pd.DataFrame({
    'text': X_test.values,
    'real': y_test.values,
    'predicted': y_pred,
})
error_df = error_df[error_df['real'] != error_df['predicted']]
print(f"\nCantidad total de errores cometidos: {len(error_df)}")
print("Ejemplos de errores cometidos por el modelo:")
print(error_df.head(10))

# 8. Comparación de arquitecturas
print("\nComparando configuraciones de capas ocultas...")
architectures = [
    (16,),
    (64, 32),
    (128, 64, 32),
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
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    
    results.append({
        'architecture': str(architecture),
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, zero_division=0),
        'recall': recall_score(y_test, predictions, zero_division=0),
        'f1_score': f1_score(y_test, predictions, zero_division=0),
        'iterations': model.n_iter_,
    })

results_df = pd.DataFrame(results)
print("\nTabla Comparativa:")
print(results_df)

# Graficar comparación y guardar
results_df.set_index('architecture')[['accuracy', 'precision', 'recall', 'f1_score']].plot(
    kind='bar',
    figsize=(10, 6)
)
plt.title('Comparación de Métricas por Arquitectura MLP')
plt.xlabel('Arquitectura')
plt.ylabel('Métrica')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('images/architecture_comparison_sentiment.png', dpi=150, bbox_inches='tight')
plt.close()

# 9. Pipeline completo (para guardar vectorizador y modelo juntos)
print("\nEntrenando Pipeline final guardable...")
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english', preprocessor=clean_text)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42, early_stopping=True))
])

final_pipeline.fit(X_train, y_train)

# 10. Probar inferencias con comentarios escritos por el estudiante (MIT Challenge)
print("\n[Reto MIT] Probando comentarios nuevos:")
sample_comments = [
    'The product is wonderful',
    'The product is awful',
    'The service was fast and excellent',
    'I am disappointed with the quality',
    'I will never buy this again',
    'It works okay, but the delivery took ages',
    'Highly recommended, best purchase ever!',
    'Waste of money and time',
    'Not bad, but could be much better',
    'I absolutely love how easy this is to use'
]

pipeline_predictions = final_pipeline.predict(sample_comments)
for comment, prediction in zip(sample_comments, pipeline_predictions):
    label = 'Positivo' if prediction == 1 else 'Negativo'
    print(f"  '{comment}' -> {label}")

# Guardar pipeline
joblib.dump(final_pipeline, 'sentiment_mlp_pipeline.joblib')
print("\nPipeline guardado como 'sentiment_mlp_pipeline.joblib' correctamente.")
