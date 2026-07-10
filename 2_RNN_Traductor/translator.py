import os
import sys
# Reconfigurar salida estándar para soportar caracteres cirílicos en Windows
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

# Fijar semillas para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

os.makedirs("images", exist_ok=True)

print("==========================================================")
print("LABORATORIO: TRADUCTOR BÁSICO RUSO -> ESPAÑOL CON RNN/LSTM")
print("==========================================================\n")

# 1. Dataset base (de la guía)
translation_pairs = [
    ('привет', 'hola'),
    ('здравствуй', 'hola'),
    ('доброе утро', 'buenos días'),
    ('добрый день', 'buenas tardes'),
    ('добрый вечер', 'buenas noches'),
    ('спокойной ноchi', 'buenas noches'),
    ('спасибо', 'gracias'),
    ('большое спасибо', 'muchas gracias'),
    ('пожалуйста', 'por favor'),
    ('извини', 'lo siento'),
    ('как дела', 'cómo estás'),
    ('я хорошо', 'estoy bien'),
    ('я плохо', 'estoy mal'),
    ('я устал', 'estoy cansado'),
    ('я голоден', 'tengo hambre'),
    ('я хочу воды', 'quiero agua'),
    ('я хочу есть', 'quiero comer'),
    ('где туалет', 'dónde está el baño'),
    ('сколько стоит', 'cuánto cuesta'),
    ('это дорого', 'esto es caro'),
    ('это дешево', 'esto es barato'),
    ('я студент', 'soy estudiante'),
    ('я преподаватель', 'soy profesor'),
    ('я из перу', 'soy de perú'),
    ('ты говоришь по испански', 'hablas español'),
    ('я говорю по русски', 'hablo ruso'),
    ('я не понимаю', 'no entiendo'),
    ('повтори пожалуйста', 'repite por favor'),
    ('я тебя люблю', 'te quiero'),
    ('я скучаю по тебе', 'te extraño'),
    ('до свидания', 'adiós'),
    ('увидимся mañana', 'nos vemos mañana'),
    ('сегодня холодно', 'hoy hace frío'),
    ('сегодня жарко', 'hoy hace calor'),
    ('мне нравится музыка', 'me gusta la música'),
    ('мне нравится кофе', 'me gusta el café'),
    ('я люблю читать', 'me gusta leer'),
    ('я люблю путешествовать', 'me gusta viajar'),
    ('это мой друг', 'este es mi amigo'),
    ('это моя подруga', 'esta es mi amiga'),
    ('у меня есть вопрос', 'tengo una pregunta'),
    ('помоги мне', 'ayúdame'),
    ('открой дверь', 'abre la puerta'),
    ('закрой дверь', 'cierra la puerta'),
    ('я дома', 'estoy en casa'),
    ('я в университете', 'estoy en la universidad'),
    ('я работаю', 'estoy trabajando'),
    ('я учусь', 'estoy estudiando'),
    ('мне нужно идти', 'tengo que irme'),
    ('до завтра', 'hasta mañana'),
]

# RETO MIT: Ampliar el dataset con 20 frases nuevas
mit_pairs = [
    ('я ем яблоко', 'como una manzana'),
    ('ты мой брат', 'eres mi hermano'),
    ('она моя сестра', 'ella es mi hermana'),
    ('кот спит', 'el gato duerme'),
    ('собака бежит', 'el perro corre'),
    ('книга на столе', 'el libro está sobre la mesa'),
    ('погода хорошая', 'el clima es bueno'),
    ('я хочу спать', 'quiero dormir'),
    ('где вокзал', 'dónde está la estación'),
    ('мне нужен билет', 'necesito un boleto'),
    ('какой сегодня день', 'qué día es hoy'),
    ('я пишу письмо', 'escribo una carta'),
    ('мы пьем чай', 'bebemos té'),
    ('они играют', 'ellos juegan'),
    ('машина быстрая', 'el carro es rápido'),
    ('мне холодно', 'tengo frío'),
    ('мне тепло', 'tengo calor'),
    ('помощь нужна', 'se necesita ayuda'),
    ('это легко', 'esto es fácil'),
    ('это трудно', 'esto es difícil')
]

print(f"Dataset base: {len(translation_pairs)} frases.")
print(f"Ampliación Reto MIT: {len(mit_pairs)} frases.")

# Unimos ambos conjuntos para el entrenamiento final
full_pairs = translation_pairs + mit_pairs
print(f"Dataset total ampliado: {len(full_pairs)} frases.\n")

df = pd.DataFrame(full_pairs, columns=['ruso', 'espanol'])

# 2. Agregar tokens especiales <start> y <end>
df['target_input'] = '<start> ' + df['espanol']
df['target_output'] = df['espanol'] + ' <end>'

# 3. Tokenización
input_tokenizer = Tokenizer(filters='')
input_tokenizer.fit_on_texts(df['ruso'])

target_tokenizer = Tokenizer(filters='')
target_tokenizer.fit_on_texts(pd.concat([df['target_input'], df['target_output']]))

input_sequences = input_tokenizer.texts_to_sequences(df['ruso'])
target_input_sequences = target_tokenizer.texts_to_sequences(df['target_input'])
target_output_sequences = target_tokenizer.texts_to_sequences(df['target_output'])

# 4. Padding
max_encoder_seq_length = max(len(seq) for seq in input_sequences)
max_decoder_seq_length = max(len(seq) for seq in target_input_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(target_input_sequences, maxlen=max_decoder_seq_length, padding='post')
decoder_target_data = pad_sequences(target_output_sequences, maxlen=max_decoder_seq_length, padding='post')

num_encoder_tokens = len(input_tokenizer.word_index) + 1
num_decoder_tokens = len(target_tokenizer.word_index) + 1

print('Longitud máxima encoder (Ruso):', max_encoder_seq_length)
print('Longitud máxima decoder (Español):', max_decoder_seq_length)
print('Vocabulario ruso:', num_encoder_tokens)
print('Vocabulario español:', num_decoder_tokens)

# Ajustar dimensión para sparse_categorical_crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1)

# 5. Construcción del modelo Encoder-Decoder con LSTM
embedding_dim = 64
latent_dim = 128

# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(
    input_dim=num_encoder_tokens,
    output_dim=embedding_dim,
    mask_zero=True,
    name='encoder_embedding'
)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding_layer = Embedding(
    input_dim=num_decoder_tokens,
    output_dim=embedding_dim,
    mask_zero=True,
    name='decoder_embedding'
)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(
    latent_dim,
    return_sequences=True,
    return_state=True,
    name='decoder_lstm'
)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Compilar
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 6. Entrenamiento
print("\nEntrenando el modelo Seq2Seq (300 épocas)...")
history = model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=8,
    epochs=300,
    validation_split=0.1,
    verbose=0
)
print("Entrenamiento finalizado.")

# 7. Graficar curvas de pérdida y exactitud
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Entrenamiento', color='blue')
plt.plot(history.history['val_loss'], label='Validación', color='orange')
plt.xlabel('Época')
plt.ylabel('Pérdida (Loss)')
plt.title('Curva de Pérdida del Traductor LSTM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/loss_curve_translator.png', dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Entrenamiento', color='blue')
plt.plot(history.history['val_accuracy'], label='Validación', color='orange')
plt.xlabel('Época')
plt.ylabel('Exactitud (Accuracy)')
plt.title('Curva de Exactitud del Traductor LSTM')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('images/accuracy_curve_translator.png', dpi=150, bbox_inches='tight')
plt.close()

# 8. Modelos de inferencia
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,), name='decoder_input_single')
decoder_embedding_single = decoder_embedding_layer(decoder_inputs_single)

decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(
    decoder_embedding_single,
    initial_state=decoder_states_inputs
)
decoder_states_single = [state_h_single, state_c_single]
decoder_outputs_single = decoder_dense(decoder_outputs_single)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs_single] + decoder_states_single
)

reverse_target_word_index = {index: word for word, index in target_tokenizer.word_index.items()}
start_token = target_tokenizer.word_index['<start>']
end_token = target_tokenizer.word_index['<end>']

def translate(sentence):
    # Tokenización y padding de entrada
    sequence = input_tokenizer.texts_to_sequences([sentence])
    sequence = pad_sequences(sequence, maxlen=max_encoder_seq_length, padding='post')
    
    # Predecir estados iniciales del decoder usando el encoder
    states_value = encoder_model.predict(sequence, verbose=0)
    
    # Generar secuencia de salida vacía con el token inicial
    target_sequence = np.array([[start_token]])
    translated_words = []
    
    for _ in range(max_decoder_seq_length):
        output_tokens, h, c = decoder_model.predict([target_sequence] + states_value, verbose=0)
        
        # Muestrear el token de mayor probabilidad
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        
        if sampled_token_index == end_token:
            break
            
        sampled_word = reverse_target_word_index.get(sampled_token_index, '')
        if sampled_word:
            translated_words.append(sampled_word)
            
        # Actualizar secuencia objetivo con el token generado
        target_sequence = np.array([[sampled_token_index]])
        states_value = [h, c]
        
    return ' '.join(translated_words)

# 9. Pruebas de traducción
print("\nPruebas con frases del dataset base:")
test_sentences = [
    'привет',
    'спасибо',
    'я тебя люблю',
    'доброе утро',
    'я студент',
    'я не понимаю',
]
for sentence in test_sentences:
    print(f"  {sentence} -> {translate(sentence)}")

print("\nPruebas con nuevas frases agregadas en el Reto MIT:")
test_mit_sentences = [
    'я ем яблоко',
    'кот спит',
    'где вокзал',
    'это легко',
]
for sentence in test_mit_sentences:
    print(f"  {sentence} -> {translate(sentence)}")

print("\nPruebas fuera del dataset (limitación del modelo):")
unknown_sentences = [
    'я люблю кофе',
    'ты студент',
    'мне нравится русский',
]
for sentence in unknown_sentences:
    print(f"  {sentence} -> {translate(sentence)}")

# 10. Guardar modelo
model.save('russian_spanish_seq2seq_lstm.keras')
print("\nModelo guardado como 'russian_spanish_seq2seq_lstm.keras' correctamente.")
