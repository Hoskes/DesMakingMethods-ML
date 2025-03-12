import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Параметры
SEQ_LENGTH = 100
BATCH_SIZE = 256
EMBEDDING_DIM = 64
RNN_UNITS = 128
EPOCHS = 1

# 1. Загрузка и подготовка данных
df = pd.read_csv('lenta-ru.csv', usecols=['text'], encoding='utf-8')
texts = df['text'].str.lower().dropna().str.replace('[^\w\s]', '', regex=True).tolist()

# 2. Токенизация на уровне символов
chars = sorted(set(''.join(texts)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
VOCAB_SIZE = len(chars)

# 3. Векторизация данных
encoded_texts = [np.array([char_to_idx[c] for c in text if c in char_to_idx], dtype=np.int32) for text in texts]

# 4. Создание генератора последовательностей
train_gen = TimeseriesGenerator(
    np.concatenate(encoded_texts),
    np.concatenate(encoded_texts),
    length=SEQ_LENGTH,
    sampling_rate=1,
    stride=1,
    batch_size=BATCH_SIZE
)

# 5. Создание модели
model = Sequential([
    Embedding(VOCAB_SIZE, EMBEDDING_DIM),  # Убран input_length
    LSTM(RNN_UNITS * 2, return_sequences=True),
    LSTM(RNN_UNITS),
    Dense(VOCAB_SIZE, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Обучение с оптимизацией
model.fit(
    train_gen,
    epochs=EPOCHS,
    callbacks=[EarlyStopping(patience=3)]
)

# 7. Генерация текста
def generate_text(seed, length=500, temperature=0.7):
    seed = seed.lower().ljust(SEQ_LENGTH)[-SEQ_LENGTH:]
    generated = []
    seed_encoded = np.array([[char_to_idx.get(c, 0) for c in seed]], dtype=np.int32)

    model.reset_states()
    for _ in range(length):
        probs = model.predict(seed_encoded, verbose=0)[0]
        probs = np.exp(np.log(probs) / temperature)
        probs /= probs.sum()

        next_idx = np.random.choice(VOCAB_SIZE, p=probs)
        generated.append(idx_to_char[next_idx])

        seed_encoded = np.roll(seed_encoded, -1)
        seed_encoded[0, -1] = next_idx

    return ''.join(generated)

# Пример генерации
print(generate_text("россия", length=50))
