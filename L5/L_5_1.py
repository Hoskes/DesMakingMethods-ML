import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Загрузка данных из CSV
df = pd.read_csv('lenta-ru.csv')
text = ' '.join(df['text'].astype(str).tolist())

# Предобработка текста
text = text.replace('\n', ' ')  # Удаляем переносы строк

# Создание словаря символов
chars = sorted(list(set(text)))
char_to_idx = {char: i for i, char in enumerate(chars)}
idx_to_char = {i: char for i, char in enumerate(chars)}

# Преобразование текста в числовые индексы
data = [char_to_idx[char] for char in text]

# Параметры последовательности
seq_length = 6
n_chars = len(chars)

# Создание обучающих данных
X = []
y = []
for i in range(len(data) - seq_length):
    seq_in = data[i:i + seq_length]
    seq_out = data[i + seq_length]
    X.append(seq_in)
    y.append(seq_out)

# Преобразование в one-hot кодирование
X_one_hot = np.zeros((len(X), seq_length, n_chars), dtype=np.bool_)
y_one_hot = np.zeros((len(y), n_chars), dtype=np.bool_)

for i, seq in enumerate(X):
    for t, idx in enumerate(seq):
        X_one_hot[i, t, idx] = 1
    y_one_hot[i, y[i]] = 1

# Создание модели LSTM
model = Sequential()
model.add(LSTM(128, input_shape=(seq_length, n_chars), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(n_chars, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Обучение модели
model.fit(X_one_hot, y_one_hot, epochs=100, batch_size=32)


# Генерация текста
def generate_text(model, start_seq, num_chars=100):
    generated = start_seq
    for _ in range(num_chars):
        x = np.zeros((1, seq_length, n_chars))
        for t, char in enumerate(generated[-seq_length:]):
            idx = char_to_idx[char]
            x[0, t, idx] = 1
        preds = model.predict(x, verbose=0)[0]
        next_idx = np.argmax(preds)
        next_char = idx_to_char[next_idx]
        generated += next_char
    return generated


# Оценка перплексии
def calculate_perplexity(model, text):
    total_log_prob = 0
    n = len(text) - seq_length
    for i in range(n):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]

        x = np.zeros((1, seq_length, n_chars))
        for t, char in enumerate(seq_in):
            idx = char_to_idx[char]
            x[0, t, idx] = 1

        preds = model.predict(x, verbose=0)[0]
        prob = preds[char_to_idx[seq_out]]
        total_log_prob += np.log(prob + 1e-10)  # Добавление небольшого значения для избежания логарифма нуля

    perplexity = np.exp(-total_log_prob / n)
    return perplexity


# Пример генерации и оценки
start_sequences = ["Текст", "На рек", "Невский"]
for start_sequence in start_sequences:
    generated_text = generate_text(model, start_sequence)
    print(f"Сгенерированный текст для '{start_sequence}':")
    print(generated_text)

    # Оценка перплексии
    perplexity = calculate_perplexity(model, generated_text)
    print(f"Перплексия: {perplexity:.2f}")
