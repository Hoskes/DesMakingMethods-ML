import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras  # Изменено на правильный импорт

# Часть 1 (нагенерили синтетических данных)
m = 50  # Количество точек
min_value = -100
max_value = 100
k_0_true = 2.54  # Истинное значение углового коэффициента
b_true = 1.0  # Истинное значение свободного члена

# Создание искусственного набора данных
x = np.linspace(min_value, max_value, m)  # Синтетические данные
noise = np.random.normal(0, 2, m)  # Уменьшение стандартного отклонения шума
y = k_0_true * x + b_true + noise  # Целевые значения с шумом

# Нормализация данных
x = (x - np.mean(x)) / np.std(x)  # Стандартизация x
y = (y - np.mean(y)) / np.std(y)  # Стандартизация y

# Визуализация данных
plt.scatter(x, y, label='Исходные данные')
plt.plot(x, (k_0_true * x + b_true - np.mean(y)) / np.std(y), color='red', label='Истинная модель (нормализованная)')
plt.legend()
plt.show()

# Часть 2
# Преобразование данных в тензоры TensorFlow
x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

# Инициализация переменных для обучения
k = tf.Variable(np.random.normal(0, 1), dtype=tf.float32)  # Инициализация случайными значениями
b = tf.Variable(np.random.normal(0, 1), dtype=tf.float32)

# Функция потерь
def reduce_loss(x, y):
    y_pred = k * x + b  # Упрощено умножение
    return tf.reduce_mean((y - y_pred) ** 2)

# Параметры обучения
learning_rate = 0.01  # Уменьшенная скорость обучения
num_iterations = 100  # Количество итераций при обучении

# Оптимизатор
optimizer = keras.optimizers.SGD(learning_rate)  # Изменено на правильный доступ

# Обучение модели
for step in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = reduce_loss(x_tensor, y_tensor)
    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    if (step + 1) % 10 == 0:
        print(f'Epoch {step + 1}, Loss: {loss.numpy()}, k: {k.numpy()}, b: {b.numpy()}')

# Визуализация результатов
plt.scatter(x, y, label='Исходные данные')
plt.plot(x, k.numpy() * x + b.numpy(), color='red', label='Обученная модель')
plt.plot(x, (k_0_true * x + b_true - np.mean(y)) / np.std(y), color='green', label='Истинная модель (нормализованная)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линейная регрессия с использованием TensorFlow')
plt.legend()
plt.show()
