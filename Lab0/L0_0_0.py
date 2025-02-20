import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers

# Часть 1 (нагенерили синтетических данных)
m = 50  # Количество точек
min_value = -100
max_value = 100
k_0_true = 2.54  # Истинное значение углового коэффициента
b_true = 1.0  # Истинное значение свободного члена
noise_value = 1 # Отклонение шума

learning_rate = 0.01  # Уменьшенная скорость обучения
num_iterations = 1000  # Количество итераций при обучении

# Создание искусственного набора данных
x = np.linspace(min_value, max_value, m)  # Синтетические данные
noise = np.random.normal(0, noise_value, m)  # Уменьшение стандартного отклонения шума

# Нормализация данных
x = (x - np.mean(x)) / np.std(x)

# Визуализация данных
plt.scatter(x, k_0_true * x + b_true + noise, label='Исходные данные')
plt.plot(x, k_0_true * x + b_true, color='red', label='Истинная модель')
plt.legend()
plt.show()

# Часть 2
# Преобразование данных в тензоры TensorFlow
x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
noise_tensor = tf.convert_to_tensor(noise, dtype=tf.float32)

# Инициализация переменных для обучения
k = tf.Variable(np.random.normal(0, 1), dtype=tf.float32)  # Инициализация близко к нулю
b = tf.Variable(np.random.normal(0, 1), dtype=tf.float32)


def reduce_loss(x, noise):
    # Используем истинные значения с шумом
    y = k_0_true * x + b_true + noise
    y_pred = tf.multiply(k, x) + b
    return tf.reduce_mean((y - y_pred) ** 2)


optimizer = optimizers.SGD(learning_rate)

for step in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = reduce_loss(x_tensor, noise_tensor)  # Используем тензоры
    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

    # Проверка на nan
    if gradients[0] is None or gradients[1] is None:
        print("Градиенты равны None. Проверьте вычисления.")
        break

    if (step + 1) % 100 == 0:
        print(f'Epoch {step + 1}, Loss: {loss.numpy()}, k: {k.numpy()}, b: {b.numpy()}')

# Визуализация результатов
plt.scatter(x, k_0_true * x + b_true + noise, label='Исходные данные')
plt.plot(x, k.numpy() * x + b.numpy(), color='green', label='Обученная модель')
plt.plot(x, k_0_true * x + b_true, color='red', label='Истинная модель')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линейная регрессия с использованием TensorFlow')
plt.legend()
plt.show()
