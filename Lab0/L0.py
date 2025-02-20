import numpy as np
import matplotlib.pyplot as plt
from keras.src.legacy.backend import gradients

# Часть 1 (нагенерили синтетических данных)
m = 50  # Количество точек
min_value = -100
max_value = 100
k_0_true = 2.0  # Истинное значение углового коэффициента
b_true = 1.0  # Истинное значение свободного члена


# Создание искусственного набора данных
x = np.linspace(min_value, max_value, m)  # Синтетические данные
noise = np.random.normal(0, 4, m)  # Шум


def calc_lin_regression_with_mess(x, noise):
    return k_0_true * x + b_true + noise


def calc_true_lin_regression(x):
    return k_0_true * x + b_true


# Визуализация данных
plt.scatter(x, calc_lin_regression_with_mess(x, noise), label='Исходные данные')
plt.plot(x, calc_true_lin_regression(x), color='red', label='Истинная модель')
plt.legend()
plt.show()

# Часть 2
import tensorflow as tf
from keras import optimizers

# x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
# y_tensor = tf.convert_to_tensor(calc_true_lin_regression(x), dtype=tf.float32)

k = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)



def reduce_loss(x,noise):
    y = calc_lin_regression_with_mess(x,noise)
    y_pred = calc_true_lin_regression(x)
    return tf.reduce_mean((y-y_pred)**2)

learning_rate = 0.1  # Скорость обучения
num_iterations = 1000  # Количество итераций при обучении


optimizer = optimizers.SGD(learning_rate)

for step in range(num_iterations):
    with tf.GradientTape() as tape:
        loss = reduce_loss(x,noise)
    gradients = tape.gradient(loss, [k,b])
    optimizer.apply_gradients(zip(gradients, [k,b]))
    if (step + 1) % 100 == 0:
        print(f'Epoch {step + 1}, Loss: {loss.numpy()}, k: {k.value()}, b: {b.value()}')

# 4. Визуализация результатов
plt.scatter(x, calc_lin_regression_with_mess(x,noise), label='Исходные данные')
plt.plot(x, k.value() * x + b.value(), color='red', label='Обученная модель')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Линейная регрессия с использованием TensorFlow')
plt.legend()
plt.show()