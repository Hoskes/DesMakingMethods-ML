import numpy as np
import matplotlib.pyplot as plt

# Параметры
m = 100
min_value = -10
max_value = 10

# Создание тензоров
uniform_tensor = np.random.uniform(min_value, max_value, m)
normal_tensor = np.random.normal(0, 1, m)

# Параметры модели линейной регрессии
k_0 = 2.0
b = 3.0

# Генерация зависимой переменной
y = k_0 * uniform_tensor + b + normal_tensor

# Визуализация
plt.scatter(uniform_tensor, y, label='Данные', color='blue')
plt.title('Искусственный набор данных')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
import tensorflow as tf

# Инициализация переменных
k = tf.Variable(np.random.randn(), dtype=tf.float32)
b = tf.Variable(np.random.randn(), dtype=tf.float32)

# Функция потерь
def loss_fn():
    y_pred = k * uniform_tensor + b
    return tf.reduce_mean(tf.square(y_pred - y))

# Оптимизатор
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# Тренировка модели
epochs = 1000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn()
    gradients = tape.gradient(loss, [k, b])
    optimizer.apply_gradients(zip(gradients, [k, b]))

# Визуализация результата
y_pred = k.numpy() * uniform_tensor + b.numpy()

plt.scatter(uniform_tensor, y, label='Данные', color='blue')
plt.plot(uniform_tensor, y_pred, label='Аппроксимация', color='red')
plt.title('Результаты градиентного спуска')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
