# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Этап 0: Загрузка и предобработка данных

# Загрузка данных
df = pd.read_csv("diabetes.csv")

# Замена нулевых значений на медиану для выбранных признаков
columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for column in columns_to_fix:
    df[column] = df[column].replace(0, df[column].median())

# Разделение на признаки и целевую переменную
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Этап 1: Нормировка данных

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Этап 2: Обучение модели и предсказание

# Создание и обучение модели с K=5
kn = 5
knn = KNeighborsClassifier(n_neighbors=kn)
knn.fit(X_train_scaled, y_train)

# Предсказание для тестовой выборки
y_pred = knn.predict(X_test_scaled)

# Этап 3: Оценка качества модели

# Расчет точности
accuracy = accuracy_score(y_test, y_pred)
print(f"[1] Модель с K=",kn)
print(f"Точность (Accuracy): {accuracy * 100:.2f}%")

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nМатрица ошибок:")
print(conf_matrix)  # Исправлено: убрано лишнее форматирование

# Подбор оптимального K с кросс-валидацией

k_values = list(range(1, 50))
mean_accuracies = []

print("\nПоиск оптимального K:")
for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        knn_temp,
        X_train_scaled,
        y_train,
        cv=5,
        scoring="accuracy"
    )
    mean_accuracy = scores.mean()
    std_accuracy = scores.std()
    mean_accuracies.append(mean_accuracy)

    # Вывод для каждой итерации
    print(f"K = {k:2d} | "
          f"Точность: {mean_accuracy * 100:.2f}% (±{std_accuracy * 100:.2f}%) | "
          f"Оценки: {[f'{s * 100:.1f}%' for s in scores]}")

# Визуализация результатов
plt.figure(figsize=(12, 6))
plt.plot(k_values, mean_accuracies, marker='o', linestyle='--')
plt.title("Зависимость точности от количества соседей")
plt.xlabel("Количество соседей (K)")
plt.ylabel("Средняя точность")
plt.xticks(k_values)
plt.grid(True)
plt.show()

# Выбор оптимального K
optimal_k = k_values[np.argmax(mean_accuracies)]
print(f"\n[2] Оптимальное значение K: {optimal_k}")

# Проверка модели с оптимальным K
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train_scaled, y_train)
y_pred_final = final_knn.predict(X_test_scaled)

# Оценка финальной модели
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\n[3] Модель с оптимальным K={optimal_k}")
print(f"Точность (Accuracy): {final_accuracy * 100:.2f}%")
print("Матрица ошибок:")
print(confusion_matrix(y_test, y_pred_final))
