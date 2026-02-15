"""
Пример использования GDRegressor
"""
import pandas as pd
import matplotlib.pyplot as plt
from linreg import GDRegressor

# Загрузка данных
print("Загрузка данных Boston Housing...")
data = pd.read_csv("housing.csv")
X = data[["RM"]]  # Используем только один признак для простоты
y = data["MEDV"]

# Масштабирование признаков
print("Масштабирование признаков...")
X_scaled = X.apply(GDRegressor.z_scaler)

# Поиск оптимальных параметров
print("Поиск оптимальных гиперпараметров...")
n_iter, alpha = GDRegressor.find_optimal_params(X_scaled, y)
print(f"Оптимальные параметры: alpha={alpha}, n_iter={n_iter}")

# Создание и обучение модели
model = GDRegressor(alpha=alpha, n_iter=n_iter, progress=True)
model.fit(X_scaled, y)

# Предсказание
y_pred = model.predict(X_scaled)

# Метрики
rmse = GDRegressor.rmse(y, y_pred)
r2 = GDRegressor.r_squared(y, y_pred)
print(f"\nРезультаты:")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"Intercept: {model.intercept_:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")

# Визуализация
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_scaled, y, alpha=0.5, label="Данные")
plt.plot(X_scaled, y_pred, color='red', label="Модель")
plt.xlabel("Среднее количество комнат (масштабировано)")
plt.ylabel("Медианная стоимость")
plt.title("Линейная регрессия: RM -> MEDV")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
GDRegressor.plot_loss_function(model)

plt.tight_layout()
plt.show()
