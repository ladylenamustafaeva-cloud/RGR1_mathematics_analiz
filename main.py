# main.py — точка входа
import numpy as np
from methods import rectangles, trapezoid, simpson, three_eighths, adaptive_integration
from plots import plot_error_vs_epsilon, plot_n_vs_epsilon

# ===== 1. Задаём функцию и точное значение интеграла =====
# Пример 1: гладкая функция
def f_smooth(x):
    return np.exp(x)  # ∫₀¹ eˣ dx = e - 1 ≈ 1.71828

a, b = 0, 1
exact_smooth = np.e - 1

# Пример 2: функция с изломом (менее гладкая)
def f_rough(x):
    return np.abs(x - 0.5)  # ∫₀¹ |x-0.5| dx = 0.25

exact_rough = 0.25

# ===== 2. Словарь методов =====
methods = {
    'Прямоугольники': rectangles,
    'Трапеции': trapezoid,
    'Симпсон': simpson,
    '3/8': three_eighths
}

# ===== 3. Набор точностей для исследования =====
epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

# ===== 4. Исследование на гладкой функции =====
print("🔍 Исследование: гладкая функция f(x)=eˣ")
plot_error_vs_epsilon(methods, f_smooth, a, b, exact_smooth, epsilons, 'graph_smooth_error.png')
plot_n_vs_epsilon(methods, f_smooth, a, b, epsilons, 'graph_smooth_n.png')

# ===== 5. Исследование на негладкой функции =====
print("🔍 Исследование: негладкая функция f(x)=|x-0.5|")
plot_error_vs_epsilon(methods, f_rough, a, b, exact_rough, epsilons, 'graph_rough_error.png')
plot_n_vs_epsilon(methods, f_rough, a, b, epsilons, 'graph_rough_n.png')

# ===== 6. Печать таблицы результатов (для отчёта) =====
print("\n📊 Таблица результатов (гладкая функция):")
print(f"{'Метод':<15} {'ε':<10} {'N':<8} {'Погрешность':<15}")
print("-" * 50)
for name, method in methods.items():
    for eps in [1e-3, 1e-5]:
        result, n = adaptive_integration(method, f_smooth, a, b, eps)
        error = abs(result - exact_smooth)
        print(f"{name:<15} {eps:<10.1e} {n:<8} {error:<15.3e}")

print("\n✅ Все графики сохранены в папке проекта!")
