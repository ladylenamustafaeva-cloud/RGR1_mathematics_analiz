# methods.py — методы численного интегрирования
import numpy as np

def rectangles(f, a, b, n):
    """Метод прямоугольников (средние точки)"""
    h = (b - a) / n
    x = np.linspace(a + h/2, b - h/2, n)  # средние точки отрезков
    return h * np.sum(f(x))

def trapezoid(f, a, b, n):
    """Метод трапеций"""
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h * (0.5*y[0] + np.sum(y[1:-1]) + 0.5*y[-1])

def simpson(f, a, b, n):
    """Метод Симпсона (n должно быть чётным!)"""
    if n % 2 == 1:
        n += 1  # делаем чётным
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    return h/3 * (y[0] + 4*np.sum(y[1:-1:2]) + 2*np.sum(y[2:-2:2]) + y[-1])

def three_eighths(f, a, b, n):
    """Метод трёх восьмых (n кратно 3)"""
    while n % 3 != 0:
        n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    result = y[0] + y[-1]
    for i in range(1, n):
        if i % 3 == 0:
            result += 2 * y[i]
        else:
            result += 3 * y[i]
    return 3*h/8 * result

def adaptive_integration(method, f, a, b, eps, max_iter=10000):
    """Адаптивный подбор n для достижения точности eps"""
    n = 4
    prev_result = method(f, a, b, n)
    
    for _ in range(max_iter):
        n *= 2
        curr_result = method(f, a, b, n)
        if abs(curr_result - prev_result) < eps:
            return curr_result, n
        prev_result = curr_result
    
    return curr_result, n  # вернули то, что получилось
