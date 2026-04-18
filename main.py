import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as leg
import os
import time

# Директория для сохранения графиков
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def transform_interval(nodes, weights, a, b):
    """Преобразование узлов и весов с [-1, 1] на [a, b]"""
    t = (b - a) / 2 * nodes + (b + a) / 2
    w = (b - a) / 2 * weights
    return t, w

def gauss_legendre(n, a, b):
    """Квадратура Гаусса-Лежандра (n узлов). Точность 2n-1."""
    nodes, weights = leg.leggauss(n)
    return transform_interval(nodes, weights, a, b)

def gauss_lobatto(n, a, b):
    """Квадратура Гаусса-Лобатто (n узлов, включая концы). Точность 2n-3."""
    if n < 2: 
        raise ValueError("Для Лобатто нужно минимум 2 узла")
    
    if n == 2:
        nodes = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
        return transform_interval(nodes, weights, a, b)
    
    P_n_minus_1 = leg.Legendre.basis(n-1)
    P_deriv = leg.legder(P_n_minus_1.coef)
    
    if len(P_deriv) <= 1 or np.all(P_deriv == 0):
        inner_nodes = np.array([])
    else:
        inner_nodes = leg.Legendre(P_deriv).roots()
    
    nodes = np.sort(np.concatenate(([-1.0], inner_nodes, [1.0])))
    
    weights = np.zeros(n)
    weights[0] = 2.0 / (n * (n - 1))
    weights[-1] = 2.0 / (n * (n - 1))
    
    for i in range(1, n-1):
        val = P_n_minus_1(nodes[i])
        weights[i] = 2.0 / (n * (n - 1) * val**2)
        
    return transform_interval(nodes, weights, a, b)

def gauss_radau(n, a, b, endpoint='left'):
    """Квадратура Гаусса-Радо (n узлов, один фиксированный конец). Точность 2n-2."""
    if n < 2: 
        raise ValueError("Для Радо нужно минимум 2 узла")
    
    P_n = leg.Legendre.basis(n)
    P_n_minus_1 = leg.Legendre.basis(n-1)
    
    if endpoint == 'left':
        sum_coef = np.zeros(n+1)
        sum_coef[:n] += P_n_minus_1.coef
        sum_coef += P_n.coef
        radau_poly = leg.Legendre(sum_coef)
        inner_nodes = radau_poly.roots()
        nodes = np.sort(np.concatenate(([-1.0], inner_nodes)))
    else:
        sum_coef = np.zeros(n+1)
        sum_coef[:n] -= P_n_minus_1.coef
        sum_coef += P_n.coef
        radau_poly = leg.Legendre(sum_coef)
        inner_nodes = radau_poly.roots()
        nodes = np.sort(np.concatenate((inner_nodes, [1.0])))

    weights = np.zeros(n)
    for i, x in enumerate(nodes):
        val = P_n_minus_1(x)
        if abs(val) < 1e-15:  # <-- Защита от деления на ноль
            weights[i] = 0.0
        else:
            if endpoint == 'left':
                weights[i] = (1 - x) / (n**2 * val**2)
            else:
                weights[i] = (1 + x) / (n**2 * val**2)
            
    return transform_interval(nodes, weights, a, b)

def integrate_custom(nodes, weights, f):
    return np.sum(weights * f(nodes))

def adaptive_integration_fast(method_func, f, a, b, eps, max_n=50):
    """Быстрый адаптивный подбор числа узлов."""
    prev_approx = None
    approx = None  # <-- Инициализируем переменную
    
    for n in range(2, max_n + 1):
        try:
            nodes, weights = method_func(n, a, b)
            if len(nodes) != len(weights):
                continue
            approx = integrate_custom(nodes, weights, f)
            
            if prev_approx is not None:
                if abs(approx - prev_approx) < eps:
                    return approx, n
            prev_approx = approx
        except Exception as e:
            continue
    
    # Если не достигли точности, возвращаем последнее успешное значение
    if approx is not None:
        return approx, max_n
    else:
        return 0.0, max_n

# Тестовые функции
funcs = {
    'exp(x)': (lambda x: np.exp(x), 0, 1, np.e - 1),
    '|x-0.5|': (lambda x: np.abs(x - 0.5), 0, 1, 0.25)
}

# Методы
methods = {
    'Гаусс': gauss_legendre,
    'Лобатто': gauss_lobatto,
    'Радо (левый)': lambda n, a, b: gauss_radau(n, a, b, 'left'),
}

# Параметры исследования
epsilons = [1e-2, 1e-3, 1e-4, 1e-5]

# Сбор данных
results = {name: {fname: {'N': [], 'error': []} for fname in funcs} for name in methods}

print("Вычисление интегралов...")
start_time = time.time()

for fname, (f, a, b, exact) in funcs.items():
    print(f"\nФункция: {fname}")
    for mname, method_func in methods.items():
        N_list, error_list = [], []
        for eps in epsilons:
            approx, N = adaptive_integration_fast(method_func, f, a, b, eps)
            error = abs(approx - exact)
            N_list.append(N)
            error_list.append(error)
            print(f"  {mname}: ε={eps:.0e} → N={N}, погрешность={error:.2e}")
        results[mname][fname]['N'] = N_list
        results[mname][fname]['error'] = error_list

end_time = time.time()
print(f"\n Время вычислений: {end_time - start_time:.2f} сек.")

#ГРАФИК 1: Гладкая функция
plt.figure(figsize=(9, 5))
for mname in methods:
    plt.semilogy(epsilons, results[mname]['exp(x)']['N'], 
                 'o-', label=mname, linewidth=2, markersize=7)

plt.semilogy(epsilons, [16, 48, 300, 400], 's--', label='Трапеции (РГР1)', linewidth=2)
plt.semilogy(epsilons, [8, 8, 8, 8], '^-', label='Симпсон (РГР1)', linewidth=2)

plt.xlabel('Требуемая точность ε')
plt.ylabel('Число узлов N')
plt.title('Влияние точности на объём вычислений (f(x)=exp(x))')
plt.legend(fontsize=9)
plt.grid(True, which="both", ls="-", alpha=0.7)
plt.gca().invert_xaxis()
plt.savefig(os.path.join(SCRIPT_DIR, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("График 1 сохранён: efficiency_comparison.png")

#ГРАФИК 2: Негладкая функция
plt.figure(figsize=(9, 5))
for mname in methods:
    plt.semilogy(epsilons, results[mname]['|x-0.5|']['N'], 
                 's-', label=mname, linewidth=2, markersize=7)

plt.xlabel('Требуемая точность ε')
plt.ylabel('Число узлов N')
plt.title('Влияние точности на объём вычислений (f(x)=|x-0.5|)')
plt.legend(fontsize=9)
plt.grid(True, which="both", ls="-", alpha=0.7)
plt.gca().invert_xaxis()
plt.savefig(os.path.join(SCRIPT_DIR, 'nonsmooth_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("График 2 сохранён: nonsmooth_comparison.png")

print("\n Все графики успешно сохранены!")
