# plots.py — построение графиков
import matplotlib.pyplot as plt
import numpy as np
from methods import adaptive_integration

def plot_error_vs_epsilon(methods_dict, f, a, b, exact_value, epsilons, filename):
    """
    Строит график: погрешность от требуемой точности
    methods_dict: {'Название метода': функция_метода}
    """
    plt.figure(figsize=(10, 6))
    
    for name, method in methods_dict.items():
        errors = []
        used_n = []
        for eps in epsilons:
            result, n = adaptive_integration(method, f, a, b, eps)
            error = abs(result - exact_value)
            errors.append(error)
            used_n.append(n)
        
        plt.loglog(epsilons, errors, marker='o', label=f'{name} (факт)')
        # Теоретическая оценка (порядок точности)
        if 'Симпсон' in name:
            plt.loglog(epsilons, [eps**4 for eps in epsilons], '--', alpha=0.3)
        elif 'трапеций' in name or 'прямоугольников' in name:
            plt.loglog(epsilons, [eps**2 for eps in epsilons], '--', alpha=0.3)
    
    plt.xlabel('Требуемая точность ε')
    plt.ylabel('Фактическая погрешность')
    plt.title('Зависимость погрешности от ε')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранён: {filename}")

def plot_n_vs_epsilon(methods_dict, f, a, b, epsilons, filename):
    """График: сколько вычислений нужно для заданной точности"""
    plt.figure(figsize=(10, 6))
    
    for name, method in methods_dict.items():
        n_values = []
        for eps in epsilons:
            _, n = adaptive_integration(method, f, a, b, eps)
            n_values.append(n)
        plt.loglog(epsilons, n_values, marker='s', label=name)
    
    plt.xlabel('Требуемая точность ε')
    plt.ylabel('Количество разбиений N')
    plt.title('Трудоёмкость метода')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График сохранён: {filename}")
