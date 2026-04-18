import matplotlib.pyplot as plt
import numpy as np

def plot_error_vs_x(methods, f, a, b, exact, filename='error_comparison.png'):
    x = np.linspace(a, b, 100)
    plt.figure(figsize=(10, 6))
    for name, (func, n) in methods.items():
        # Накопленная ошибка
        err = np.abs(func(f, a, x, n) - (exact * (x - a) / (b - a)))
        plt.semilogy(x, err + 1e-12, label=f'{name} (n={n})')
    plt.xlabel('x')
    plt.ylabel('Ошибка накопленного интеграла')
    plt.title('Сравнение ошибки (лог. шкала)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_n_vs_epsilon(methods_dict, f, a, b, exact, epsilons, filename='efficiency_comparison.png'):
    plt.figure(figsize=(10, 6))
    for name, method in methods_dict.items():
        ns = []
        for eps in epsilons:
            _, n = adaptive_gauss(method, f, a, b, eps)
            ns.append(n)
        plt.loglog(epsilons, ns, 'o-', label=name, markersize=4)
    plt.xlabel('Требуемая точность ε')
    plt.ylabel('Необходимое число узлов n')
    plt.title('Эффективность методов (лог. шкала)')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
