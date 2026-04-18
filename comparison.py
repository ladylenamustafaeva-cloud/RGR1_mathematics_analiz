import numpy as np
from gauss_methods import gauss_legendre, gauss_radau, gauss_lobatto, chebyshev_quadrature
from adaptive import adaptive_gauss

def newton_cotes_trapezoid(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))

def newton_cotes_simpson(f, a, b, n):
    if n % 2 != 0: n += 1
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h/3 * (f(x[0]) + 4*np.sum(f(x[1:-1:2])) + 2*np.sum(f(x[2:-1:2])) + f(x[-1]))

def run_comparison():
    f1 = lambda x: np.exp(x)
    exact1 = np.e - 1
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    
    methods = {
        'Трапеции': newton_cotes_trapezoid,
        'Симпсон': newton_cotes_simpson,
        'Гаусс': gauss_legendre,
        'Радо': lambda f,a,b,n: gauss_radau(f,a,b,n),
        'Лобатто': gauss_lobatto,
        'Чебышёв': chebyshev_quadrature
    }
    
    print("Число узлов для f(x)=e^x:")
    for name, method in methods.items():
        ns = [adaptive_gauss(method, f1, 0, 1, eps)[1] for eps in epsilons]
        print(f"{name:10} | {ns}")
