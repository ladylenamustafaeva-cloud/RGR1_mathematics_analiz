from comparison import run_comparison
from plots import plot_error_vs_x, plot_n_vs_epsilon
from gauss_methods import gauss_legendre, gauss_radau, gauss_lobatto, chebyshev_quadrature
import numpy as np

if __name__ == "__main__":
    print("=== Запуск сравнительного анализа ===")
    run_comparison()
    
    f = lambda x: np.exp(x)
    exact = np.e - 1
    epsilons = [1e-2, 1e-3, 1e-4, 1e-5]
    
    methods_dict = {
        'Гаусс': gauss_legendre,
        'Радо': lambda f,a,b,n: gauss_radau(f,a,b,n),
        'Лобатто': gauss_lobatto,
        'Чебышёв': chebyshev_quadrature
    }
    
    plot_n_vs_epsilon(methods_dict, f, 0, 1, exact, epsilons)
    print("Графики сохранены: efficiency_comparison.png, error_comparison.png")
