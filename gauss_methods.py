import numpy as np
from numpy.polynomial import legendre as L
from scipy.special import roots_sh_legendre

def gauss_legendre(f, a, b, n):
    nodes, weights = L.leggauss(n)
    x = ((b - a) * nodes + (b + a)) / 2
    w = (b - a) * weights / 2
    return np.sum(w * f(x))

def gauss_radau(f, a, b, n, endpoint='left'):
    nodes, weights = roots_sh_legendre(n)
    if endpoint == 'right':
        nodes = -nodes
    x = ((b - a) * nodes + (b + a)) / 2
    w = (b - a) * weights / 2
    return np.sum(w * f(x))

def gauss_lobatto(f, a, b, n):
    if n == 2:
        nodes = np.array([-1, 1])
        weights = np.array([1, 1])
    else:
        inner_nodes, _ = L.leggauss(n - 2)
        nodes = np.concatenate([[-1], inner_nodes, [1]])
        weights = np.zeros(n)
        weights[0] = weights[-1] = 2 / (n * (n - 1))
        for i in range(1, n - 1):
            P_nm1 = L.Legendre.basis(n - 1)(nodes[i])
            weights[i] = 2 / (n * (n - 1) * P_nm1**2)
    x = ((b - a) * nodes + (b + a)) / 2
    w = (b - a) * weights / 2
    return np.sum(w * f(x))

def chebyshev_quadrature(f, a, b, n):
    if n >= 8:
        print("Warning: для n >= 8 формула Чебышёва может быть неточной")
    k = np.arange(1, n + 1)
    nodes = np.cos((2 * k - 1) * np.pi / (2 * n))
    weights = np.full(n, 2 / n)
    x = ((b - a) * nodes + (b + a)) / 2
    w = (b - a) * weights / 2
    return np.sum(w * f(x))
