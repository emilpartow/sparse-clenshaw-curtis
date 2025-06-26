# Sparse Clenshaw–Curtis Quadrature

Efficient, flexible Python implementation of sparse grid integration for high-dimensional problems using Clenshaw–Curtis quadrature nodes.  
This project demonstrates the construction, application, and convergence of Smolyak-type sparse grids on cubes $[a,b]^d$.

---

## Features

- (Smolyak) Sparse grid quadrature in arbitrary dimensions
- Clenshaw–Curtis nodes and weights with closed nonlinear growth rule
- Easy integration of arbitrary functions over $[a,b]^d$

---

## Mathematical Background

Given $a < b$, a weight function $\omega \colon [a,b] \to (0,\infty)$, and an integrand $f \colon [a,b] \to \mathbb{R}$, a univariate $m$-point \emph{quadrature rule} is a set of $m$ nodes and weights $(w_j, \xi_j)_{j=1}^m \subset \mathbb{R} \times [a,b]$ used to approximate the weighted integral
\[
\int_a^b f(x) \omega(x)\, dx
\]
by the discrete sum $\sum_{j=1}^m w_j f(\xi_j).$

Sparse grid quadrature provides efficient numerical integration for high-dimensional functions, drastically reducing the number of required points compared to full tensor grids.  
The approach is based on the **Smolyak algorithm** and uses **Clenshaw–Curtis** quadrature rules as the one-dimensional building block.

See:  
- Smolyak, S. A. (1963), *Dokl. Akad. Nauk SSSR*  
- Bungartz & Griebel (2004), "Sparse Grids", *Acta Numerica*.

---

## Installation

Clone the repository:
```sh
git clone https://github.com/<your-username>/sparse-clenshaw-curtis.git
cd sparse-clenshaw-curtis
```

---

## Install dependencies (preferably in a virtual environment):

```sh
pip install numpy matplotlib modepy
```

---
## Usage Example
Integrate a function over $[0,1]^2$ using the sparse grid quadrature:
```python
from sparse_clenshaw_curtis import sparse_grid_nodes_weights

def f(x):
    return x[0] * x[1]

dim = 2
level = 5

nodes, weights = sparse_grid_nodes_weights(dim, level, a = 0, b = 1)
integral_approx = (weights * [f(xi) for xi in nodes]).sum()
print(f"Approximate integral: {integral_approx:.8f}")
```
