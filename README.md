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
sh```

## Install dependencies (preferably in a virtual environment):

```sh
pip install numpy matplotlib modepy
