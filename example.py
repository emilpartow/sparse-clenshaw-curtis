"""
Sparse Grid Quadrature: Convergence Study Example (d=2)
=======================================================

This example demonstrates the accuracy of sparse grid quadrature using Clenshaw–Curtis nodes 
for integrating different test functions on the unit square [0, 1]^2. 
Error decay is shown as a function of the sparse grid level.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# --- Import your previously defined nodes/weights construction ---
from sparse_clenshaw_curtis import sparse_grid_nodes_weights

# --- Define Integration Operator ---
def sparse_grid_integrate(func, dim, level):
    """
    Integrate `func` over [0,1]^dim using sparse grid quadrature of given level.
    """
    nodes, weights = sparse_grid_nodes_weights(dim, level)
    # Vectorized evaluation: func should accept array of shape (N, dim)
    vals = np.apply_along_axis(func, 1, nodes)
    return np.dot(vals, weights)

# --- Define test functions and exact integrals for d=2 ---
test_functions = [
    {
        "func": lambda x: x[0]**x[1],
        "exact": math.log(2),      # \int_0^1 x dx = 0.5, so 0.5*0.5 = 0.25
        "label": r"$f(x)=x_1**x_2$"
    },
    {
        "func": lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]),
        "exact": 4 / (np.pi**2),
        "label": r"$f(x)=\sin(\pi x_1)\sin(\pi x_2)$"
    },
    {
        "func": lambda x: np.exp(x[0] + x[1]),
        "exact": (np.e - 1)**2,
        "label": r"$f(x)=\exp(x_1 + x_2)$"
    },
]

levels = range(1, 7)
dim = 2

# --- Plotting ---
fig, axes = plt.subplots(1, len(test_functions), figsize=(5 * len(test_functions), 4), sharey=True)

if len(test_functions) == 1:
    axes = [axes]  # Ensure axes is always iterable

tol = 1e-14  # All errors below this are considered zero

for ax, tf in zip(axes, test_functions):
    errors = []
    for l in levels:
        approx = sparse_grid_integrate(tf["func"], dim=dim, level=l)
        errors.append(np.abs(approx - tf["exact"]))
    errors = np.array(errors)
    # Set errors below tol to zero (for clearer plot & numerical stability)
    errors = np.where(errors < tol, 0, errors)
    ax.semilogy(levels, errors, marker='o', label=tf["label"])
    ax.set_xlabel("Sparse Grid Level")
    ax.set_title(f"Test Function: {tf['label']}")
    ax.grid(True, which="both", ls="--")
    ax.legend()
axes[0].set_ylabel("Absolute Error (log scale)")
fig.suptitle(
    "Sparse Grid Quadrature: Convergence for Test Functions ($d=2$)\n"
    f"(Errors below $10^{{-10}}$ set to zero)",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
