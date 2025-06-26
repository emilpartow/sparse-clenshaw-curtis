"""
Sparse Grid Quadrature Construction using Clenshaw–Curtis Nodes.

This module implements the construction of Smolyak-type sparse grids for numerical integration 
over the d-dimensional unit hypercube [a, b]^d, using Clenshaw–Curtis quadrature nodes and weights.

Key features:
    - Closed nonlinear growth rule for one-dimensional quadrature.
    - Generation of multidimensional tensor product rules.
    - Efficient sparse grid construction with correct Smolyak coefficients.
    - Flexible interface: returns all nodes and associated weights for further use.

References
----------
- Smolyak, S. A. "Quadrature and interpolation formulas for tensor products of certain classes of functions." Dokl. Akad. Nauk SSSR, 1963.
- Bungartz, H.-J., & Griebel, M. "Sparse grids." Acta Numerica, 13, 2004.

Dependencies
------------
- numpy
- modepy (for Clenshaw–Curtis nodes and weights)
"""

from itertools import product
import numpy as np
from math import comb
from modepy.quadrature.clenshaw_curtis import _make_clenshaw_curtis_nodes_and_weights

def index_set(dim: int, level: int):
    """
    Construct the multi-index set for the sparse grid according to the Smolyak algorithm.

    For given dimension `dim` and total level `level`, returns all integer multi-indices
    (i_1, ..., i_dim) satisfying the Smolyak selection rule.

    Parameters
    ----------
    dim : int
        Number of spatial dimensions.
    level : int
        Total (maximum) level of the sparse grid.

    Returns
    -------
    list of tuple of int
        List of multi-indices for the tensor product rules to be combined.
    """
    return [idx for idx in product(range(1, level + dim + 1), repeat=dim)
            if level + 1 <= sum(idx) <= level + dim]
    
def closed_non_linear_growth_rule(level: int) -> int:
    """
    Compute the number of quadrature points for a given level according to the
    non-linear closed growth rule used by Clenshaw–Curtis.

    Parameters
    ----------
    level : int
        Quadrature level (must be >= 1).

    Returns
    -------
    int
        Number of quadrature points for the given level.

    Raises
    ------
    ValueError
        If the provided level is less than 1.
    """
    if level < 1:
        raise ValueError("Level must be >= 1 for Sparse Grid.")
    if level == 1:
        return 1
    else:
        return 2**(level - 1) + 1

def clenshaw_curtis_rule(level: int):
    """
    Construct 1D Clenshaw–Curtis quadrature nodes and weights on [-1, 1].

    The number of nodes is determined by the non-linear closed growth rule:
        m = 2^{level-1} + 1

    Parameters
    ----------
    level : int
        Quadrature level (must be >= 1).

    Returns
    -------
    x : ndarray, shape (m,)
        Quadrature nodes in [-1, 1].
    w : ndarray, shape (m,)
        Corresponding quadrature weights.
    """
    m = closed_non_linear_growth_rule(level)
    x, w = _make_clenshaw_curtis_nodes_and_weights(m)
    return x, w

def tensor_product_rule(indices, one_d_rule, a, b):
    """
    Construct a tensor product quadrature rule (nodes and weights) in multiple dimensions.

    Nodes are mapped from [-1, 1]^d to the hypercube [a, b]^d.

    Parameters
    ----------
    indices : sequence of int
        Levels (one per dimension) for the 1D quadrature rules.
    one_d_rule : callable
        Function returning (nodes, weights) for a given level in 1D.
    a : int, optional
        Left endpoint of the integration domain (default is 0).
    b : int, optional
        Right endpoint of the integration domain (default is 1).

    Returns
    -------
    grid : ndarray, shape (n_points, d)
        Array of multidimensional quadrature nodes.
    weights : ndarray, shape (n_points,)
        Array of corresponding quadrature weights.

    Raises
    ------
    ValueError
        If the interval [a, b] is empty.
    TypeError
        If interval endpoints are not integers.
    """
    if not a < b:
        raise ValueError("Empty interval. Check if a < b...!")
    if not type(a) == type(b) == int:
        raise TypeError("Interval boundaries need to be integers.")
    pts_1d = [one_d_rule(l)[0] for l in indices]
    wts_1d = [one_d_rule(l)[1] for l in indices]
    grid = np.array(list(product(*pts_1d)))
    weights = np.prod(np.array(list(product(*wts_1d))), axis=1).astype(float)
    if not (a == -1 and b == 1):
        grid = (b - a) / 2 * grid + (a + b) / 2
        weights *= ((b - a) / 2) ** len(indices)
    return grid, weights

def sparse_grid_nodes_weights(dim, level, a = 0, b = 1):
    """
    Construct the nodes and weights for a sparse grid quadrature rule
    using the Smolyak algorithm and Clenshaw–Curtis points on [a, b]^d.

    The result can be used for the integration of arbitrary functions over [a, b]^d.

    Parameters
    ----------
    dim : int
        Number of spatial dimensions.
    level : int
        Total (maximum) level of the sparse grid.

    Returns
    -------
    points : ndarray, shape (n_total_points, dim)
        All quadrature nodes (not necessarily unique).
    weights : ndarray, shape (n_total_points,)
        Corresponding quadrature weights (with Smolyak coefficients applied).

    Notes
    -----
    - Some nodes may appear multiple times with different weights.
      For best efficiency, unique nodes should be merged and their weights summed.
    """
    all_points = []
    all_weights = []
    for idx in index_set(dim, level):
        coeff = (-1) ** (level + dim - sum(idx)) * comb(dim - 1, level + dim - sum(idx))
        x, w = tensor_product_rule(idx, clenshaw_curtis_rule, a, b)
        for xi, wi in zip(x, w):
            all_points.append(tuple(xi))
            all_weights.append(coeff * wi)
    return np.array(all_points), np.array(all_weights)

