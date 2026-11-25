"""
Métodos iterativos (Jacobi e Gauss-Seidel), usados para comparar com a
Eliminação de Gauss, que é um método direto.

Os dois só convergem com garantia se a matriz for, por exemplo,
estritamente diagonal dominante: |a_ii| > soma_{j != i} |a_ij|.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def diagonal_dominante(A) -> bool:
    A = np.abs(np.array(A, dtype=float))
    diag = np.diag(A)
    return bool(np.all(diag > A.sum(axis=1) - diag))


def jacobi(A, b, x0=None, tol=1e-10, max_iter=500) -> Tuple[np.ndarray, List[float]]:
    """x_i^(k+1) = (b_i - sum_{j != i} a_ij x_j^(k)) / a_ii"""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    D = np.diag(A)
    R = A - np.diagflat(D)
    historico = []
    for _ in range(max_iter):
        x_novo = (b - R @ x) / D
        erro = float(np.max(np.abs(x_novo - x)))
        historico.append(erro)
        x = x_novo
        if erro < tol:
            break
    return x, historico


def gauss_seidel(A, b, x0=None, tol=1e-10, max_iter=500) -> Tuple[np.ndarray, List[float]]:
    """Igual ao Jacobi, mas já usa os valores novos assim que são calculados."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float)
    historico = []
    for _ in range(max_iter):
        x_ant = x.copy()
        for i in range(n):
            soma = A[i, :i] @ x[:i] + A[i, i + 1 :] @ x_ant[i + 1 :]
            x[i] = (b[i] - soma) / A[i, i]
        erro = float(np.max(np.abs(x - x_ant)))
        historico.append(erro)
        if erro < tol:
            break
    return x, historico
