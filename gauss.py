"""
Método de Eliminação de Gauss para sistemas lineares Ax = b.

Duas implementações:

* ``eliminacao_gauss``: versão didática em Python puro (listas). Funciona com
  ``float`` e com ``fractions.Fraction`` (aritmética exata), e pode registrar
  cada operação elementar feita nas linhas, igual fazemos no caderno.
* ``eliminacao_gauss_numpy``: mesma ideia, mas vetorizada com NumPy, usada
  para medir desempenho em sistemas grandes.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

Matriz = List[List]
Vetor = List


class SistemaSingularError(ValueError):
    """O sistema não tem solução única (matriz singular)."""


def _fmt(valor) -> str:
    """Formata número para exibição (fração ou decimal)."""
    if isinstance(valor, Fraction):
        return str(valor) if valor.denominator != 1 else str(valor.numerator)
    if abs(valor - round(valor)) < 1e-12:
        return str(int(round(valor)))
    return f"{valor:.6g}"


def formatar_matriz_aumentada(A: Matriz, b: Vetor) -> str:
    """Retorna a matriz aumentada [A | b] como texto alinhado."""
    linhas = [[_fmt(v) for v in A[i]] + ["|", _fmt(b[i])] for i in range(len(A))]
    larguras = [max(len(l[j]) for l in linhas) for j in range(len(linhas[0]))]
    return "\n".join(
        "[ " + "  ".join(c.rjust(w) for c, w in zip(l, larguras)) + " ]" for l in linhas
    )


def retrossubstituicao(U: Matriz, c: Vetor) -> Vetor:
    """
    Resolve Ux = c com U triangular superior, de baixo para cima:

        x_n = c_n / u_nn
        x_i = (c_i - sum_{j>i} u_ij * x_j) / u_ii
    """
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        soma = sum(U[i][j] * x[j] for j in range(i + 1, n))
        if U[i][i] == 0:
            raise SistemaSingularError("Pivô nulo na retrossubstituição.")
        x[i] = (c[i] - soma) / U[i][i]
    return x


def eliminacao_gauss(
    A: Sequence[Sequence],
    b: Sequence,
    pivoteamento: str = "parcial",
    registrar: bool = False,
    tolerancia: float = 0.0,
) -> Tuple[Vetor, List[str]]:
    """
    Resolve Ax = b pelo Método de Eliminação de Gauss.

    Parâmetros
    ----------
    A, b : coeficientes e termos independentes (listas, tuplas ou arrays).
    pivoteamento : "parcial" (troca pela linha de maior |pivô|, padrão) ou
        "nenhum" (só troca linhas se o pivô for exatamente zero, como na
        resolução manual).
    registrar : se True, devolve a lista de passos com as operações elementares
        e a matriz aumentada após cada etapa.
    tolerancia : pivôs com módulo menor ou igual a isso são tratados como zero.
        O padrão (0) só considera zero o zero exato, como na resolução manual.

    Retorna
    -------
    (x, passos)
    """
    if pivoteamento not in ("parcial", "nenhum"):
        raise ValueError("pivoteamento deve ser 'parcial' ou 'nenhum'")

    # Cópias para não alterar a entrada do usuário
    M = [list(linha) for linha in A]
    v = list(b)
    n = len(M)
    if any(len(linha) != n for linha in M) or len(v) != n:
        raise ValueError("A deve ser quadrada (n x n) e b deve ter n elementos.")

    def eh_zero(t) -> bool:
        return abs(t) <= tolerancia

    passos: List[str] = []
    if registrar:
        passos.append("Matriz aumentada inicial:\n" + formatar_matriz_aumentada(M, v))

    for k in range(n - 1):
        # 1) Escolha do pivô
        if pivoteamento == "parcial":
            p = max(range(k, n), key=lambda i: abs(M[i][k]))
        else:
            p = next((i for i in range(k, n) if not eh_zero(M[i][k])), k)

        if eh_zero(M[p][k]):
            raise SistemaSingularError(
                f"Coluna {k + 1} sem pivô não nulo: o sistema não tem solução única."
            )

        if p != k:
            M[k], M[p] = M[p], M[k]
            v[k], v[p] = v[p], v[k]
            if registrar:
                passos.append(f"L{k + 1} <-> L{p + 1}")

        # 2) Zerar os elementos abaixo do pivô: L_i <- L_i - m_ik * L_k
        ops = []
        for i in range(k + 1, n):
            m = M[i][k] / M[k][k]
            if m == 0:
                continue
            for j in range(k, n):
                M[i][j] = M[i][j] - m * M[k][j]
            M[i][k] = 0 * m  # zero exato (mantém o tipo: Fraction ou float)
            v[i] = v[i] - m * v[k]
            ops.append(f"L{i + 1} <- L{i + 1} - ({_fmt(m)})*L{k + 1}")

        if registrar and ops:
            passos.append(
                f"Passo {k + 1}: zerando abaixo do pivô (coluna {k + 1})\n"
                + "\n".join(ops)
                + "\n"
                + formatar_matriz_aumentada(M, v)
            )

    if eh_zero(M[n - 1][n - 1]):
        raise SistemaSingularError("Último pivô nulo: o sistema não tem solução única.")

    x = retrossubstituicao(M, v)
    if registrar:
        passos.append(
            "Retrossubstituição:\n"
            + "\n".join(f"x{i + 1} = {_fmt(xi)}" for i, xi in enumerate(x))
        )
    return x, passos


def eliminacao_gauss_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Eliminação de Gauss com pivoteamento parcial, vetorizada com NumPy."""
    M = np.array(A, dtype=float)
    v = np.array(b, dtype=float)
    n = M.shape[0]
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(M[k:, k])))
        if M[p, k] == 0:
            raise SistemaSingularError("Matriz singular.")
        if p != k:
            M[[k, p]] = M[[p, k]]
            v[[k, p]] = v[[p, k]]
        m = M[k + 1 :, k] / M[k, k]
        M[k + 1 :, k:] -= np.outer(m, M[k, k:])
        v[k + 1 :] -= m * v[k]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (v[i] - M[i, i + 1 :] @ x[i + 1 :]) / M[i, i]
    return x


def residuo(A, b, x) -> float:
    """Norma infinito do resíduo ||b - Ax||, mede o quanto x satisfaz o sistema."""
    A = np.array(A, dtype=float)
    return float(np.max(np.abs(np.array(b, dtype=float) - A @ np.array(x, dtype=float))))


def para_fracoes(A, b) -> Tuple[Matriz, Vetor]:
    """Converte A e b para Fraction (resolução exata, igual à feita à mão)."""
    return [[Fraction(e) for e in linha] for linha in A], [Fraction(e) for e in b]


def resolver_e_mostrar(
    A, b, nomes: Optional[Sequence[str]] = None, pivoteamento: str = "nenhum"
) -> Vetor:
    """Resolve com frações, imprimindo o passo a passo."""
    Af, bf = para_fracoes(A, b)
    x, passos = eliminacao_gauss(Af, bf, pivoteamento=pivoteamento, registrar=True)
    for p in passos:
        print(p, end="\n\n")
    nomes = nomes or [f"x{i + 1}" for i in range(len(x))]
    print("Solução: " + ", ".join(f"{n} = {_fmt(v)}" for n, v in zip(nomes, x)))
    return x
