import os
import sys
import unittest
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gauss import (  # noqa: E402
    SistemaSingularError,
    eliminacao_gauss,
    eliminacao_gauss_numpy,
    para_fracoes,
)
from iterativos import gauss_seidel, jacobi  # noqa: E402


class TestExemplosDaAula(unittest.TestCase):
    def resolver_exato(self, A, b):
        return eliminacao_gauss(*para_fracoes(A, b), pivoteamento="nenhum")[0]

    def test_exemplo_3x3(self):
        x = self.resolver_exato([[2, 3, 3], [1, -7, 2], [5, 8, -1]], [13, 17, -2])
        self.assertEqual(x, [2, -1, 4])

    def test_exemplo_servidores(self):
        x = self.resolver_exato([[1, 2, 1], [2, 1, 3], [3, 2, 2]], [9, 13, 14])
        self.assertEqual(x, [Fraction(8, 7), Fraction(18, 7), Fraction(19, 7)])

    def test_exemplo_4x4(self):
        A = [[3, 1, 1, 1], [2, -3, -3, -1], [1, -1, 4, 5], [2, 2, -1, -10]]
        x = self.resolver_exato(A, [6, -2, -11, 37])
        self.assertEqual(x, [2, 2, 1, -3])

    def test_para_praticar(self):
        A = [[1, 1, 1], [2, 3, 1], [1, 4, 3]]
        x = self.resolver_exato(A, [6, 14, 20])
        self.assertEqual([sum(a * xi for a, xi in zip(l, x)) for l in A], [6, 14, 20])


class TestPropriedades(unittest.TestCase):
    def test_compara_com_numpy(self):
        rng = np.random.default_rng(0)
        for n in (2, 5, 20, 60):
            A = rng.random((n, n)) + np.eye(n)
            b = rng.random(n)
            esperado = np.linalg.solve(A, b)
            np.testing.assert_allclose(eliminacao_gauss(A.tolist(), b.tolist())[0], esperado, rtol=1e-9)
            np.testing.assert_allclose(eliminacao_gauss_numpy(A, b), esperado, rtol=1e-9)

    def test_pivo_zero_exige_troca(self):
        x, _ = eliminacao_gauss([[0, 1], [1, 1]], [1, 2], pivoteamento="nenhum")
        self.assertEqual(x, [1, 1])

    def test_pivoteamento_parcial_corrige_pivo_pequeno(self):
        A, b = [[1e-20, 1.0], [1.0, 1.0]], [1.0, 2.0]
        x, _ = eliminacao_gauss(A, b, pivoteamento="parcial")
        np.testing.assert_allclose(x, [1, 1])

    def test_singular(self):
        with self.assertRaises(SistemaSingularError):
            eliminacao_gauss([[1, 2], [2, 4]], [3, 6])

    def test_nao_altera_entrada(self):
        A, b = [[0, 1], [1, 1]], [1, 2]
        eliminacao_gauss(A, b)
        self.assertEqual((A, b), ([[0, 1], [1, 1]], [1, 2]))

    def test_iterativos_convergem(self):
        A = np.array([[10, 1, 1], [2, 10, 1], [2, 2, 10]], dtype=float)
        b = np.array([12, 13, 14], dtype=float)
        for metodo in (jacobi, gauss_seidel):
            x, _ = metodo(A, b)
            np.testing.assert_allclose(x, [1, 1, 1], atol=1e-8)


if __name__ == "__main__":
    unittest.main()
