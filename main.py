"""
Demonstração computacional: Método de Eliminação de Gauss.

Executar:
    python main.py

Gera a saída no terminal e os gráficos na pasta ``resultados/``.
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import numpy as np

from gauss import (
    _fmt,
    eliminacao_gauss,
    eliminacao_gauss_numpy,
    residuo,
    resolver_e_mostrar,
)
from iterativos import diagonal_dominante, gauss_seidel, jacobi

PASTA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(PASTA, exist_ok=True)


def titulo(texto: str) -> None:
    print("\n" + "=" * 72)
    print(texto)
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1) Exemplos resolvidos em aula, refeitos pelo programa com frações exatas
# ---------------------------------------------------------------------------
def exemplos_da_aula() -> None:
    titulo("1) EXEMPLOS DA AULA: resolução exata com frações")

    exemplos = [
        (
            "Exemplo 1",
            [[2, 3, 3], [1, -7, 2], [5, 8, -1]],
            [13, 17, -2],
            ["x", "y", "z"],
            "esperado: x = 2, y = -1, z = 4",
        ),
        (
            "Exemplo 2 (três servidores)",
            [[1, 2, 1], [2, 1, 3], [3, 2, 2]],
            [9, 13, 14],
            ["x", "y", "z"],
            "esperado: x = 8/7, y = 18/7, z = 19/7",
        ),
        (
            "Exemplo 3 (4 x 4)",
            [[3, 1, 1, 1], [2, -3, -3, -1], [1, -1, 4, 5], [2, 2, -1, -10]],
            [6, -2, -11, 37],
            ["x", "y", "z", "w"],
            "esperado: x = 2, y = 2, z = 1, w = -3",
        ),
        (
            "Para praticar",
            [[1, 1, 1], [2, 3, 1], [1, 4, 3]],
            [6, 14, 20],
            ["a", "b", "c"],
            "",
        ),
    ]
    for nome, A, b, var, esperado in exemplos:
        print(f"\n--- {nome} ---  {esperado}")
        resolver_e_mostrar(A, b, var)


# ---------------------------------------------------------------------------
# 2) Aplicação: balanceamento de carga entre servidores
# ---------------------------------------------------------------------------
def balanceamento_de_carga() -> None:
    titulo("2) APLICAÇÃO: balanceamento de carga em um cluster de 3 servidores")
    print(
        """Cada servidor (A, B, C) tem hardware diferente. Queremos saber quantas
requisições por segundo (x, y, z) mandar para cada um de forma que:
  * o total atenda à demanda: 1200 req/s;
  * a CPU do cluster seja totalmente aproveitada: 3400 ms de CPU por segundo
    (cada requisição gasta 2 ms em A, 3 ms em B e 4 ms em C);
  * o link de rede de 930 Mbps seja totalmente aproveitado
    (0,5 Mbps por requisição em A, 0,8 em B e 1,2 em C).

Modelo:
     x +    y +    z = 1200
    2x +   3y +   4z = 3400
  0.5x + 0.8y + 1.2z =  930
"""
    )
    A = [[1, 1, 1], [2, 3, 4], [0.5, 0.8, 1.2]]
    b = [1200, 3400, 930]
    x, passos = eliminacao_gauss(A, b, registrar=True)
    for p in passos:
        print(p, end="\n\n")
    print(f"Resíduo ||b - Ax||_inf = {residuo(A, b, x):.2e}")
    print(
        f"\nInterpretação: o balanceador deve enviar {x[0]:.0f} req/s ao servidor A, "
        f"{x[1]:.0f} req/s ao B e {x[2]:.0f} req/s ao C."
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(["Servidor A", "Servidor B", "Servidor C"], x, color=["#2b6cb0", "#2f855a", "#c05621"])
    ax.bar_label(barras, labels=[f"{v:.0f} req/s" for v in x])
    ax.set_ylabel("Requisições por segundo")
    ax.set_title("Distribuição de carga calculada por Gauss")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "balanceamento_carga.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3) Aplicação: PageRank (ranking de páginas, como em buscadores)
# ---------------------------------------------------------------------------
def pagerank() -> None:
    titulo("3) APLICAÇÃO: PageRank como sistema linear")
    paginas = ["Home", "Blog", "Loja", "Contato", "Sobre", "FAQ"]
    # links[j] = páginas para onde a página j aponta
    links = {
        0: [1, 2, 4],
        1: [0, 2],
        2: [0, 3, 5],
        3: [0],
        4: [0, 1],
        5: [2, 3],
    }
    n = len(paginas)
    d = 0.85  # fator de amortecimento (probabilidade de seguir um link)
    M = np.zeros((n, n))
    for j, destinos in links.items():
        for i in destinos:
            M[i, j] = 1 / len(destinos)

    print(
        """O PageRank de cada página é a chance de um "navegador aleatório" estar
nela. Ele satisfaz  r = d*M*r + (1-d)/n,  ou seja, o sistema linear

    (I - d*M) r = (1-d)/n * [1, 1, ..., 1]

onde M[i][j] = 1/(nº de links da página j) se j aponta para i, e d = 0,85.
"""
    )
    for j, destinos in links.items():
        print(f"  {paginas[j]:8s} -> " + ", ".join(paginas[i] for i in destinos))

    A = np.eye(n) - d * M
    b = np.full(n, (1 - d) / n)
    r, _ = eliminacao_gauss(A.tolist(), b.tolist())
    r = np.array(r)

    # Comparação com o método da potência (iterativo, usado pelo Google)
    r_pot = np.full(n, 1 / n)
    iteracoes = 0
    while True:
        novo = d * M @ r_pot + (1 - d) / n
        iteracoes += 1
        if np.max(np.abs(novo - r_pot)) < 1e-12:
            r_pot = novo
            break
        r_pot = novo

    print("\nPágina     PageRank (Gauss)   PageRank (potência)")
    for i in np.argsort(-r):
        print(f"  {paginas[i]:8s}   {r[i]:.6f}           {r_pot[i]:.6f}")
    print(f"Soma dos PageRanks = {r.sum():.6f} (deve ser 1)")
    print(
        f"Diferença máxima entre os dois métodos: {np.max(np.abs(r - r_pot)):.2e} "
        f"(método da potência levou {iteracoes} iterações)"
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ordem = np.argsort(-r)
    barras = ax.bar([paginas[i] for i in ordem], r[ordem], color="#2b6cb0")
    ax.bar_label(barras, labels=[f"{v:.3f}" for v in r[ordem]])
    ax.set_ylabel("PageRank")
    ax.set_title("PageRank das páginas (solução por Eliminação de Gauss)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "pagerank.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4) Análise de precisão
# ---------------------------------------------------------------------------
def analise_precisao() -> None:
    titulo("4) ANÁLISE DE PRECISÃO")

    # 4a) Por que pivotear? Pivô muito pequeno amplifica o erro de arredondamento.
    eps = 1e-20
    A = [[eps, 1.0], [1.0, 1.0]]
    b = [1.0, 2.0]
    x_sem, _ = eliminacao_gauss(A, b, pivoteamento="nenhum")
    x_com, _ = eliminacao_gauss(A, b, pivoteamento="parcial")
    print(
        f"""4a) Sistema  {eps:g}*x1 + x2 = 1  e  x1 + x2 = 2
    Solução real: x1 ~ 1, x2 ~ 1
    Sem pivoteamento:       x1 = {x_sem[0]:.6f}, x2 = {x_sem[1]:.6f}   (ERRADO)
    Com pivoteamento parcial: x1 = {x_com[0]:.6f}, x2 = {x_com[1]:.6f}   (correto)
    Motivo: dividir por um pivô ~0 gera o multiplicador 1e20, e o "1" da
    segunda equação some no arredondamento do float (precisão ~1e-16)."""
    )

    # 4b) Mal condicionamento: matriz de Hilbert H_ij = 1/(i+j-1)
    print(
        "\n4b) Matriz de Hilbert (mal condicionada). Solução exata: x = [1, 1, ..., 1]"
    )
    print("    n    número de condição    erro máximo em x     resíduo")
    ns, conds, erros, residuos = [], [], [], []
    for n in range(2, 14):
        H = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
        x_exato = np.ones(n)
        b = H @ x_exato
        x, _ = eliminacao_gauss(H.tolist(), b.tolist())
        erro = float(np.max(np.abs(np.array(x) - x_exato)))
        res = residuo(H, b, x)
        cond = float(np.linalg.cond(H))
        ns.append(n)
        conds.append(cond)
        erros.append(erro)
        residuos.append(res)
        print(f"   {n:2d}    {cond:16.3e}    {erro:16.3e}   {res:10.2e}")
    print(
        """    Conclusão: o resíduo continua pequeno (o método é estável), mas o erro
    em x cresce junto com o número de condição. Regra prática: perdemos
    cerca de log10(cond) dígitos dos ~16 que o float oferece."""
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ns, conds, "o-", label="Número de condição cond(H)")
    ax.semilogy(ns, erros, "s-", label="Erro máximo em x")
    # resíduo exatamente 0 não aparece em escala log: desenhamos como 1e-17
    ax.semilogy(ns, [max(r, 1e-17) for r in residuos], "^-", label="Resíduo ||b - Hx|| (0 mostrado como 1e-17)")
    ax.set_xlabel("Tamanho n da matriz de Hilbert")
    ax.set_title("Precisão da Eliminação de Gauss em sistemas mal condicionados")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "precisao_hilbert.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5) Comparação de desempenho: custo O(n^3)
# ---------------------------------------------------------------------------
def _cronometrar(f, repeticoes=3) -> float:
    melhor = float("inf")
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        f()
        melhor = min(melhor, time.perf_counter() - t0)
    return melhor


def desempenho() -> None:
    titulo("5) DESEMPENHO: tempo x tamanho do sistema")
    rng = np.random.default_rng(42)
    tamanhos = [10, 25, 50, 100, 200, 400, 800]
    t_puro, t_np, t_lapack = [], [], []
    print("     n    Python puro (s)   Gauss NumPy (s)   numpy.linalg.solve (s)   resíduo")
    for n in tamanhos:
        A = rng.random((n, n)) + n * np.eye(n)
        b = rng.random(n)
        if n <= 200:
            tp = _cronometrar(lambda: eliminacao_gauss(A.tolist(), b.tolist()), 1)
        else:
            tp = float("nan")
        tn = _cronometrar(lambda: eliminacao_gauss_numpy(A, b))
        tl = _cronometrar(lambda: np.linalg.solve(A, b))
        res = residuo(A, b, eliminacao_gauss_numpy(A, b))
        t_puro.append(tp)
        t_np.append(tn)
        t_lapack.append(tl)
        tp_txt = f"{tp:12.4f}" if tp == tp else "           -"
        print(f"  {n:4d}    {tp_txt}      {tn:12.4f}      {tl:16.5f}          {res:.1e}")

    print(
        """    O número de operações da eliminação é ~ 2n^3/3: dobrar n multiplica o
    tempo por ~8. A biblioteca LAPACK (numpy.linalg.solve) faz o mesmo método
    (fatoração LU = Gauss com pivoteamento), só que em código compilado e
    otimizado para o processador."""
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    validos = [i for i, t in enumerate(t_puro) if t == t]
    ax.loglog([tamanhos[i] for i in validos], [t_puro[i] for i in validos], "o-", label="Gauss em Python puro")
    ax.loglog(tamanhos, t_np, "s-", label="Gauss vetorizado (NumPy)")
    ax.loglog(tamanhos, t_lapack, "^-", label="numpy.linalg.solve (LAPACK)")
    ref = np.array(tamanhos, dtype=float)
    ax.loglog(ref, t_np[-1] * (ref / ref[-1]) ** 3, "k--", alpha=0.5, label="referência ~ n³")
    ax.set_xlabel("Tamanho do sistema (n)")
    ax.set_ylabel("Tempo (s)")
    ax.set_title("Custo computacional da Eliminação de Gauss")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "desempenho.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6) Método direto x métodos iterativos
# ---------------------------------------------------------------------------
def direto_vs_iterativo() -> None:
    titulo("6) COMPARAÇÃO: Gauss (direto) x Jacobi e Gauss-Seidel (iterativos)")
    rng = np.random.default_rng(7)
    n = 50
    A = rng.uniform(-1, 1, (n, n))
    A += np.diag(np.abs(A).sum(axis=1) + 1)  # garante diagonal dominante
    x_exato = rng.uniform(-5, 5, n)
    b = A @ x_exato
    print(f"Sistema {n}x{n}, diagonal dominante: {diagonal_dominante(A)}")

    x_g = np.array(eliminacao_gauss(A.tolist(), b.tolist())[0])
    x_j, hist_j = jacobi(A, b)
    x_s, hist_s = gauss_seidel(A, b)
    for nome, x, it in [
        ("Eliminação de Gauss", x_g, "-"),
        ("Jacobi", x_j, len(hist_j)),
        ("Gauss-Seidel", x_s, len(hist_s)),
    ]:
        erro = np.max(np.abs(x - x_exato))
        print(f"  {nome:20s} iterações: {str(it):>4s}   erro máximo: {erro:.2e}")
    print(
        """    O método direto chega à resposta num número fixo de passos. Os iterativos
    vão se aproximando aos poucos e só têm convergência garantida em casos
    especiais (como diagonal dominante), mas gastam pouca memória e são
    preferidos em sistemas enormes e esparsos."""
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(range(1, len(hist_j) + 1), hist_j, "o-", ms=3, label="Jacobi")
    ax.semilogy(range(1, len(hist_s) + 1), hist_s, "s-", ms=3, label="Gauss-Seidel")
    ax.axhline(np.max(np.abs(x_g - x_exato)) or 1e-16, color="k", ls="--", label="Erro da Eliminação de Gauss")
    ax.set_xlabel("Iteração")
    ax.set_ylabel("Variação máxima entre iterações")
    ax.set_title("Convergência dos métodos iterativos (sistema 50x50)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, "direto_vs_iterativo.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    exemplos_da_aula()
    balanceamento_de_carga()
    pagerank()
    analise_precisao()
    desempenho()
    direto_vs_iterativo()
    print("\nGráficos salvos na pasta resultados/")


if __name__ == "__main__":
    main()
