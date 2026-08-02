import os
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from numba import njit
from dataclasses import dataclass
import pandas as pd

@dataclass(slots=True)
class GraphCSR:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    degrees: np.ndarray
    N: int
    vol_total: int

def nx_to_csr(G: nx.Graph) -> GraphCSR:
    adj = nx.adjacency_matrix(G)
    csr = adj.tocsr()
    degrees = np.diff(csr.indptr)
    return GraphCSR(
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data.astype(np.float64),
        degrees=degrees,
        N=G.number_of_nodes(),
        vol_total=int(degrees.sum()),
    )

def calculate_metrics(G: nx.Graph, part: np.ndarray):
    cut = sum(1 for u, v in G.edges() if part[u] != part[v])
    s0 = np.sum(part == 0)
    s1 = np.sum(part == 1)
    balance = abs(s0 - s1) / len(part)
    return cut, balance, s0, s1

@njit(cache=True)
def pfv_refine_tolerance(indptr, indices, part, max_iters=10, tolerance=0.0):
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)

    # Conteo inicial O(N)
    n0 = 0
    for i in range(N):
        if part[i] == 0:
            n0 += 1
    n1 = N - n0

    for _ in range(max_iters):
        improved = False
        for u in range(N):
            curr_p = part[u]
            ext_deg = int_deg = 0
            for idx in range(indptr[u], indptr[u + 1]):
                if part[indices[idx]] == curr_p:
                    int_deg += 1
                else:
                    ext_deg += 1
            
            if ext_deg > int_deg:
                target_p = 1 - curr_p
                if curr_p == 0:
                    if n0 - 1 >= min_size and n1 + 1 <= max_size:
                        part[u] = target_p
                        n0 -= 1
                        n1 += 1
                        improved = True
                else:
                    if n1 - 1 >= min_size and n0 + 1 <= max_size:
                        part[u] = target_p
                        n1 -= 1
                        n0 += 1
                        improved = True
        if not improved:
            break
    return part

def run_sensitivity_analysis():
    print("=" * 75)
    print(" BARRIDO DE SENSIBILIDAD: TOLERANCIA DE BALANCE EN ERDŐS-RÉNYI (N=500)")
    print("=" * 75)

    np.random.seed(42)
    G_raw = nx.erdos_renyi_graph(500, 0.05, seed=42)
    G = nx.convert_node_labels_to_integers(G_raw)
    csr = nx_to_csr(G)

    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D

    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    tolerances = [0.0, 0.02, 0.05, 0.10]
    results = []

    for tol in tolerances:
        # Partición base por mediana
        part = (fiedler >= np.median(fiedler)).astype(np.int32)
        if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
            part = np.zeros(G.number_of_nodes(), dtype=np.int32)
            part[G.number_of_nodes() // 2:] = 1

        t0 = time.perf_counter()
        part_refined = pfv_refine_tolerance(csr.indptr, csr.indices, part, max_iters=10, tolerance=tol)
        t1 = time.perf_counter()

        cut, balance, s0, s1 = calculate_metrics(G, part_refined)
        results.append({
            "Tolerancia": f"{int(tol * 100)}%",
            "Distribución": f"{s0}/{s1}",
            "Corte": cut,
            "Balance": f"{balance:.4f}",
            "Tiempo (s)": f"{(t1 - t0):.5f}"
        })

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("=" * 75)

if __name__ == "__main__":
    run_sensitivity_analysis()