import os
import time
import networkx as nx
import numpy as np
import scipy.sparse as sp
from numba import njit
from dataclasses import dataclass

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
def pfv_refine_strict_bisection(indptr, indices, part, max_iters=10, tolerance=0.05):
    N = len(part)
    min_size = int((0.5 - tolerance) * N)  # Límite inferior estricto (ej. 225 para N=500)
    max_size = int((0.5 + tolerance) * N)  # Límite superior estricto (ej. 275 para N=500)

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
                # Guard-rail con límites estrictos de bisección simétrica O(1)
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

def run_pfv_strict_er(G: nx.Graph):
    csr = nx_to_csr(G)
    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D

    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    part = (fiedler >= np.median(fiedler)).astype(np.int32)
    
    # Resguardo inicial por si la mediana colapsa
    if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
        part = np.zeros(G.number_of_nodes(), dtype=np.int32)
        part[G.number_of_nodes() // 2:] = 1

    return pfv_refine_strict_bisection(csr.indptr, csr.indices, part)

if __name__ == "__main__":
    print("=" * 75)
    print(" PRUEBA AISLADA: ERDŐS-RÉNYI (N=500) CON RESTRICCION ESTRICTA DE BALANCE")
    print("=" * 75)
    
    np.random.seed(42)
    G = nx.erdos_renyi_graph(500, 0.05, seed=42)
    G = nx.convert_node_labels_to_integers(G)

    t0 = time.perf_counter()
    part = run_pfv_strict_er(G)
    t1 = time.perf_counter()

    cut, balance, s0, s1 = calculate_metrics(G, part)
    
    print(f" -> Corte obtenido: {cut}")
    print(f" -> Balance: {balance:.4f}")
    print(f" -> Distribución: S0={s0}, S1={s1}")
    print(f" -> Tiempo de ejecución: {(t1 - t0):.5f} s")
    print("=" * 75)