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
    return cut, balance

@njit(cache=True)
def pfv_refine_safe(indptr, indices, part, max_iters=10):
    N = len(part)
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
            # Condición de cambio protegida para evitar colapso total
            if ext_deg > int_deg:
                target_p = 1 - curr_p
                # Verificar que no dejemos un bloque vacío (protección de balance extremo)
                count_target = np.sum(part == target_p)
                if count_target + 1 < N - 1:  
                    part[u] = target_p
                    improved = True
        if not improved:
            break
    return part

def run_pfv_debug_er(G: nx.Graph):
    csr = nx_to_csr(G)
    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D

    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    med = np.median(fiedler)
    part = (fiedler >= med).astype(np.int32)

    # CHECK DE SEGURIDAD CONTRA DEGENERACIÓN: Si la mediana dejó un bloque vacío, forzar bisección geométrica pura
    if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
        print("    [!] ADVERTENCIA: Colapso espectral detectado en ER. Aplicando bisección de respaldo por índice.")
        part = np.zeros(G.number_of_nodes(), dtype=np.int32)
        part[G.number_of_nodes() // 2:] = 1

    return pfv_refine_safe(csr.indptr, csr.indices, part)

def debug_execution():
    print("=" * 70)
    print(" MODO DEBUG: AISLANDO INSTANCIA ERDŐS-RÉNYI (N=500)")
    print("=" * 70)

    np.random.seed(42)
    G = nx.erdos_renyi_graph(500, 0.05, seed=42)
    G = nx.convert_node_labels_to_integers(G)

    print(f" -> Grafo generado: N={G.number_of_nodes()}, E={G.number_of_edges()}")
    
    t0 = time.perf_counter()
    part = run_pfv_debug_er(G)
    t1 = time.perf_counter()

    cut, balance = calculate_metrics(G, part)
    print("\n [RESULTADOS DEPURADOS - ERDŐS-RÉNYI]")
    print(f"    - Corte: {cut}")
    print(f"    - Balance: {balance:.4f}")
    print(f"    - Tiempo: {(t1 - t0):.5f} s")
    print(f"    - Distribución de bloques: S0={np.sum(part == 0)}, S1={np.sum(part == 1)}")
    print("=" * 70)

if __name__ == "__main__":
    debug_execution()