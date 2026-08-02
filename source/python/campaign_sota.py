import os
import time
import tracemalloc
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp
from numba import njit


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


def run_metis(G: nx.Graph):
    adj_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
    _, parts = pymetis.part_graph(2, adjacency=adj_list)
    return np.array(parts)


def run_louvain(G: nx.Graph):
    communities = list(nx.community.greedy_modularity_communities(G))
    parts = np.zeros(G.number_of_nodes(), dtype=int)
    if len(communities) > 1:
        for node in communities[1]:
            parts[node] = 1
    return parts


def run_kernighan_lin(G: nx.Graph):
    _, part1 = nx.community.kernighan_lin_bisection(G)
    parts = np.zeros(G.number_of_nodes(), dtype=int)
    for node in part1:
        parts[node] = 1
    return parts


@njit(cache=True)
def pfv_refine_strict(indptr, indices, part, max_iters=10, tolerance=0.0):
    """Refinador local certificado con estricta restricción de bisección simétrica O(1)."""
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)

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


def run_pfv_sota(G: nx.Graph, tolerance: float = 0.0):
    csr = nx_to_csr(G)
    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D

    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    part = (fiedler >= np.median(fiedler)).astype(np.int32)
    
    # Resguardo estructural de emergencia si el TopSolver deja un bloque vacío
    if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
        part = np.zeros(G.number_of_nodes(), dtype=np.int32)
        part[G.number_of_nodes() // 2:] = 1

    return pfv_refine_strict(csr.indptr, csr.indices, part, max_iters=10, tolerance=tolerance)


def warmup_jit():
    pfv_refine_strict(
        np.array([0, 1, 2], dtype=np.int32),
        np.array([1, 0], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        tolerance=0.0
    )


def generate_graph_families():
    families = {}
    families["SBM_Balanced_500"] = nx.stochastic_block_model([250, 250], [[0.1, 0.01], [0.01, 0.1]], seed=42)
    families["SBM_Dense_500"] = nx.stochastic_block_model([250, 250], [[0.2, 0.05], [0.05, 0.2]], seed=42)
    families["Erdos_Renyi_500"] = nx.erdos_renyi_graph(500, 0.05, seed=42)
    families["Barabasi_Albert_500"] = nx.barabasi_albert_graph(500, 5, seed=42)
    families["Grid_20x25"] = nx.grid_2d_graph(20, 25)
    return families


def execute_campaign():
    print("=" * 95)
    print(" CAMPAÑA EXPERIMENTAL SOTA UNIFICADA (Tolerancia de Bisección Fija: 0%)")
    print("=" * 95)

    print("\n[WARM-UP] Ejecutando warm-up de Numba JIT para PFV v17.1...")
    warmup_jit()
    print("[WARM-UP] Rutinas listas. Iniciando barrido experimental...\n")

    families = generate_graph_families()
    algorithms = {
        "PFV v17.1": lambda g: run_pfv_sota(g, tolerance=0.0),
        "METIS": run_metis,
        "Louvain/Mod": run_louvain,
        "Kernighan-Lin": run_kernighan_lin,
    }

    campaign_results = []

    for family_name, G_raw in families.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)

        print(f" -> Evaluando Familia: {family_name} (N={G.number_of_nodes()}, E={G.number_of_edges()})")

        for algo_name, func in algorithms.items():
            tracemalloc.start()
            t0 = time.perf_counter()

            try:
                part = func(G)
                t1 = time.perf_counter()
                _, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                cut, balance = calculate_metrics(G, part)
                campaign_results.append({
                    "Familia": family_name,
                    "Nodos": G.number_of_nodes(),
                    "Algoritmo": algo_name,
                    "Corte": cut,
                    "Balance": f"{balance:.4f}",
                    "Tiempo (s)": f"{(t1 - t0):.5f}",
                    "Memoria (MB)": f"{peak_mem / 1048576:.2f}",
                    "Estado": "OK"
                })
            except Exception as e:
                tracemalloc.stop()
                campaign_results.append({
                    "Familia": family_name,
                    "Nodos": G.number_of_nodes(),
                    "Algoritmo": algo_name,
                    "Corte": "N/A",
                    "Balance": "N/A",
                    "Tiempo (s)": "N/A",
                    "Memoria (MB)": "N/A",
                    "Estado": f"Error: {e}"
                })

    df_results = pd.DataFrame(campaign_results)
    print("\n" + "=" * 95)
    print(" [RESULTADOS FINALES DE LA CAMPAÑA SOTA UNIFICADA]")
    print("=" * 95)
    print(df_results.to_string(index=False))
    print("=" * 95)
    print(" Campaña experimental unificada finalizada con éxito.")


if __name__ == "__main__":
    execute_campaign()
