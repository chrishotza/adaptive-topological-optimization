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


@njit(cache=True)
def pfv_refine_strict(indptr, indices, part, max_iters=10, tolerance=0.0):
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


def generate_families_at_scale(N_scale: int, seed: int):
    families = {}
    half = N_scale // 2
    families[f"SBM_Balanced_{N_scale}"] = nx.stochastic_block_model([half, half], [[0.1, 0.01], [0.01, 0.1]], seed=seed)
    families[f"Erdos_Renyi_{N_scale}"] = nx.erdos_renyi_graph(N_scale, 0.02, seed=seed)
    families[f"Barabasi_Albert_{N_scale}"] = nx.barabasi_albert_graph(N_scale, 5, seed=seed)
    return families


def execute_multiseed_campaign():
    print("=" * 100)
    print(" CAMPAÑA ESTADÍSTICA MULTI-SEMILLA Y ESCALABILIDAD (PFV v17.1 vs METIS)")
    print("=" * 100)

    warmup_jit()
    
    seeds = [42, 100, 2026]
    scale_sizes = [500, 2000]
    all_records = []

    for N_scale in scale_sizes:
        print(f"\n[ESCALA] Evaluando tamaño N = {N_scale}")
        for seed in seeds:
            families = generate_families_at_scale(N_scale, seed)
            for family_name, G_raw in families.items():
                G = nx.convert_node_labels_to_integers(G_raw)
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                    G = nx.convert_node_labels_to_integers(G)

                # Evaluar PFV
                t0 = time.perf_counter()
                part_pfv = run_pfv_sota(G, tolerance=0.0)
                t_pfv = time.perf_counter() - t0
                cut_pfv, _ = calculate_metrics(G, part_pfv)

                # Evaluar METIS
                t0 = time.perf_counter()
                part_metis = run_metis(G)
                t_metis = time.perf_counter() - t0
                cut_metis, _ = calculate_metrics(G, part_metis)

                all_records.append({"Familia": family_name, "Seed": seed, "Algoritmo": "PFV v17.1", "Corte": cut_pfv, "Tiempo": t_pfv})
                all_records.append({"Familia": family_name, "Seed": seed, "Algoritmo": "METIS", "Corte": cut_metis, "Tiempo": t_metis})

    df = pd.DataFrame(all_records)
    
    summary = df.groupby(["Familia", "Algoritmo"]).agg(
        Corte_Mean=("Corte", "mean"),
        Corte_Std=("Corte", "std"),
        Tiempo_Mean=("Tiempo", "mean")
    ).reset_index()

    summary["Corte_Mean"] = summary["Corte_Mean"].round(1)
    summary["Corte_Std"] = summary["Corte_Std"].fillna(0).round(1)
    summary["Tiempo_Mean"] = summary["Tiempo_Mean"].round(4)

    print("\n" + "=" * 100)
    print(" [RESUMEN ESTADÍSTICO MULTI-SEMILLA Y ESCALABILIDAD]")
    print("=" * 100)
    print(summary.to_string(index=False))
    print("=" * 100)
    print(" Campaña estadística avanzada completada con éxito.")


if __name__ == "__main__":
    execute_multiseed_campaign()
