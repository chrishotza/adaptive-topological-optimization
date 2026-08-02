import os
import time
import warnings
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp
from numba import njit

warnings.filterwarnings("ignore", category=UserWarning)

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
    balance_ratio = abs(s0 - s1) / len(part)
    return cut, balance_ratio

def run_metis_timed(G: nx.Graph):
    start_time = time.perf_counter()
    adj_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
    _, parts = pymetis.part_graph(2, adjacency=adj_list)
    elapsed = time.perf_counter() - start_time
    return np.array(parts), elapsed

@njit(cache=True)
def pfv_refine_with_calibrated_enlc(indptr, indices, part, max_iters=20, tolerance=0.05, max_escapes=3):
    """
    Refinador PFV con Escape No Local Controlado (ENLC) Calibrado.
    - Activación por estancamiento natural (0 flips en una pasada).
    - Filtro de candidatos relajado: selecciona nodos frontera con menor ganancia local,
      respetando estrictamente los límites de balance (Tolerancia 5%).
    """
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    n0 = 0
    for i in range(N):
        if part[i] == 0:
            n0 += 1
    n1 = N - n0

    total_flips = 0
    escapes_performed = 0
    escape_logs = []

    for iteration in range(max_iters):
        improved = False
        pass_flips = 0
        
        for u in range(N):
            curr_p = part[u]
            ext_deg = 0
            int_deg = 0
            for idx in range(indptr[u], indptr[u + 1]):
                v = indices[idx]
                if part[v] == curr_p:
                    int_deg += 1
                else:
                    ext_deg += 1

            gain = float(int_deg - ext_deg)
            
            if gain < 0:
                can_move = False
                if curr_p == 0:
                    if n0 - 1 >= min_size and n1 + 1 <= max_size:
                        can_move = True
                else:
                    if n1 - 1 >= min_size and n0 + 1 <= max_size:
                        can_move = True

                if can_move:
                    part[u] = 1 - curr_p
                    if curr_p == 0:
                        n0 -= 1
                        n1 += 1
                    else:
                        n1 -= 1
                        n0 += 1
                    total_flips += 1
                    pass_flips += 1
                    improved = True

        # ESTANCAMIENTO DETECTADO: Aplicar ENLC Calibrado
        if not improved and escapes_performed < max_escapes and iteration < max_iters - 1:
            cut_before_escape = 0
            for u in range(N):
                curr_p = part[u]
                for idx in range(indptr[u], indptr[u + 1]):
                    if part[indices[idx]] != curr_p:
                        cut_before_escape += 1
            cut_before_escape //= 2

            mutated = 0
            target_mutations = max(1, int(0.01 * N))
            
            for u in range(N):
                if mutated >= target_mutations:
                    break
                curr_p = part[u]
                int_d = 0
                ext_d = 0
                for idx in range(indptr[u], indptr[u + 1]):
                    if part[indices[idx]] == curr_p:
                        int_d += 1
                    else:
                        ext_d += 1
                
                # Filtro calibrado: nodos en frontera (ext_d > 0)
                if ext_d > 0:
                    if curr_p == 0 and (n0 - 1 >= min_size and n1 + 1 <= max_size):
                        part[u] = 1
                        n0 -= 1
                        n1 += 1
                        total_flips += 1
                        mutated += 1
                    elif curr_p == 1 and (n1 - 1 >= min_size and n0 + 1 <= max_size):
                        part[u] = 0
                        n1 -= 1
                        n0 += 1
                        total_flips += 1
                        mutated += 1

            if mutated > 0:
                escapes_performed += 1
                cut_after_escape = 0
                for u in range(N):
                    curr_p = part[u]
                    for idx in range(indptr[u], indptr[u + 1]):
                        if part[indices[idx]] != curr_p:
                            cut_after_escape += 1
                cut_after_escape //= 2

                escape_logs.append({
                    "escape_id": escapes_performed,
                    "iteracion": iteration,
                    "nodos_afectados": mutated,
                    "corte_previo": cut_before_escape,
                    "corte_post_escape": cut_after_escape
                })
                
                continue
            else:
                break
        elif not improved:
            break
            
    return part, total_flips, escapes_performed, escape_logs

def run_pipeline_with_calibrated_enlc(G: nx.Graph):
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
        
    part_base, _, _, _ = pfv_refine_with_calibrated_enlc(csr.indptr, csr.indices, part.copy(), max_iters=20, tolerance=0.05, max_escapes=0)
    cut_base, _ = calculate_metrics(G, part_base)
    
    part_enlc, _, escapes_count, logs = pfv_refine_with_calibrated_enlc(csr.indptr, csr.indices, part.copy(), max_iters=20, tolerance=0.05, max_escapes=3)
    cut_enlc, _ = calculate_metrics(G, part_enlc)
    
    return cut_base, cut_enlc, escapes_count, logs

def warmup_jit():
    dummy_indptr = np.array([0, 1, 2], dtype=np.int32)
    dummy_indices = np.array([1, 0], dtype=np.int32)
    dummy_part = np.array([0, 1], dtype=np.int32)
    pfv_refine_with_calibrated_enlc(dummy_indptr, dummy_indices, dummy_part, max_iters=2, tolerance=0.05, max_escapes=1)

def execute_etapa2b_campaign():
    print("=" * 130)
    print(" ETAPA 2B: VALIDACIÓN DEL ESCAPE NO LOCAL CONTROLADO (ENLC) CALIBRADO")
    print("=" * 130)
    warmup_jit()
    
    datasets = {
        "Barabasi_Albert": nx.barabasi_albert_graph(2000, 5, seed=42),
        "Erdos_Renyi": nx.erdos_renyi_graph(2000, 0.02, seed=42),
        "PowerGrid_Sim": nx.powerlaw_cluster_graph(2000, 3, 0.05, seed=42),
        "Social_Network": nx.newman_watts_strogatz_graph(1500, 10, 0.1, seed=42)
    }
    
    results = []
    
    for name, G_raw in datasets.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
            
        part_metis, _ = run_metis_timed(G)
        cut_metis, _ = calculate_metrics(G, part_metis)
        
        cut_base, cut_enlc, escapes_count, logs = run_pipeline_with_calibrated_enlc(G)
        
        delta_vs_base = cut_enlc - cut_base
        delta_vs_metis = cut_enlc - cut_metis
        
        results.append({
            "Instancia": name,
            "N": G.number_of_nodes(),
            "METIS": cut_metis,
            "PFV Baseline": cut_base,
            "PFV + ENLC Calibrated": cut_enlc,
            "Δ vs Baseline": f"{delta_vs_base:+d}",
            "Δ vs METIS": f"{delta_vs_metis:+d}",
            "Escapes": escapes_count
        })
        
        print(f"[REGISTRO] {name:<18} | Escapes ejecutados: {escapes_count} | Logs de transición: {logs}")
        
    df = pd.DataFrame(results)
    print("\n" + "=" * 130)
    print(" [TABLA CONSOLIDADA ETAPA 2B: RENDIMIENTO DE ENLC CALIBRADO]")
    print("=" * 130)
    print(df.to_string(index=False))
    print("=" * 130)

if __name__ == "__main__":
    execute_etapa2b_campaign()
