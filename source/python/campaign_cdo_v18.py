import os
import time
import warnings
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")

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
def pfv_refine_cdo(indptr, indices, part, max_iters=20, tolerance=0.0, lmbda=0.5, m_max=3, ping_pong_penalty=2.0):
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    n0 = 0
    for i in range(N):
        if part[i] == 0:
            n0 += 1
    n1 = N - n0

    memory = np.zeros(N, dtype=np.float64)
    last_partition = np.full(N, -1, dtype=np.int32)
    
    total_flips = 0
    total_reversions = 0

    for _ in range(max_iters):
        improved = False
        for i in range(N):
            if memory[i] > 0.0:
                memory[i] = max(0.0, memory[i] - 1.0)

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

            # Corrección de signo: buscamos nodos con mayor grado externo que interno (ext_deg > int_deg)
            # Definimos ganancia como (int_deg - ext_deg) para que sea negativa cuando conviene moverse.
            gain = float(int_deg - ext_deg) 
            
            if gain < 0: # Significa ext_deg > int_deg (mejora el corte)
                target_p = 1 - curr_p
                
                cdo_penalty = lmbda * memory[u]
                if last_partition[u] == target_p:
                    cdo_penalty += ping_pong_penalty

                # Score efectivo con penalización de memoria
                effective_score = gain + cdo_penalty

                if effective_score <= 0.0:
                    can_move = False
                    if curr_p == 0:
                        if n0 - 1 >= min_size and n1 + 1 <= max_size:
                            can_move = True
                    else:
                        if n1 - 1 >= min_size and n0 + 1 <= max_size:
                            can_move = True

                    if can_move:
                        if last_partition[u] == target_p:
                            total_reversions += 1

                        last_partition[u] = curr_p
                        part[u] = target_p
                        
                        if curr_p == 0:
                            n0 -= 1
                            n1 += 1
                        else:
                            n1 -= 1
                            n0 += 1

                        memory[u] = float(m_max)
                        total_flips += 1
                        improved = True

        if not improved:
            break
            
    return part, total_flips, total_reversions

def run_pfv_cdo_pipeline(G: nx.Graph, tolerance: float = 0.0):
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
        
    part_refined, flips, revs = pfv_refine_cdo(csr.indptr, csr.indices, part, max_iters=20, tolerance=tolerance)
    return part_refined, flips, revs

def warmup_jit():
    pfv_refine_cdo(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32), np.array([0, 1], dtype=np.int32), tolerance=0.0)

def generate_families_at_scale(N_scale: int, seed: int):
    families = {}
    half = N_scale // 2
    families["SBM_Balanced"] = nx.stochastic_block_model([half, half], [[0.1, 0.01], [0.01, 0.1]], seed=seed)
    families["Erdos_Renyi"] = nx.erdos_renyi_graph(N_scale, 0.02, seed=seed)
    families["Barabasi_Albert"] = nx.barabasi_albert_graph(N_scale, 5, seed=seed)
    return families

def execute_cdo_campaign():
    print("=" * 125)
    print(" CAMPAÑA EXPERIMENTAL PFV v18-CDO (Corregida: Activa con Memoria y Anti-Ping-Pong)")
    print("=" * 125)
    warmup_jit()
    
    seeds = list(range(1000, 1030))
    scale_sizes = [500, 2000]
    all_records = []

    for N_scale in scale_sizes:
        print(f"\n[EJECUCIÓN] Escala N = {N_scale} | Evaluando CDO activo en 30 semillas...")
        for seed in seeds:
            families = generate_families_at_scale(N_scale, seed)
            for family_name, G_raw in families.items():
                G = nx.convert_node_labels_to_integers(G_raw)
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                    G = nx.convert_node_labels_to_integers(G)

                part_cdo, flips, revs = run_pfv_cdo_pipeline(G, tolerance=0.0)
                cut_cdo, _ = calculate_metrics(G, part_cdo)
                
                part_metis = run_metis(G)
                cut_metis, _ = calculate_metrics(G, part_metis)

                all_records.append({"Familia": family_name, "N": N_scale, "Seed": seed, "Algoritmo": "PFV v18-CDO", "Corte": cut_cdo, "Flips": flips, "Reversions": revs})
                all_records.append({"Familia": family_name, "N": N_scale, "Seed": seed, "Algoritmo": "METIS", "Corte": cut_metis, "Flips": 0, "Reversions": 0})

    df = pd.DataFrame(all_records)
    
    print("\nProcesando estadísticas inferenciales y métricas de convergencia CDO...")
    stats_results = []
    
    for (family, n_scale), group in df.groupby(["Familia", "N"]):
        cdo_cuts = group[group["Algoritmo"] == "PFV v18-CDO"]["Corte"].values
        metis_cuts = group[group["Algoritmo"] == "METIS"]["Corte"].values
        cdo_flips = group[group["Algoritmo"] == "PFV v18-CDO"]["Flips"].values
        cdo_revs = group[group["Algoritmo"] == "PFV v18-CDO"]["Reversions"].values
        
        cdo_mean, cdo_std = np.mean(cdo_cuts), np.std(cdo_cuts, ddof=1)
        metis_mean, metis_std = np.mean(metis_cuts), np.std(metis_cuts, ddof=1)
        
        delta_pct = 100 * (cdo_mean - metis_mean) / metis_mean if metis_mean > 0 else 0.0
        
        se = stats.sem(cdo_cuts)
        ci_lower, ci_upper = stats.t.interval(0.95, len(cdo_cuts)-1, loc=cdo_mean, scale=se) if se > 0 else (cdo_mean, cdo_mean)
            
        diffs = cdo_cuts - metis_cuts
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            res = stats.wilcoxon(cdo_cuts, metis_cuts)
            p_val = res.pvalue
            
        significativo = "Sí" if p_val < 0.05 else "No"
            
        stats_results.append({
            "Grafo": f"{family} ({n_scale})",
            "CDO Media ± Std": f"{cdo_mean:.1f} ± {cdo_std:.1f}",
            "CDO IC 95%": f"[{ci_lower:.1f}, {ci_upper:.1f}]",
            "METIS Media": f"{metis_mean:.1f}",
            "Δ (%)": f"{delta_pct:+.2f}%",
            "p-value": f"{p_val:.4f}",
            "Flips Medios": f"{np.mean(cdo_flips):.1f}",
            "Reversiones": f"{np.mean(cdo_revs):.1f}",
            "Sig.": significativo
        })

    df_stats = pd.DataFrame(stats_results)
    print("\n" + "=" * 125)
    print(" [TABLA COMPARATIVA DEFINITIVA: PFV v18-CDO ACTIVO vs METIS]")
    print("=" * 125)
    print(df_stats.to_string(index=False))
    print("=" * 125)

if __name__ == "__main__":
    execute_cdo_campaign()
