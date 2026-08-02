import os
import time
import tracemalloc
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

# Suprimir warnings de Wilcoxon en caso de empates perfectos (ej. SBM)
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
    pfv_refine_strict(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32), np.array([0, 1], dtype=np.int32), tolerance=0.0)

def generate_families_at_scale(N_scale: int, seed: int):
    families = {}
    half = N_scale // 2
    families["SBM_Balanced"] = nx.stochastic_block_model([half, half], [[0.1, 0.01], [0.01, 0.1]], seed=seed)
    families["Erdos_Renyi"] = nx.erdos_renyi_graph(N_scale, 0.02, seed=seed)
    families["Barabasi_Albert"] = nx.barabasi_albert_graph(N_scale, 5, seed=seed)
    return families

def execute_statistical_campaign():
    print("=" * 110)
    print(" CAMPAÑA DE INFERENCIA ESTADÍSTICA (Fase 1: Preparación para Publicación)")
    print("=" * 110)
    warmup_jit()
    
    # 7 semillas para otorgar validez al test de Wilcoxon
    seeds = [42, 100, 1024, 2024, 2026, 9999, 12345]
    scale_sizes = [500, 2000]
    all_records = []

    for N_scale in scale_sizes:
        print(f"\n[EJECUCIÓN] Escala N = {N_scale} | 7 Semillas...")
        for seed in seeds:
            families = generate_families_at_scale(N_scale, seed)
            for family_name, G_raw in families.items():
                G = nx.convert_node_labels_to_integers(G_raw)
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                    G = nx.convert_node_labels_to_integers(G)

                # PFV
                part_pfv = run_pfv_sota(G, tolerance=0.0)
                cut_pfv, _ = calculate_metrics(G, part_pfv)
                
                # METIS
                part_metis = run_metis(G)
                cut_metis, _ = calculate_metrics(G, part_metis)

                all_records.append({"Familia": family_name, "N": N_scale, "Seed": seed, "Algoritmo": "PFV v17.1", "Corte": cut_pfv})
                all_records.append({"Familia": family_name, "N": N_scale, "Seed": seed, "Algoritmo": "METIS", "Corte": cut_metis})

    df = pd.DataFrame(all_records)
    
    # INFERENCIA ESTADÍSTICA
    print("\nCalculando inferencias estadísticas (Wilcoxon, Intervalos de Confianza, Delta %)...")
    stats_results = []
    
    for (family, n_scale), group in df.groupby(["Familia", "N"]):
        pfv_cuts = group[group["Algoritmo"] == "PFV v17.1"]["Corte"].values
        metis_cuts = group[group["Algoritmo"] == "METIS"]["Corte"].values
        
        pfv_mean, pfv_std = np.mean(pfv_cuts), np.std(pfv_cuts)
        metis_mean, metis_std = np.mean(metis_cuts), np.std(metis_cuts)
        
        # Delta relativo
        delta_pct = 100 * (pfv_mean - metis_mean) / metis_mean if metis_mean > 0 else 0.0
        
        # Intervalo de confianza (95%) para PFV
        se = stats.sem(pfv_cuts)
        if se > 0:
            ci_lower, ci_upper = stats.t.interval(0.95, len(pfv_cuts)-1, loc=pfv_mean, scale=se)
        else:
            ci_lower, ci_upper = pfv_mean, pfv_mean
            
        # Prueba de rangos con signo de Wilcoxon
        diffs = pfv_cuts - metis_cuts
        if np.all(diffs == 0):
            p_val = 1.0 # Empate exacto en todas las semillas
        else:
            res = stats.wilcoxon(pfv_cuts, metis_cuts)
            p_val = res.pvalue
            
        significativo = "Sí" if p_val < 0.05 else "No"
            
        stats_results.append({
            "Grafo": f"{family} ({n_scale})",
            "PFV Media ± Std": f"{pfv_mean:.1f} ± {pfv_std:.1f}",
            "PFV IC 95%": f"[{ci_lower:.1f}, {ci_upper:.1f}]",
            "METIS Media": f"{metis_mean:.1f}",
            "Δ (%)": f"{delta_pct:+.2f}%",
            "p-value": f"{p_val:.4f}",
            "Sig.": significativo
        })

    df_stats = pd.DataFrame(stats_results)
    print("\n" + "=" * 110)
    print(" [TABLA DE INFERENCIA ESTADÍSTICA]")
    print("=" * 110)
    print(df_stats.to_string(index=False))
    print("=" * 110)

    # VISUALIZACIÓN (Violin Plots)
    print("\nGenerando gráficos de distribución...")
    df['Escenario'] = df['Familia'] + " (N=" + df['N'].astype(str) + ")"
    plt.figure(figsize=(14, 7))
    sns.violinplot(data=df, x="Escenario", y="Corte", hue="Algoritmo", split=True, inner="quart", palette="muted")
    plt.title("Distribución de Cortes: PFV v17.1 vs METIS (7 Semillas)", fontsize=14, pad=15)
    plt.ylabel("Corte", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = "results_violin.png"
    plt.savefig(plot_path, dpi=300)
    print(f"Gráfico guardado exitosamente en: {plot_path}")

if __name__ == "__main__":
    execute_statistical_campaign()
