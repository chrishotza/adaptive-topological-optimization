import warnings
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp
import scipy.stats as stats
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
    balance_ratio = abs(s0 - s1) / len(part)
    return cut, balance_ratio

def run_metis_standard(G: nx.Graph):
    adj_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
    _, parts = pymetis.part_graph(2, adjacency=adj_list)
    return np.array(parts)

@njit(cache=True)
def pfv_refine_v171(indptr, indices, part, max_iters=20, tolerance=0.0):
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    n0 = 0
    for i in range(N):
        if part[i] == 0:
            n0 += 1
    n1 = N - n0

    total_flips = 0

    for _ in range(max_iters):
        improved = False
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
                    improved = True

        if not improved:
            break
            
    return part, total_flips

def run_pipeline(G: nx.Graph, tolerance: float):
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
        
    part_refined, flips = pfv_refine_v171(csr.indptr, csr.indices, part, max_iters=20, tolerance=tolerance)
    return part_refined, flips

def warmup_jit():
    pfv_refine_v171(np.array([0, 1, 2], dtype=np.int32), np.array([1, 0], dtype=np.int32), np.array([0, 1], dtype=np.int32), tolerance=0.0)

def generate_families_at_scale(N_scale: int, seed: int):
    families = {}
    half = N_scale // 2
    families["SBM_Balanced"] = nx.stochastic_block_model([half, half], [[0.1, 0.01], [0.01, 0.1]], seed=seed)
    families["Erdos_Renyi"] = nx.erdos_renyi_graph(N_scale, 0.02, seed=seed)
    families["Barabasi_Albert"] = nx.barabasi_albert_graph(N_scale, 5, seed=seed)
    return families

def execute_fair_campaign():
    print("=" * 140)
    print(" CAMPAÑA EXPERIMENTAL: BARRIDO DE TOLERANCIA PFV v17.1 vs METIS (Nativo)")
    print("=" * 140)
    warmup_jit()
    
    seeds = list(range(1000, 1030))
    scale_sizes = [500, 2000]
    tolerances = [0.0, 0.01, 0.02, 0.05]
    all_records = []

    for N_scale in scale_sizes:
        for tol in tolerances:
            print(f"[EJECUCIÓN] Escala N = {N_scale} | PFV Tolerancia = {tol*100:.1f}% | 30 Semillas...")
            for seed in seeds:
                families = generate_families_at_scale(N_scale, seed)
                for family_name, G_raw in families.items():
                    G = nx.convert_node_labels_to_integers(G_raw)
                    if not nx.is_connected(G):
                        largest_cc = max(nx.connected_components(G), key=len)
                        G = G.subgraph(largest_cc).copy()
                        G = nx.convert_node_labels_to_integers(G)

                    # PFV con tolerancia específica del barrido
                    part_pfv, flips = run_pipeline(G, tolerance=tol)
                    cut_pfv, balance_pfv = calculate_metrics(G, part_pfv)
                    
                    # METIS con su configuración nativa estándar
                    part_metis = run_metis_standard(G)
                    cut_metis, balance_metis = calculate_metrics(G, part_metis)

                    all_records.append({
                        "Familia": family_name, "N": N_scale, "Tolerancia": f"{tol*100:.1f}%", 
                        "Seed": seed, "Algoritmo": f"PFV (tol={tol*100:.1f}%)", "Corte": cut_pfv, 
                        "Flips": flips, "Balance": balance_pfv
                    })
                    all_records.append({
                        "Familia": family_name, "N": N_scale, "Tolerancia": f"{tol*100:.1f}%", 
                        "Seed": seed, "Algoritmo": "METIS", "Corte": cut_metis, 
                        "Flips": 0, "Balance": balance_metis
                    })

    df = pd.DataFrame(all_records)
    
    print("\nProcesando estadísticas inferenciales de la campaña...")
    stats_results = []
    
    for (family, n_scale, tol_str), group in df.groupby(["Familia", "N", "Tolerancia"]):
        pfv_cuts = group[group["Algoritmo"].str.startswith("PFV")]["Corte"].values
        metis_cuts = group[group["Algoritmo"] == "METIS"]["Corte"].values
        pfv_flips = group[group["Algoritmo"].str.startswith("PFV")]["Flips"].values
        pfv_balances = group[group["Algoritmo"].str.startswith("PFV")]["Balance"].values
        
        pfv_mean, pfv_std = np.mean(pfv_cuts), np.std(pfv_cuts, ddof=1)
        metis_mean, metis_std = np.mean(metis_cuts), np.std(metis_cuts, ddof=1)
        
        delta_pct = 100 * (pfv_mean - metis_mean) / metis_mean if metis_mean > 0 else 0.0
        
        se = stats.sem(pfv_cuts)
        ci_lower, ci_upper = stats.t.interval(0.95, len(pfv_cuts)-1, loc=pfv_mean, scale=se) if se > 0 else (pfv_mean, pfv_mean)
            
        diffs = pfv_cuts - metis_cuts
        if np.all(diffs == 0):
            p_val = 1.0
        else:
            res = stats.wilcoxon(pfv_cuts, metis_cuts)
            p_val = res.pvalue
            
        significativo = "Sí" if p_val < 0.05 else "No"
            
        stats_results.append({
            "Escenario": f"{family} ({n_scale})",
            "Tol. PFV": tol_str,
            "PFV Media ± Std": f"{pfv_mean:.1f} ± {pfv_std:.1f}",
            "IC 95%": f"[{ci_lower:.1f}, {ci_upper:.1f}]",
            "METIS Media ± Std": f"{metis_mean:.1f} ± {metis_std:.1f}",
            "Δ Gap (%)": f"{delta_pct:+.2f}%",
            "Flips": f"{np.mean(pfv_flips):.1f}",
            "p-value": f"{p_val:.4f}",
            "Sig.": significativo
        })

    df_stats = pd.DataFrame(stats_results)
    print("\n" + "=" * 145)
    print(" [TABLA DEFINITIVA: PFV v17.1 (BARRIDO DE TOLERANCIA) vs METIS NATIVO]")
    print("=" * 145)
    print(df_stats.to_string(index=False))
    print("=" * 145)

if __name__ == "__main__":
    execute_fair_campaign()