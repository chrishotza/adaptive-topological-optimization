# ==============================================================================
# campaign_bloc_validation.py (Versión Optimizada y Corregida)
# Campaña de Validación Estadística Multi-Semilla (30 Semillas)
# Refinador Candidato v2: BLOC-RELOC vs PFV Baseline
# ==============================================================================

import time
import warnings
import json
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass(slots=True)
class GraphCSR:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    degrees: np.ndarray
    row_indices: np.ndarray
    N: int
    vol_total: int

def nx_to_csr(G: nx.Graph) -> GraphCSR:
    adj = nx.adjacency_matrix(G)
    csr = adj.tocsr()
    degrees = np.diff(csr.indptr)
    row_indices = np.repeat(np.arange(csr.shape[0]), degrees)
    return GraphCSR(
        indptr=csr.indptr,
        indices=csr.indices,
        data=csr.data.astype(np.float64),
        degrees=degrees,
        row_indices=row_indices,
        N=G.number_of_nodes(),
        vol_total=int(degrees.sum()),
    )

def compute_cut_vectorized(row_indices, indices, p_arr):
    """Cálculo de corte vectorizado en NumPy de alto rendimiento."""
    return int(np.sum(p_arr[row_indices] != p_arr[indices])) // 2

def run_refinement(csr: GraphCSR, part_init, max_iters=20, tolerance=0.05, max_escapes=0):
    start_time = time.perf_counter()
    N = csr.N
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    part = part_init.copy()
    n0 = np.sum(part == 0)
    n1 = N - n0

    best_cut = compute_cut_vectorized(csr.row_indices, csr.indices, part)
    
    escapes_performed = 0
    blocks_evaluated = 0
    blocks_accepted = 0
    block_sizes_accepted = []
    block_deltas = []

    for iteration in range(max_iters):
        improved = False
        
        # 1. Refinamiento local estándar (movimientos individuales optimizados)
        for u in range(N):
            curr_p = part[u]
            start_idx = csr.indptr[u]
            end_idx = csr.indptr[u + 1]
            if start_idx == end_idx:
                continue
            
            neighbors = csr.indices[start_idx:end_idx]
            neighbor_parts = part[neighbors]
            
            int_deg = np.sum(neighbor_parts == curr_p)
            ext_deg = len(neighbors) - int_deg

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
                    improved = True

        current_cut = compute_cut_vectorized(csr.row_indices, csr.indices, part)
        if current_cut < best_cut:
            best_cut = current_cut

        # 2. Estancamiento: BLOC-RELOC
        if not improved and escapes_performed < max_escapes and max_escapes > 0 and iteration < max_iters - 1:
            mutated = 0
            target_blocks = max(1, int(0.01 * N))
            
            for u in range(N):
                if mutated >= target_blocks:
                    break
                curr_p = part[u]
                start_idx = csr.indptr[u]
                end_idx = csr.indptr[u + 1]
                if start_idx == end_idx:
                    continue
                
                neighbors = csr.indices[start_idx:end_idx]
                if np.any(part[neighbors] != curr_p):
                    blocks_evaluated += 1
                    same_color_neighbors = neighbors[part[neighbors] == curr_p]
                    nodes_to_move = [u] + list(same_color_neighbors[:4])
                    
                    block_size = len(nodes_to_move)
                    if block_size < 2:
                        continue
                        
                    can_shift = False
                    if curr_p == 0:
                        if n0 - block_size >= min_size and n1 + block_size <= max_size:
                            can_shift = True
                    else:
                        if n1 - block_size >= min_size and n0 + block_size <= max_size:
                            can_shift = True
                            
                    if can_shift:
                        for v_node in nodes_to_move:
                            part[v_node] = 1 - curr_p
                            
                        cut_candidate = compute_cut_vectorized(csr.row_indices, csr.indices, part)
                        
                        if cut_candidate < current_cut:
                            delta = current_cut - cut_candidate
                            if curr_p == 0:
                                n0 -= block_size
                                n1 += block_size
                            else:
                                n1 -= block_size
                                n0 += block_size
                            current_cut = cut_candidate
                            mutated += 1
                            improved = True
                            
                            blocks_accepted += 1
                            block_sizes_accepted.append(block_size)
                            block_deltas.append(delta)
                        else:
                            for v_node in nodes_to_move:
                                part[v_node] = curr_p

            if mutated > 0:
                escapes_performed += 1
                if current_cut < best_cut:
                    best_cut = current_cut
                continue
            else:
                break
        elif not improved:
            break
            
    elapsed = time.perf_counter() - start_time
    metrics = {
        "cut": int(best_cut),
        "time": elapsed,
        "blocks_evaluated": blocks_evaluated,
        "blocks_accepted": blocks_accepted,
        "avg_block_size": np.mean(block_sizes_accepted) if block_sizes_accepted else 0.0,
        "avg_block_delta": np.mean(block_deltas) if block_deltas else 0.0
    }
    return metrics

def get_spectral_partition(G, csr):
    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D
    _, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]
    part_init = (fiedler >= np.median(fiedler)).astype(np.int32)
    if np.sum(part_init == 0) == 0 or np.sum(part_init == 1) == 0:
        part_init = np.zeros(G.number_of_nodes(), dtype=np.int32)
        part_init[G.number_of_nodes() // 2:] = 1
    return part_init

def main():
    print("=" * 80)
    print(" INICIO DE CAMPAÑA DE VALIDACIÓN ESTADÍSTICA (30 SEMILLAS) - BLOC-RELOC")
    print("=" * 80)
    
    datasets = {
        "LesMis": nx.les_miserables_graph(),
        "PowerGrid_Sim": nx.powerlaw_cluster_graph(2000, 3, 0.05, seed=42),
        "Social_Network": nx.newman_watts_strogatz_graph(1500, 10, 0.1, seed=42),
        "Barabasi_Albert": nx.barabasi_albert_graph(2000, 5, seed=42),
        "Erdos_Renyi": nx.erdos_renyi_graph(2000, 0.02, seed=42)
    }
    
    num_seeds = 30
    raw_records = []
    summary_records = []
    
    for name, G_raw in datasets.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
            
        csr = nx_to_csr(G)
        print(f"\n[INSTANCIA] Procesando {name} (N={G.number_of_nodes()}, E={G.number_of_edges()}) con {num_seeds} semillas...")
        
        base_cuts = []
        bloc_cuts = []
        base_times = []
        bloc_times = []
        bloc_acc_list = []
        
        for seed in range(1, num_seeds + 1):
            np.random.seed(seed)
            part_init = get_spectral_partition(G, csr)
            
            res_base = run_refinement(csr, part_init, max_iters=20, tolerance=0.05, max_escapes=0)
            res_bloc = run_refinement(csr, part_init, max_iters=20, tolerance=0.05, max_escapes=5)
            
            base_cuts.append(res_base["cut"])
            bloc_cuts.append(res_bloc["cut"])
            base_times.append(res_base["time"])
            bloc_times.append(res_bloc["time"])
            bloc_acc_list.append(res_bloc["blocks_accepted"])
            
            raw_records.append({
                "Instancia": name,
                "Seed": seed,
                "Cut_Baseline": res_base["cut"],
                "Cut_BlocReloc": res_bloc["cut"],
                "Delta_Cut": res_bloc["cut"] - res_base["cut"],
                "Time_Baseline": res_base["time"],
                "Time_BlocReloc": res_bloc["time"],
                "Blocks_Accepted": res_bloc["blocks_accepted"],
                "Avg_Block_Size": res_bloc["avg_block_size"],
                "Avg_Block_Delta": res_bloc["avg_block_delta"]
            })
            
        b_arr = np.array(base_cuts)
        r_arr = np.array(bloc_cuts)
        
        mean_base, std_base = np.mean(b_arr), np.std(b_arr, ddof=1)
        mean_bloc, std_bloc = np.mean(r_arr), np.std(r_arr, ddof=1)
        
        if std_bloc == 0 or len(np.unique(r_arr)) == 1:
            ci_95 = (mean_bloc, mean_bloc)
        else:
            ci_95 = stats.t.interval(0.95, len(r_arr)-1, loc=mean_bloc, scale=stats.sem(r_arr))
            
        if np.array_equal(b_arr, r_arr):
            wilcoxon_p = 1.0
        else:
            try:
                w_res = stats.wilcoxon(b_arr, r_arr, alternative='two-sided')
                wilcoxon_p = w_res.pvalue
            except Exception:
                wilcoxon_p = 1.0
            
        summary_records.append({
            "Instancia": name,
            "N": G.number_of_nodes(),
            "Base_Mean": mean_base,
            "Base_Std": std_base,
            "Bloc_Mean": mean_bloc,
            "Bloc_Std": std_bloc,
            "CI95_Low": ci_95[0],
            "CI95_High": ci_95[1],
            "Wilcoxon_p": wilcoxon_p,
            "Time_Bloc_Mean": np.mean(bloc_times),
            "Mean_Blocks_Accepted": np.mean(bloc_acc_list)
        })
        print(f"  -> Baseline Mean Cut: {mean_base:.2f} ± {std_base:.2f} | BLOC-RELOC Mean Cut: {mean_bloc:.2f} ± {std_bloc:.2f} | p-value: {wilcoxon_p:.4f}")

    df_raw = pd.DataFrame(raw_records)
    df_summary = pd.DataFrame(summary_records)
    
    df_raw.to_csv("Bloc_Report_Raw.csv", index=False)
    df_summary.to_csv("Bloc_Report.csv", index=False)
    
    with open("Bloc_Report.json", "w") as f:
        json.dump({
            "summary": summary_records,
            "metadata": {"seeds": num_seeds, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        }, f, indent=4)
        
    print("\n" + "=" * 80)
    print(" CAMPAÑA FINALIZADA EXITOSAMENTE. Reportes generados:")
    print(" - Bloc_Report.csv")
    print(" - Bloc_Report_Raw.csv")
    print(" - Bloc_Report.json")
    print("=" * 80)

if __name__ == "__main__":
    main()