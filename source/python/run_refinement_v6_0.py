# ==============================================================================
# run_refinement_v6_0.py
# BLOC-RELOC v6.0: Generador de Dataset para Meta-Aprendizaje
# ==============================================================================

import time
import warnings
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

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
    return int(np.sum(p_arr[row_indices] != p_arr[indices])) // 2

def run_refinement_modular(csr: GraphCSR, part_init, strategy="frontier", max_iters=20, tolerance=0.05, max_escapes=5):
    start_time = time.perf_counter()
    N = csr.N
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    part = part_init.copy()
    n0 = np.sum(part == 0)
    n1 = N - n0

    best_cut = compute_cut_vectorized(csr.row_indices, csr.indices, part)
    escapes_performed = 0
    blocks_accepted = 0

    for iteration in range(max_iters):
        improved = False
        
        for u in range(N):
            curr_p = part[u]
            start_idx, end_idx = csr.indptr[u], csr.indptr[u + 1]
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
                        n0, n1 = n0 - 1, n1 + 1
                    else:
                        n1, n0 = n1 - 1, n0 + 1
                    improved = True

        current_cut = compute_cut_vectorized(csr.row_indices, csr.indices, part)
        if current_cut < best_cut:
            best_cut = current_cut

        if not improved and escapes_performed < max_escapes and iteration < max_iters - 1:
            mutated = 0
            target_blocks = max(1, int(0.01 * N))
            
            for u in range(N):
                if mutated >= target_blocks:
                    break
                curr_p = part[u]
                start_idx, end_idx = csr.indptr[u], csr.indptr[u + 1]
                if start_idx == end_idx:
                    continue
                
                neighbors = csr.indices[start_idx:end_idx]
                neighbor_parts = part[neighbors]
                
                if np.any(neighbor_parts != curr_p):
                    same_color_mask = (neighbor_parts == curr_p)
                    same_color_neighbors = neighbors[same_color_mask]
                    
                    if len(same_color_neighbors) == 0:
                        continue
                    
                    if strategy == "cohesion":
                        u_neighbors_set = set(neighbors)
                        cohesion_scores = []
                        for v in same_color_neighbors:
                            v_start, v_end = csr.indptr[v], csr.indptr[v + 1]
                            v_neighbors_set = set(csr.indices[v_start:v_end])
                            shared_count = len(u_neighbors_set.intersection(v_neighbors_set))
                            cohesion_scores.append((shared_count, v))
                        cohesion_scores.sort(key=lambda x: x[0], reverse=True)
                        dynamic_limit = min(6, max(1, len(cohesion_scores)))
                        nodes_to_move = [u] + [v for _, v in cohesion_scores[:dynamic_limit]]
                    else:
                        dynamic_limit = min(8, max(2, len(same_color_neighbors) // 2))
                        nodes_to_move = [u] + list(same_color_neighbors[:dynamic_limit])
                        
                    block_size = len(nodes_to_move)
                    
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
                            if curr_p == 0:
                                n0, n1 = n0 - block_size, n1 + block_size
                            else:
                                n1, n0 = n1 - block_size, n0 + block_size
                            current_cut = cut_candidate
                            mutated += 1
                            improved = True
                            blocks_accepted += 1
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
    return int(best_cut), elapsed, blocks_accepted

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

def extract_rich_descriptor_vector(G, csr):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    degrees = csr.degrees
    
    clustering = nx.average_clustering(G)
    deg_mean = np.mean(degrees)
    deg_std = np.std(degrees)
    deg_het = deg_std / (deg_mean + 1e-6)
    density = (2.0 * E) / (N * (N - 1))
    assortativity = nx.degree_assortativity_coefficient(G)
    if np.isnan(assortativity):
        assortativity = 0.0
        
    return {
        "N": N,
        "E": E,
        "clustering": clustering,
        "deg_mean": deg_mean,
        "deg_std": deg_std,
        "deg_het": deg_het,
        "density": density,
        "assortativity": assortativity
    }

def main():
    print("=" * 80)
    print(" INICIO DE CAMPAÑA - BLOC-RELOC v6.0 (Extracción de Dataset para Meta-Aprendizaje)")
    print("=" * 80)
    
    datasets = {
        "LesMis": nx.les_miserables_graph(),
        "PowerGrid_Sim": nx.powerlaw_cluster_graph(2000, 3, 0.05, seed=42),
        "Social_Network": nx.newman_watts_strogatz_graph(1500, 10, 0.1, seed=42),
        "Barabasi_Albert": nx.barabasi_albert_graph(2000, 5, seed=42),
        "Erdos_Renyi": nx.erdos_renyi_graph(2000, 0.02, seed=42)
    }
    
    dataset_records = []
    
    for name, G_raw in datasets.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
            
        csr = nx_to_csr(G)
        features = extract_rich_descriptor_vector(G, csr)
        
        np.random.seed(42)
        part_init = get_spectral_partition(G, csr)
        
        cut_frontier, time_frontier, blocks_f = run_refinement_modular(csr, part_init, strategy="frontier")
        cut_cohesion, time_cohesion, blocks_c = run_refinement_modular(csr, part_init, strategy="cohesion")
        
        optimal_strategy = "cohesion" if cut_cohesion < cut_frontier else "frontier"
        if cut_cohesion == cut_frontier:
            optimal_strategy = "frontier"
            
        print(f"\n[INSTANCIA] {name} ➔ [Ground Truth Optimal: {optimal_strategy.upper()}]")
        print(f"  Cut Frontier: {cut_frontier} | Cut Cohesion: {cut_cohesion}")
        print(f"  Features: Clust={features['clustering']:.3f}, HetDeg={features['deg_het']:.3f}, Dens={features['density']:.3f}, Assort={features['assortativity']:.3f}")
        
        record = {
            "Instancia": name,
            **features,
            "Cut_Frontier": cut_frontier,
            "Cut_Cohesion": cut_cohesion,
            "Optimal_Strategy": optimal_strategy
        }
        dataset_records.append(record)

    df_dataset = pd.DataFrame(dataset_records)
    df_dataset.to_csv("meta_training_dataset.csv", index=False)
    
    print("\n" + "=" * 80)
    print(" CAMPAÑA v6.0 GENERACIÓN COMPLETADA. Dataset guardado en: meta_training_dataset.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
