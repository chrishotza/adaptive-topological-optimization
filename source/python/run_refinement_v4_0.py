# ==============================================================================
# run_refinement_v4_0.py
# BLOC-RELOC v4.0: Arquitectura Modular con Dispatcher Topológico Automático
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
    """
    v4.0: Refinador modular que soporta múltiples estrategias de generación de bloques:
    - 'frontier': Basado en vecinos de frontera dinámicos (v2.1) -> Ideal para Social y BA
    - 'cohesion': Basado en vecinos compartidos y cohesión estructural (v3.0) -> Ideal para PowerGrid
    """
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
        
        # 1. Refinamiento local estándar (movimientos individuales)
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

        # 2. Módulo de Bloques Seleccionado por el Dispatcher
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
                    blocks_evaluated += 1
                    same_color_mask = (neighbor_parts == curr_p)
                    same_color_neighbors = neighbors[same_color_mask]
                    
                    if len(same_color_neighbors) == 0:
                        continue
                    
                    # Generación según la estrategia asignada
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
                    else:  # Estrategia 'frontier' (v2.1)
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
                            delta = current_cut - cut_candidate
                            if curr_p == 0:
                                n0, n1 = n0 - block_size, n1 + block_size
                            else:
                                n1, n0 = n1 - block_size, n0 + block_size
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
    return {
        "cut": int(best_cut),
        "time": elapsed,
        "blocks_evaluated": blocks_evaluated,
        "blocks_accepted": blocks_accepted,
        "avg_block_size": np.mean(block_sizes_accepted) if block_sizes_accepted else 0.0,
        "avg_block_delta": np.mean(block_deltas) if block_deltas else 0.0
    }

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
    print(" INICIO DE CAMPAÑA - BLOC-RELOC v4.0 (Arquitectura Modular & Dispatcher)")
    print("=" * 80)
    
    datasets = {
        "LesMis": nx.les_miserables_graph(),
        "PowerGrid_Sim": nx.powerlaw_cluster_graph(2000, 3, 0.05, seed=42),
        "Social_Network": nx.newman_watts_strogatz_graph(1500, 10, 0.1, seed=42),
        "Barabasi_Albert": nx.barabasi_albert_graph(2000, 5, seed=42),
        "Erdos_Renyi": nx.erdos_renyi_graph(2000, 0.02, seed=42)
    }
    
    summary_records = []
    
    for name, G_raw in datasets.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
            
        csr = nx_to_csr(G)
        
        # Dispatcher Topológico Automático: Asigna la estrategia según el perfil estructural
        avg_clustering = nx.average_clustering(G)
        if avg_clustering > 0.15:  # Perfil cuasi-regular / alta agrupación local
            strategy = "cohesion"
        else:                      # Perfil libre de escala / comunitario
            strategy = "frontier"
            
        print(f"\n[INSTANCIA] Procesando {name} (N={G.number_of_nodes()}, Clust={avg_clustering:.3f}) ➔ [Dispatcher: {strategy.upper()}]...")
        
        np.random.seed(42)
        part_init = get_spectral_partition(G, csr)
        
        res_v40 = run_refinement_modular(csr, part_init, strategy=strategy, max_iters=20, tolerance=0.05, max_escapes=5)
        
        summary_records.append({
            "Instancia": name,
            "N": G.number_of_nodes(),
            "Strategy": strategy,
            "Cut_v4.0": res_v40["cut"],
            "Time_v4.0": res_v40["time"],
            "Blocks_Accepted": res_v40["blocks_accepted"]
        })
        print(f"  -> v4.0 Cut: {res_v40['cut']} | Tiempo: {res_v40['time']:.4f}s | Bloques Aceptados: {res_v40['blocks_accepted']}")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("Bloc_Report_v4_0.csv", index=False)
    
    print("\n" + "=" * 80)
    print(" CAMPAÑA v4.0 FINALIZADA. Reporte guardado en: Bloc_Report_v4_0.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
