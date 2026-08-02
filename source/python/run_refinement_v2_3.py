# ==============================================================================
# run_refinement_v2_3.py
# BLOC-RELOC v2.3: Selección Inteligente y Ranking de Bloques Candidatos
# ==============================================================================

import os
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

def run_refinement_v2_3(csr: GraphCSR, part_init, max_iters=20, tolerance=0.05, max_escapes=5):
    """
    v2.3: Escanea la frontera, puntúa a los bloques candidatos según su ganancia 
    potencial real y ejecuta exclusivamente el mejor bloque disponible por iteración.
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

        # 2. BLOC-RELOC v2.3: Evaluación y Selección Óptima de Bloques por Ranking
        if not improved and escapes_performed < max_escapes and iteration < max_iters - 1:
            candidate_blocks = []
            
            # Recolectar todos los bloques candidatos válidos en la frontera
            for u in range(N):
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
                    
                    # Tamaño dinámico basado en v2.1 (hasta 8 nodos)
                    dynamic_limit = min(8, max(2, len(same_color_neighbors) // 2))
                    nodes_to_move = [u] + list(same_color_neighbors[:dynamic_limit])
                    block_size = len(nodes_to_move)
                    
                    # Validar restricciones de balance global
                    can_shift = False
                    if curr_p == 0:
                        if n0 - block_size >= min_size and n1 + block_size <= max_size:
                            can_shift = True
                    else:
                        if n1 - block_size >= min_size and n0 + block_size <= max_size:
                            can_shift = True
                            
                    if can_shift:
                        # Evaluar simulación del corte para medir ganancia real
                        for v_node in nodes_to_move:
                            part[v_node] = 1 - curr_p
                        cut_candidate = compute_cut_vectorized(csr.row_indices, csr.indices, part)
                        delta = current_cut - cut_candidate
                        
                        # Revertir temporalmente
                        for v_node in nodes_to_move:
                            part[v_node] = curr_p
                            
                        if delta > 0:
                            candidate_blocks.append({
                                "nodes": nodes_to_move,
                                "curr_p": curr_p,
                                "block_size": block_size,
                                "delta": delta,
                                "cut_candidate": cut_candidate
                            })

            # Seleccionar y aplicar estrictamente el bloque con mayor reducción de corte
            if candidate_blocks:
                candidate_blocks.sort(key=lambda x: x["delta"], reverse=True)
                best_candidate = candidate_blocks[0]
                
                nodes_to_move = best_candidate["nodes"]
                curr_p = best_candidate["curr_p"]
                block_size = best_candidate["block_size"]
                delta = best_candidate["delta"]
                
                for v_node in nodes_to_move:
                    part[v_node] = 1 - curr_p
                    
                if curr_p == 0:
                    n0, n1 = n0 - block_size, n1 + block_size
                else:
                    n1, n0 = n1 - block_size, n0 + block_size
                    
                current_cut = best_candidate["cut_candidate"]
                escapes_performed += 1
                improved = True
                
                blocks_accepted += 1
                block_sizes_accepted.append(block_size)
                block_deltas.append(delta)
                
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
    print(" INICIO DE CAMPAÑA - BLOC-RELOC v2.3 (Selección Inteligente por Ranking)")
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
        print(f"\n[INSTANCIA] Procesando {name} (N={G.number_of_nodes()}, E={G.number_of_edges()})...")
        
        np.random.seed(42)
        part_init = get_spectral_partition(G, csr)
        
        res_v23 = run_refinement_v2_3(csr, part_init, max_iters=20, tolerance=0.05, max_escapes=5)
        
        summary_records.append({
            "Instancia": name,
            "N": G.number_of_nodes(),
            "Cut_v2.3": res_v23["cut"],
            "Time_v2.3": res_v23["time"],
            "Blocks_Accepted": res_v23["blocks_accepted"],
            "Avg_Block_Size": res_v23["avg_block_size"],
            "Avg_Delta": res_v23["avg_block_delta"]
        })
        print(f"  -> v2.3 Cut: {res_v23['cut']} | Tiempo: {res_v23['time']:.4f}s | Bloques Aceptados: {res_v23['blocks_accepted']}")

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("Bloc_Report_v2_3.csv", index=False)
    
    print("\n" + "=" * 80)
    print(" CAMPAÑA v2.3 FINALIZADA. Reporte guardado en: Bloc_Report_v2_3.csv")
    print("=" * 80)

if __name__ == "__main__":
    main()
