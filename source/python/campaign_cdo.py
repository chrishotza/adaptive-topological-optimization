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
def pfv_refine_v171(indptr, indices, part, max_iters=20, tolerance=0.05):
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

@njit(cache=True)
def cdo_refine_with_perturbation(indptr, indices, part, max_iters=20, tolerance=0.05, perturbation_prob=0.02):
    """
    CDO (Controlled Disruption Operator): 
    Cuando el refinador greedy estándar se bloquea (local minimum trap en baja conductancia),
    aplica una perturbación controlada (flipping estocástico/determinista de nodos) 
    respetando los límites de balance para romper la barrera geométrica.
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

    for iteration in range(max_iters):
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
            # Fase CDO: Inyección de perturbación para escapar del mínimo local
            mutated = 0
            for u in range(N):
                if (u * 31 + iteration * 17) % 100 < int(perturbation_prob * 100):
                    curr_p = part[u]
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
                continue # Retoma la búsqueda tras la disrupción
            else:
                break
            
    return part, total_flips

def run_pipeline_with_cdo(G: nx.Graph):
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
        
    # Ejecutamos PFV estándar (tol=5%)
    part_pfv, flips_pfv = pfv_refine_v171(csr.indptr, csr.indices, part.copy(), max_iters=20, tolerance=0.05)
    
    # Ejecutamos PFV + CDO (tol=5% con perturbación activa)
    part_cdo, flips_cdo = cdo_refine_with_perturbation(csr.indptr, csr.indices, part.copy(), max_iters=20, tolerance=0.05, perturbation_prob=0.02)
    
    return part_pfv, flips_pfv, part_cdo, flips_cdo

def warmup_jit():
    dummy_indptr = np.array([0, 1, 2], dtype=np.int32)
    dummy_indices = np.array([1, 0], dtype=np.int32)
    dummy_part = np.array([0, 1], dtype=np.int32)
    pfv_refine_v171(dummy_indptr, dummy_indices, dummy_part, max_iters=2, tolerance=0.05)
    cdo_refine_with_perturbation(dummy_indptr, dummy_indices, dummy_part, max_iters=2, tolerance=0.05, perturbation_prob=0.02)

def main():
    print("=" * 110)
    print(" CAMPAÑA COMPARATIVA: METIS vs PFV (Tol=5%) vs PFV + CDO (Perturbación)")
    print("=" * 110)
    warmup_jit()
    
    # Bancos de prueba (foco especial en Clase B: Social_Network)
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
            
        # Baseline METIS
        part_metis, _ = run_metis_timed(G)
        cut_metis, _ = calculate_metrics(G, part_metis)
        
        # Pipeline PFV & CDO
        part_pfv, flips_pfv, part_cdo, flips_cdo = run_pipeline_with_cdo(G)
        cut_pfv, _ = calculate_metrics(G, part_pfv)
        cut_cdo, _ = calculate_metrics(G, part_cdo)
        
        results.append({
            "Instancia": name,
            "N": G.number_of_nodes(),
            "METIS Corte": cut_metis,
            "PFV (Tol 5%)": cut_pfv,
            "PFV + CDO": cut_cdo,
            "Δ CDO vs PFV": cut_cdo - cut_pfv,
            "Δ CDO vs METIS": cut_cdo - cut_metis
        })
        
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("=" * 110)

if __name__ == "__main__":
    main()
