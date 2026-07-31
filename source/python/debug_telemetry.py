import networkx as nx
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass

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

def pfv_refine_numba_core(indptr, indices, part, max_iters=10):
    N = len(part)
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
                part[u] = 1 - curr_p
                improved = True
        if not improved:
            break
    return part

def run_pfv_v17_isolated_telemetry(G: nx.Graph):
    csr = nx_to_csr(G)
    adj = nx.adjacency_matrix(G).astype(float)
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D

    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]

    # BLOQUE 1: TopSolver / Bisección Inicial
    part = (fiedler >= np.median(fiedler)).astype(np.int32)
    print(f"[DIAGNOSTICO - TopSolver] S0={np.sum(part == 0)}, S1={np.sum(part == 1)}")

    # BLOQUE 2: Proyección (Auditoría estructural intermedia)
    print(f"[DIAGNOSTICO - Proyeccion] S0={np.sum(part == 0)}, S1={np.sum(part == 1)}")

    # Guard-rail defensivo exigido por el CTO
    if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
        print("    [!] ALERTA: Partición degenerada detectada antes de FM.")

    # BLOQUE 3: Post-FM / Refinamiento Core certificado
    part = pfv_refine_numba_core(csr.indptr, csr.indices, part)
    print(f"[DIAGNOSTICO - Post-FM]     S0={np.sum(part == 0)}, S1={np.sum(part == 1)}")
    return part

if __name__ == "__main__":
    print("=" * 70)
    print(" AUDITORÍA TELEMÉTRICA AISLADA -- ERDŐS-RÉNYI (N=500)")
    print("=" * 70)
    np.random.seed(42)
    G = nx.erdos_renyi_graph(500, 0.05, seed=42)
    G = nx.convert_node_labels_to_integers(G)
    run_pfv_v17_isolated_telemetry(G)
    print("=" * 70)
