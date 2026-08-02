import os
import warnings
import networkx as nx
import numpy as np
import scipy.sparse as sp
from numba import njit

warnings.filterwarnings("ignore")

@njit
def audit_refiner_funnel(indptr, indices, part, max_iters=20, tolerance=0.0):
    N = len(part)
    min_size = int((0.5 - tolerance) * N)
    max_size = int((0.5 + tolerance) * N)
    
    n0 = 0
    for i in range(N):
        if part[i] == 0:
            n0 += 1
    n1 = N - n0

    # Historial por iteración: [candidates, rej_gain, rej_balance, accepted, sum_gains, max_gain]
    history = np.zeros((max_iters, 6), dtype=np.float64)
    actual_iters = 0

    for iteration in range(max_iters):
        improved = False
        candidates_evaluated = 0
        rejected_balance = 0
        rejected_gain = 0
        accepted_moves = 0
        sum_gains = 0.0
        max_gain = -999.0

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

            candidates_evaluated += 1
            gain = float(int_deg - ext_deg)
            sum_gains += gain
            if gain > max_gain:
                max_gain = gain

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
                    accepted_moves += 1
                    improved = True
                else:
                    rejected_balance += 1
            else:
                rejected_gain += 1

        history[iteration, 0] = candidates_evaluated
        history[iteration, 1] = rejected_gain
        history[iteration, 2] = rejected_balance
        history[iteration, 3] = accepted_moves
        history[iteration, 4] = sum_gains
        history[iteration, 5] = max_gain
        actual_iters += 1

        if not improved:
            break
            
    return part, history[:actual_iters]

def run_audit():
    print("=" * 70)
    print(" EJECUTANDO AUDITORÍA INTERNA DE REFINAMIENTO (RefinerScope)")
    print("=" * 70)
    
    G_raw = nx.barabasi_albert_graph(500, 5, seed=1000)
    G = nx.convert_node_labels_to_integers(G_raw)
    
    adj = nx.adjacency_matrix(G).astype(float)
    csr = adj.tocsr()
    deg = np.array(adj.sum(axis=1)).flatten()
    D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
    L = sp.eye(G.number_of_nodes()) - D @ adj @ D
    vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
    fiedler = vecs[:, 1]
    part = (fiedler >= np.median(fiedler)).astype(np.int32)
    
    if np.sum(part == 0) == 0 or np.sum(part == 1) == 0:
        part = np.zeros(G.number_of_nodes(), dtype=np.int32)
        part[G.number_of_nodes() // 2:] = 1

    print(f"[REFINERSCOPE INIT] N={len(part)} | Límites Balance: 0% tolerancia")
    
    _, history = audit_refiner_funnel(csr.indptr, csr.indices, part, max_iters=5, tolerance=0.0)
    
    for idx, row in enumerate(history):
        cands, rej_g, rej_b, accepted, sum_g, max_g = row
        avg_gain = sum_g / cands if cands > 0 else 0.0
        print(f"--- Iteración {idx + 1} ---")
        print(f"  Candidatos evaluados ........ {int(cands)}")
        print(f"  Rechazados por Ganancia ..... {int(rej_g)} (gain >= 0)")
        print(f"  Rechazados por Balance ...... {int(rej_b)} (violan límites 0%)")
        print(f"  Movimientos Aceptados ....... {int(accepted)}")
        print(f"  Ganancia Promedio ........... {avg_gain:.4f}")
        print(f"  Ganancia Máxima ............. {max_g:.4f}")

if __name__ == "__main__":
    run_audit()
