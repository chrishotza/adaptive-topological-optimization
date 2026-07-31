import os
import sys
import time
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import networkx as nx

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')

# =====================================================================
# 1. GENERACIÓN PARAMÉTRICA DE GRAFOS (DIVERSIFICACIÓN ESTRUCTURAL)
# =====================================================================

def generate_graph_corpus(num_graphs: int = 1200, seed: int = 2026) -> List[Tuple[str, nx.Graph]]:
    rng = np.random.default_rng(seed)
    corpus = []
    families = [
        'erdos_renyi', 
        'barabasi_albert', 
        'watts_strogatz', 
        'stochastic_block', 
        'random_geometric', 
        'grid_2d', 
        'random_regular'
    ]
    
    per_family = num_graphs // len(families)
    
    for family in families:
        for i in range(per_family):
            n = rng.integers(60, 450)
            try:
                if family == 'erdos_renyi':
                    p = rng.uniform(0.02, 0.12)
                    G = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(1e6)))
                    
                elif family == 'barabasi_albert':
                    m = rng.integers(2, min(8, n - 1))
                    G = nx.barabasi_albert_graph(n=n, m=m, seed=int(rng.integers(1e6)))
                    
                elif family == 'watts_strogatz':
                    k = rng.integers(4, min(10, n - 1))
                    if k % 2 != 0: k += 1
                    p = rng.uniform(0.05, 0.25)
                    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=int(rng.integers(1e6)))
                    
                elif family == 'stochastic_block':
                    num_blocks = rng.integers(2, 5)
                    sizes = list(rng.multinomial(n, [1.0/num_blocks]*num_blocks))
                    sizes = [s for s in sizes if s > 0]
                    p_in = rng.uniform(0.15, 0.35)
                    p_out = rng.uniform(0.005, 0.04)
                    p_matrix = np.full((len(sizes), len(sizes)), p_out)
                    np.fill_diagonal(p_matrix, p_in)
                    G = nx.stochastic_block_model(sizes, p_matrix, seed=int(rng.integers(1e6)))
                    
                elif family == 'random_geometric':
                    radius = rng.uniform(0.12, 0.28)
                    G = nx.random_geometric_graph(n=n, radius=radius, seed=int(rng.integers(1e6)))
                    
                elif family == 'grid_2d':
                    dim1 = rng.integers(7, 20)
                    dim2 = rng.integers(7, 20)
                    G = nx.grid_2d_graph(dim1, dim2)
                    G = nx.convert_node_labels_to_integers(G)
                    
                elif family == 'random_regular':
                    d = rng.integers(3, 6)
                    if (n * d) % 2 != 0: n += 1
                    G = nx.random_regular_graph(d=d, n=n, seed=int(rng.integers(1e6)))

                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                    G = nx.convert_node_labels_to_integers(G)
                    
                if G.number_of_nodes() >= 40 and G.number_of_edges() >= 40:
                    corpus.append((family, G))
            except Exception:
                continue

    return corpus


# =====================================================================
# 2. DESCRIPTORES TOPOLÓGICOS (CARACTERÍSTICAS V14)
# =====================================================================

def compute_topological_descriptors(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])
    deg_mean = np.mean(degrees)
    deg_std = np.std(degrees)
    
    deg_skew = np.mean(((degrees - deg_mean) / deg_std) ** 3) if deg_std > 1e-9 else 0.0
    deg_kurt = np.mean(((degrees - deg_mean) / deg_std) ** 4) - 3.0 if deg_std > 1e-9 else 0.0
        
    deg_het = deg_std / deg_mean if deg_mean > 0 else 0.0
    density = 2.0 * e / (n * (n - 1)) if n > 1 else 0.0
    clustering = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    if np.isnan(assortativity): assortativity = 0.0
        
    L = nx.laplacian_matrix(G).astype(float)
    try:
        vals = spla.eigsh(L, k=min(3, n - 1), which='SM', return_eigenvectors=False)
        vals = np.sort(vals)
        fiedler_value = float(vals[1]) if len(vals) > 1 else 0.0
        spectral_gap = float(vals[2] - vals[1]) if len(vals) > 2 else 0.0
    except Exception:
        fiedler_value = 0.0
        spectral_gap = 0.0
        
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        num_communities = len(communities)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        num_communities = 1
        modularity = 0.0

    return {
        'N': float(n), 'E': float(e), 'density': float(density),
        'deg_mean': float(deg_mean), 'deg_std': float(deg_std),
        'deg_skew': float(deg_skew), 'deg_kurt': float(deg_kurt),
        'deg_het': float(deg_het), 'clustering': float(clustering),
        'assortativity': float(assortativity), 'spectral_gap': float(spectral_gap),
        'fiedler_value': float(fiedler_value), 'num_communities': float(num_communities),
        'modularity': float(modularity)
    }


# =====================================================================
# 3. EJECUCIÓN REAL DE REFINADORES Y FUNCIÓN OBJETIVO ADIMENSIONAL
# =====================================================================

def compute_conductance(G: nx.Graph, part: Dict[int, int], cut_val: int) -> float:
    vol_0 = sum(G.degree(node) for node, p in part.items() if p == 0)
    vol_1 = sum(G.degree(node) for node, p in part.items() if p == 1)
    min_vol = min(vol_0, vol_1)
    return cut_val / (min_vol + 1e-5)

def run_frontier_refinement(G: nx.Graph, initial_part: Dict[int, int], max_iter: int = 30) -> Dict[int, int]:
    part = initial_part.copy()
    for _ in range(max_iter):
        moved = False
        cut_nodes = [u for u, v in G.edges() if part[u] != part[v]]
        frontier = list(set(cut_nodes))
        np.random.shuffle(frontier)
        
        for u in frontier:
            current_p = part[u]
            other_p = 1 - current_p
            
            # Ganancia en el corte al mover u
            internal_neighbors = sum(1 for v in G.neighbors(u) if part[v] == current_p)
            external_neighbors = sum(1 for v in G.neighbors(u) if part[v] == other_p)
            gain = external_neighbors - internal_neighbors
            
            if gain > 0:
                part[u] = other_p
                moved = True
                break
        if not moved:
            break
    return part

def run_cohesion_refinement(G: nx.Graph, initial_part: Dict[int, int], max_iter: int = 30) -> Dict[int, int]:
    part = initial_part.copy()
    for _ in range(max_iter):
        moved = False
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        
        for u in nodes:
            current_p = part[u]
            other_p = 1 - current_p
            
            # Criterio Cohesion: maximizar densidad intra-cluster ponderada por el grado
            vol_target = sum(G.degree(v) for v in G.neighbors(u) if part[v] == other_p)
            vol_source = sum(G.degree(v) for v in G.neighbors(u) if part[v] == current_p)
            
            if vol_target > vol_source + 1.0:
                part[u] = other_p
                moved = True
                break
        if not moved:
            break
    return part

def evaluate_objective_function(G: nx.Graph) -> Tuple[str, Dict[str, float]]:
    # Partición Bipartita Inicial (Baseline)
    try:
        communities = nx.community.kernighan_lin_bisection(G)
        part_0 = {node: 0 for node in communities[0]}
        part_0.update({node: 1 for node in communities[1]})
    except Exception:
        nodes = list(G.nodes())
        part_0 = {nodes[i]: 0 if i < len(nodes)//2 else 1 for i in range(len(nodes))}

    cut_0 = sum(1 for u, v in G.edges() if part_0[u] != part_0[v])
    phi_0 = compute_conductance(G, part_0, cut_0)

    # 1. Aplicar Refinador Frontier
    part_f = run_frontier_refinement(G, part_0)
    cut_f = sum(1 for u, v in G.edges() if part_f[u] != part_f[v])
    phi_f = compute_conductance(G, part_f, cut_f)
    
    # 2. Aplicar Refinador Cohesion
    part_c = run_cohesion_refinement(G, part_0)
    cut_c = sum(1 for u, v in G.edges() if part_c[u] != part_c[v])
    phi_c = compute_conductance(G, part_c, cut_c)

    # 3. Métricas Adimensionales Normalizadas
    delta_R_cut_f = (cut_0 - cut_f) / (cut_0 + 1e-5)
    delta_phi_f = (phi_0 - phi_f) / (phi_0 + 1e-5)
    J_frontier = 0.5 * delta_R_cut_f + 0.5 * delta_phi_f

    delta_R_cut_c = (cut_0 - cut_c) / (cut_0 + 1e-5)
    delta_phi_c = (phi_0 - phi_c) / (phi_0 + 1e-5)
    J_cohesion = 0.5 * delta_R_cut_c + 0.5 * delta_phi_c

    # 4. Asignación Estricta con Margen \tau = 0.015
    tau = 0.015
    diff = J_cohesion - J_frontier
    
    if abs(diff) <= tau:
        label = 'tie'
    elif diff > tau:
        label = 'cohesion'
    else:
        label = 'frontier'

    return label, {
        'J_frontier': J_frontier,
        'J_cohesion': J_cohesion,
        'diff': diff,
        'cut_0': cut_0,
        'cut_f': cut_f,
        'cut_c': cut_c
    }


# =====================================================================
# 4. EXPERIMENTO v14 Y VALIDACIÓN DE HIPÓTESIS
# =====================================================================

def run_phase1_v14_experiment():
    print("======================================================================")
    print("FASE 1 (v14): REFORMULACIÓN FORMAL DE LA FUNCIÓN OBJETIVO Y BENCHMARK")
    print("======================================================================\n")
    
    num_graphs = 1200
    print(f"[*] Generando corpus diversificado (N = {num_graphs})...")
    start = time.time()
    corpus = generate_graph_corpus(num_graphs=num_graphs, seed=2026)
    print(f"[✓] {len(corpus)} grafos generados en {time.time() - start:.2f}s.\n")

    records = []
    print("[*] Ejecutando simulaciones de refinamiento y extrayendo métricas...")
    for idx, (family, G) in enumerate(corpus):
        descriptors = compute_topological_descriptors(G)
        label, metrics = evaluate_objective_function(G)
        
        row = {'graph_id': idx, 'family': family, 'Optimal_Operator': label}
        row.update(descriptors)
        row.update(metrics)
        records.append(row)
        
    df = pd.DataFrame(records)
    df.to_csv('meta_dataset_v14_adimensional.csv', index=False)
    
    print("\n================ DISTRIBUCIÓN REAL DE CLASES (v14) ================")
    print(df['Optimal_Operator'].value_counts())
    print("===================================================================\n")
    
    df_filtered = df[df['Optimal_Operator'] != 'tie'].copy()
    print(f"[*] Muestras no degeneradas para entrenamiento: {len(df_filtered)}")
    
    if len(df_filtered['Optimal_Operator'].unique()) < 2:
        print("[ERR] El dataset sigue monolítico. Se requiere revisar el margen tau.")
        sys.exit(1)

    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_skew', 'deg_kurt',
        'deg_het', 'clustering', 'assortativity', 'spectral_gap', 
        'fiedler_value', 'num_communities', 'modularity'
    ]
    
    X = df_filtered[feature_cols].values
    y = (df_filtered['Optimal_Operator'] == 'cohesion').astype(int).values
    
    seeds = [42, 100, 2026]
    f1_scores, accuracies = [], []
    
    print("\n================ EVALUACIÓN MULTI-SEMILLA OOF (v14) ================")
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            clf = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3, random_state=seed
            )
            clf.fit(X_train, y_train)
            oof_preds[val_idx] = clf.predict(X_val)
            
        acc = accuracy_score(y, oof_preds)
        f1_macro = f1_score(y, oof_preds, average='macro')
        accuracies.append(acc)
        f1_scores.append(f1_macro)
        print(f"Semilla {seed:4d} | Accuracy OOF: {acc:.4f} | F1 Macro OOF: {f1_macro:.4f}")

    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)
    mean_f1, std_f1 = np.mean(f1_scores), np.std(f1_scores)

    print("\n---------------- ESTADÍSTICAS ENSEMBLE MULTI-SEMILLA ----------------")
    print(f"Accuracy OOF Promedio: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1 Macro OOF Promedio: {mean_f1:.4f} ± {std_f1:.4f}")
    
    print("\nMatriz de Confusión Out-of-Fold (Última Semilla):")
    print(confusion_matrix(y, oof_preds))
    
    print("\nReporte de Clasificación OOF (Última Semilla):")
    print(classification_report(y, oof_preds, target_names=['Frontier', 'Cohesion'], digits=4))
    
    print("================ CONCLUSIÓN EXPERIMENTAL HIPÓTESIS H1/v14 ================")
    if mean_f1 >= 0.65 and std_f1 < 0.04:
        print("[ACEPTADA] La función objetivo adimensional eliminó el sesgo y permite discriminación real.")
        print("Siguiente cuello de botella: Selección rigurosa de descriptores no redundantes (Fase 2).")
    else:
        print("[RECHAZADA] Se requiere mayor capacidad de representación en las características.")
    print("==========================================================================\n")

if __name__ == '__main__':
    run_phase1_v14_experiment()
