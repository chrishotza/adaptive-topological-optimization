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

def generate_graph_corpus(num_graphs: int = 1200, seed: int = 42) -> List[Tuple[str, nx.Graph]]:
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
            n = rng.integers(50, 500)
            
            try:
                if family == 'erdos_renyi':
                    p = rng.uniform(0.01, 0.1)
                    G = nx.erdos_renyi_graph(n=n, p=p, seed=int(rng.integers(1e6)))
                    
                elif family == 'barabasi_albert':
                    m = rng.integers(1, min(10, n - 1))
                    G = nx.barabasi_albert_graph(n=n, m=m, seed=int(rng.integers(1e6)))
                    
                elif family == 'watts_strogatz':
                    k = rng.integers(2, min(12, n - 1))
                    if k % 2 != 0:
                        k += 1
                    p = rng.uniform(0.05, 0.3)
                    G = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=int(rng.integers(1e6)))
                    
                elif family == 'stochastic_block':
                    num_blocks = rng.integers(2, 6)
                    sizes = list(rng.multinomial(n, [1.0/num_blocks]*num_blocks))
                    sizes = [s for s in sizes if s > 0]
                    num_blocks = len(sizes)
                    
                    p_in = rng.uniform(0.1, 0.3)
                    p_out = rng.uniform(0.001, 0.05)
                    p_matrix = np.full((num_blocks, num_blocks), p_out)
                    np.fill_diagonal(p_matrix, p_in)
                    
                    G = nx.stochastic_block_model(sizes, p_matrix, seed=int(rng.integers(1e6)))
                    
                elif family == 'random_geometric':
                    radius = rng.uniform(0.1, 0.3)
                    G = nx.random_geometric_graph(n=n, radius=radius, seed=int(rng.integers(1e6)))
                    
                elif family == 'grid_2d':
                    dim1 = rng.integers(6, 25)
                    dim2 = rng.integers(6, 25)
                    G = nx.grid_2d_graph(dim1, dim2)
                    G = nx.convert_node_labels_to_integers(G)
                    
                elif family == 'random_regular':
                    d = rng.integers(2, 6)
                    if (n * d) % 2 != 0:
                        n += 1
                    G = nx.random_regular_graph(d=d, n=n, seed=int(rng.integers(1e6)))

                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                    G = nx.convert_node_labels_to_integers(G)
                    
                if G.number_of_nodes() >= 30 and G.number_of_edges() >= 30:
                    corpus.append((family, G))
                    
            except Exception:
                continue

    return corpus


# =====================================================================
# 2. EXTRACCIÓN DE DESCRIPTORES ESTRUCTURALES
# =====================================================================

def compute_topological_descriptors(G: nx.Graph) -> Dict[str, float]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    
    degrees = np.array([d for _, d in G.degree()])
    deg_mean = np.mean(degrees)
    deg_std = np.std(degrees)
    
    if deg_std > 1e-9:
        deg_skew = np.mean(((degrees - deg_mean) / deg_std) ** 3)
        deg_kurt = np.mean(((degrees - deg_mean) / deg_std) ** 4) - 3.0
    else:
        deg_skew = 0.0
        deg_kurt = 0.0
        
    deg_het = deg_std / deg_mean if deg_mean > 0 else 0.0
    density = 2.0 * e / (n * (n - 1)) if n > 1 else 0.0
    clustering = nx.average_clustering(G)
    assortativity = nx.degree_assortativity_coefficient(G)
    if np.isnan(assortativity):
        assortativity = 0.0
        
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
        'N': float(n),
        'E': float(e),
        'density': float(density),
        'deg_mean': float(deg_mean),
        'deg_std': float(deg_std),
        'deg_skew': float(deg_skew),
        'deg_kurt': float(deg_kurt),
        'deg_het': float(deg_het),
        'clustering': float(clustering),
        'assortativity': float(assortativity),
        'spectral_gap': float(spectral_gap),
        'fiedler_value': float(fiedler_value),
        'num_communities': float(num_communities),
        'modularity': float(modularity)
    }


# =====================================================================
# 3. EVALUACIÓN Y ETIQUETADO DE OPERADORES
# =====================================================================

def evaluate_operators_on_graph(G: nx.Graph) -> Tuple[str, Dict[str, float]]:
    try:
        communities = nx.community.kernighan_lin_bisection(G)
        part = {node: 0 for node in communities[0]}
        part.update({node: 1 for node in communities[1]})
    except Exception:
        nodes = list(G.nodes())
        part = {nodes[i]: 0 if i < len(nodes)//2 else 1 for i in range(len(nodes))}

    cut_edges = [(u, v) for u, v in G.edges() if part[u] != part[v]]
    frontier_nodes = set([u for u, v in cut_edges] + [v for u, v in cut_edges])
    
    gain_frontier = sum(G.degree(node) for node in frontier_nodes) / (len(frontier_nodes) + 1e-5)
    internal_edges = G.number_of_edges() - len(cut_edges)
    gain_cohesion = internal_edges / (G.number_of_nodes() + 1e-5)
    
    diff = (gain_cohesion - gain_frontier) / (max(gain_cohesion, gain_frontier) + 1e-5)
    
    if abs(diff) < 0.05:
        label = 'tie'
    elif diff > 0:
        label = 'cohesion'
    else:
        label = 'frontier'
        
    metrics = {
        'gain_cohesion': gain_cohesion,
        'gain_frontier': gain_frontier,
        'diff': diff
    }
    return label, metrics


# =====================================================================
# 4. PIPELINE PRINCIPAL Y EVALUACIÓN
# =====================================================================

def run_phase1_experiment():
    print("======================================================================")
    print("FASE 1: ROBUSTEZ EXPERIMENTAL Y EVALUACIÓN DE HIPÓTESIS H1")
    print("======================================================================\n")
    
    num_graphs = 1200
    print(f"[*] Generando corpus paramétrico diversificado (N = {num_graphs})...")
    start_time = time.time()
    corpus = generate_graph_corpus(num_graphs=num_graphs, seed=2026)
    print(f"[✓] Corpus generado: {len(corpus)} grafos válidos en {time.time() - start_time:.2f}s.\n")
    
    records = []
    print("[*] Extrayendo descriptores topológicos y etiquetas operacionales...")
    for idx, (family, G) in enumerate(corpus):
        descriptors = compute_topological_descriptors(G)
        label, op_metrics = evaluate_operators_on_graph(G)
        
        row = {'graph_id': idx, 'family': family, 'Optimal_Operator': label}
        row.update(descriptors)
        row.update(op_metrics)
        records.append(row)
        
    df = pd.DataFrame(records)
    dataset_file = 'meta_statistical_dataset_v13.csv'
    df.to_csv(dataset_file, index=False)
    print(f"[✓] Dataset guardado exitosamente en '{dataset_file}'.\n")
    
    print("================ DISTRIBUCIÓN DE CLASES (v13) ================")
    print(df['Optimal_Operator'].value_counts())
    print("==============================================================")
    
    df_filtered = df[df['Optimal_Operator'] != 'tie'].copy()
    print(f"\n[*] Muestras post-filtrado de empates ('tie'): {len(df_filtered)}")
    
    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_skew', 'deg_kurt',
        'deg_het', 'clustering', 'assortativity', 'spectral_gap', 
        'fiedler_value', 'num_communities', 'modularity'
    ]
    
    X = df_filtered[feature_cols].values
    y = (df_filtered['Optimal_Operator'] == 'cohesion').astype(int).values
    
    seeds = [42, 100, 2026]
    f1_scores = []
    accuracies = []
    
    print("\n================ EVALUACIÓN ROBUSTA MULTI-SEMILLA (OOF) ================")
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=seed
            )
            clf.fit(X_train, y_train)
            oof_preds[val_idx] = clf.predict(X_val)
            
        acc = accuracy_score(y, oof_preds)
        f1_macro = f1_score(y, oof_preds, average='macro')
        accuracies.append(acc)
        f1_scores.append(f1_macro)
        
        print(f"Semilla {seed:4d} | Accuracy OOF: {acc:.4f} | F1 Macro OOF: {f1_macro:.4f}")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print("\n---------------- ESTADÍSTICAS ENSEMBLE MULTI-SEMILLA ----------------")
    print(f"Accuracy OOF Promedio: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"F1 Macro OOF Promedio: {mean_f1:.4f} ± {std_f1:.4f}")
    
    print("\nMatriz de Confusión Out-of-Fold (Última Semilla):")
    cm = confusion_matrix(y, oof_preds)
    print(cm)
    
    print("\nReporte de Clasificación OOF (Última Semilla):")
    target_names = ['Frontier', 'Cohesion']
    print(classification_report(y, oof_preds, target_names=target_names, digits=4))
    
    print("================ CONCLUSIÓN EXPERIMENTAL HIPÓTESIS H1 ================")
    if mean_f1 > 0.70 and std_f1 < 0.03:
        print("[ACEPTADA] La escala del corpus eliminó el sesgo de varianza y estabilizó la métrica.")
        print("Siguiente cuello de botella: Discriminación redundante entre descriptores topológicos (Fase 2).")
    else:
        print("[RECHAZADA / INCOMPLETA] El volumen no resolvió el sesgo por completo.")
        print("Siguiente cuello de botella: Desbalance persistente o representatividad de características.")
    print("======================================================================\n")

if __name__ == '__main__':
    run_phase1_experiment()
