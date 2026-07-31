import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla
import scipy.stats as stats
import networkx as nx

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')

# =====================================================================
# 1. CARGA O GENERACIÓN DE DATASET v14 CON FUNCIÓN OBJETIVO ADIMENSIONAL
# =====================================================================

def generate_graph_corpus(num_graphs: int = 1200, seed: int = 2026) -> List[Tuple[str, nx.Graph]]:
    rng = np.random.default_rng(seed)
    corpus = []
    families = [
        'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 
        'stochastic_block', 'random_geometric', 'grid_2d', 'random_regular'
    ]
    per_family = num_graphs // len(families)
    
    for family in families:
        for _ in range(per_family):
            n = rng.integers(60, 450)
            try:
                if family == 'erdos_renyi':
                    G = nx.erdos_renyi_graph(n=n, p=rng.uniform(0.02, 0.12), seed=int(rng.integers(1e6)))
                elif family == 'barabasi_albert':
                    G = nx.barabasi_albert_graph(n=n, m=rng.integers(2, min(8, n - 1)), seed=int(rng.integers(1e6)))
                elif family == 'watts_strogatz':
                    k = rng.integers(4, min(10, n - 1))
                    if k % 2 != 0: k += 1
                    G = nx.watts_strogatz_graph(n=n, k=k, p=rng.uniform(0.05, 0.25), seed=int(rng.integers(1e6)))
                elif family == 'stochastic_block':
                    num_blocks = rng.integers(2, 5)
                    sizes = list(rng.multinomial(n, [1.0/num_blocks]*num_blocks))
                    sizes = [s for s in sizes if s > 0]
                    p_matrix = np.full((len(sizes), len(sizes)), rng.uniform(0.005, 0.04))
                    np.fill_diagonal(p_matrix, rng.uniform(0.15, 0.35))
                    G = nx.stochastic_block_model(sizes, p_matrix, seed=int(rng.integers(1e6)))
                elif family == 'random_geometric':
                    G = nx.random_geometric_graph(n=n, radius=rng.uniform(0.12, 0.28), seed=int(rng.integers(1e6)))
                elif family == 'grid_2d':
                    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rng.integers(7, 20), rng.integers(7, 20)))
                elif family == 'random_regular':
                    d = rng.integers(3, 6)
                    if (n * d) % 2 != 0: n += 1
                    G = nx.random_regular_graph(d=d, n=n, seed=int(rng.integers(1e6)))

                if not nx.is_connected(G):
                    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                    G = nx.convert_node_labels_to_integers(G)
                    
                if G.number_of_nodes() >= 40:
                    corpus.append((family, G))
            except Exception:
                continue
    return corpus

def compute_topological_descriptors(G: nx.Graph) -> Dict[str, float]:
    n, e = G.number_of_nodes(), G.number_of_edges()
    degrees = np.array([d for _, d in G.degree()])
    deg_mean, deg_std = np.mean(degrees), np.std(degrees)
    
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
        fiedler_value, spectral_gap = 0.0, 0.0
        
    try:
        communities = list(nx.community.greedy_modularity_communities(G))
        num_communities = len(communities)
        modularity = nx.community.modularity(G, communities)
    except Exception:
        num_communities, modularity = 1, 0.0

    return {
        'N': float(n), 'E': float(e), 'density': float(density),
        'deg_mean': float(deg_mean), 'deg_std': float(deg_std),
        'deg_skew': float(deg_skew), 'deg_kurt': float(deg_kurt),
        'deg_het': float(deg_het), 'clustering': float(clustering),
        'assortativity': float(assortativity), 'spectral_gap': float(spectral_gap),
        'fiedler_value': float(fiedler_value), 'num_communities': float(num_communities),
        'modularity': float(modularity)
    }

def compute_conductance(G: nx.Graph, part: Dict[int, int], cut_val: int) -> float:
    vol_0 = sum(G.degree(node) for node, p in part.items() if p == 0)
    vol_1 = sum(G.degree(node) for node, p in part.items() if p == 1)
    return cut_val / (min(vol_0, vol_1) + 1e-5)

def run_frontier_refinement(G: nx.Graph, initial_part: Dict[int, int], max_iter: int = 30) -> Dict[int, int]:
    part = initial_part.copy()
    for _ in range(max_iter):
        moved = False
        frontier = list(set([u for u, v in G.edges() if part[u] != part[v]]))
        np.random.shuffle(frontier)
        for u in frontier:
            cp, op = part[u], 1 - part[u]
            gain = sum(1 for v in G.neighbors(u) if part[v] == op) - sum(1 for v in G.neighbors(u) if part[v] == cp)
            if gain > 0:
                part[u] = op
                moved = True
                break
        if not moved: break
    return part

def run_cohesion_refinement(G: nx.Graph, initial_part: Dict[int, int], max_iter: int = 30) -> Dict[int, int]:
    part = initial_part.copy()
    for _ in range(max_iter):
        moved = False
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        for u in nodes:
            cp, op = part[u], 1 - part[u]
            vol_target = sum(G.degree(v) for v in G.neighbors(u) if part[v] == op)
            vol_source = sum(G.degree(v) for v in G.neighbors(u) if part[v] == cp)
            if vol_target > vol_source + 1.0:
                part[u] = op
                moved = True
                break
        if not moved: break
    return part

def evaluate_objective_function(G: nx.Graph) -> Tuple[str, Dict[str, float]]:
    try:
        comm = nx.community.kernighan_lin_bisection(G)
        part_0 = {node: 0 for node in comm[0]}
        part_0.update({node: 1 for node in comm[1]})
    except Exception:
        nodes = list(G.nodes())
        part_0 = {nodes[i]: 0 if i < len(nodes)//2 else 1 for i in range(len(nodes))}

    cut_0 = sum(1 for u, v in G.edges() if part_0[u] != part_0[v])
    phi_0 = compute_conductance(G, part_0, cut_0)

    part_f = run_frontier_refinement(G, part_0)
    cut_f = sum(1 for u, v in G.edges() if part_f[u] != part_f[v])
    phi_f = compute_conductance(G, part_f, cut_f)
    
    part_c = run_cohesion_refinement(G, part_0)
    cut_c = sum(1 for u, v in G.edges() if part_c[u] != part_c[v])
    phi_c = compute_conductance(G, part_c, cut_c)

    delta_R_cut_f = (cut_0 - cut_f) / (cut_0 + 1e-5)
    delta_phi_f = (phi_0 - phi_f) / (phi_0 + 1e-5)
    J_frontier = 0.5 * delta_R_cut_f + 0.5 * delta_phi_f

    delta_R_cut_c = (cut_0 - cut_c) / (cut_0 + 1e-5)
    delta_phi_c = (phi_0 - phi_c) / (phi_0 + 1e-5)
    J_cohesion = 0.5 * delta_R_cut_c + 0.5 * delta_phi_c

    tau = 0.015
    diff = J_cohesion - J_frontier
    
    if abs(diff) <= tau:
        label = 'tie'
    elif diff > tau:
        label = 'cohesion'
    else:
        label = 'frontier'

    return label, {'J_frontier': J_frontier, 'J_cohesion': J_cohesion, 'diff': diff, 'abs_diff': abs(diff)}


# =====================================================================
# 2. PIPELINE DE CARACTERIZACIÓN ESTADÍSTICA DE EMPATES (FASE 2)
# =====================================================================

def run_phase2_tie_characterization():
    print("======================================================================")
    print("FASE 2: CARACTERIZACIÓN ESTADÍSTICA DEL RÉGIMEN DE EMPATES (TIE)")
    print("======================================================================\n")

    dataset_path = 'meta_dataset_v14_adimensional.csv'
    if os.path.exists(dataset_path):
        print(f"[*] Cargando dataset existente '{dataset_path}'...")
        df = pd.read_csv(dataset_path)
    else:
        print("[*] Generando dataset v14 desde cero...")
        corpus = generate_graph_corpus(num_graphs=1200, seed=2026)
        records = []
        for idx, (family, G) in enumerate(corpus):
            desc = compute_topological_descriptors(G)
            label, metrics = evaluate_objective_function(G)
            row = {'graph_id': idx, 'family': family, 'Optimal_Operator': label}
            row.update(desc)
            row.update(metrics)
            records.append(row)
        df = pd.DataFrame(records)
        df.to_csv(dataset_path, index=False)

    df['is_tie'] = (df['Optimal_Operator'] == 'tie').astype(int)

    # ---------------------------------------------------------------------
    # MÓDULO 1: DISTRIBUCIÓN POR FAMILIA
    # ---------------------------------------------------------------------
    print("================ 1. INCIDENCIA DE EMPATES POR FAMILIA DE GRAFOS ================")
    fam_stats = df.groupby('family')['is_tie'].agg(
        Total='count',
        Empates='sum',
        Tasa_Empate=lambda x: np.mean(x) * 100
    ).reset_index().sort_values(by='Tasa_Empate', ascending=False)
    
    for _, row in fam_stats.iterrows():
        print(f"Familia: {row['family']:<20} | Total: {row['Total']:<4} | Empates: {row['Empates']:<4} | Tasa: {row['Tasa_Empate']:.2f}%")
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 2: TEST ESTADÍSTICO KS (TIE VS NON-TIE)
    # ---------------------------------------------------------------------
    print("================ 2. PRUEBA DE KOLMOGOROV-SMIRNOV (TIE vs NON-TIE) ================")
    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_skew', 'deg_kurt',
        'deg_het', 'clustering', 'assortativity', 'spectral_gap', 
        'fiedler_value', 'num_communities', 'modularity'
    ]
    
    df_tie = df[df['is_tie'] == 1]
    df_nontie = df[df['is_tie'] == 0]
    
    ks_results = []
    for col in feature_cols:
        stat, pval = stats.ks_2samp(df_tie[col], df_nontie[col])
        ks_results.append({
            'Descriptor': col,
            'Media_Tie': df_tie[col].mean(),
            'Media_NonTie': df_nontie[col].mean(),
            'KS_Stat': stat,
            'p_value': pval,
            'Significativo': 'SÍ (p < 0.01)' if pval < 0.01 else 'NO'
        })
        
    df_ks = pd.DataFrame(ks_results).sort_values(by='p_value')
    print(df_ks.to_string(index=False))
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 3: ANÁLISIS DE LA DISTRIBUCIÓN DEL MARGEN DIFFERENTIAL (|diff|)
    # ---------------------------------------------------------------------
    print("================ 3. ANÁLISIS DEL MARGEN DE RENDIMIENTO (|diff|) ================")
    abs_diff = df['abs_diff'].values
    print(f"Margen Mínimo: {np.min(abs_diff):.6f}")
    print(f"Margen Máximo: {np.max(abs_diff):.6f}")
    print(f"Margen Medio : {np.mean(abs_diff):.6f}")
    print(f"Percentil 25 : {np.percentile(abs_diff, 25):.6f}")
    print(f"Percentil 50 : {np.percentile(abs_diff, 50):.6f} (Mediana)")
    print(f"Percentil 75 : {np.percentile(abs_diff, 75):.6f}")
    print("=================================================================================\n")

    # ---------------------------------------------------------------------
    # MÓDULO 4: BENCHMARK DEL CLASIFICADOR TRI-CLASE (META-LEARNING)
    # ---------------------------------------------------------------------
    print("================ 4. BENCHMARK DE META-LEARNING TRI-CLASE ================")
    print("Clases objetivo: 0=frontier, 1=cohesion, 2=tie")
    
    label_map = {'frontier': 0, 'cohesion': 1, 'tie': 2}
    X_tri = df[feature_cols].values
    y_tri = df['Optimal_Operator'].map(label_map).values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    oof_preds_tri = np.zeros(len(X_tri))
    
    for train_idx, val_idx in skf.split(X_tri, y_tri):
        X_train, y_train = X_tri[train_idx], y_tri[train_idx]
        X_val = X_tri[val_idx]
        
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=2026)
        clf.fit(X_train, y_train)
        oof_preds_tri[val_idx] = clf.predict(X_val)
        
    acc_tri = accuracy_score(y_tri, oof_preds_tri)
    f1_tri = f1_score(y_tri, oof_preds_tri, average='macro')
    
    print(f"Accuracy Tri-Clase OOF: {acc_tri:.4f}")
    print(f"F1 Macro Tri-Clase OOF: {f1_tri:.4f}\n")
    print("Matriz de Confusión Tri-Clase:")
    print(confusion_matrix(y_tri, oof_preds_tri))
    print("\nReporte de Clasificación Tri-Clase:")
    print(classification_report(y_tri, oof_preds_tri, target_names=['Frontier', 'Cohesion', 'Tie'], digits=4))
    
    print("================ EVALUACIÓN DE HIPÓTESIS H2 ================")
    sig_count = sum(1 for r in ks_results if r['p_value'] < 0.01)
    if sig_count >= 3 and f1_tri >= 0.70:
        print("[ACEPTADA] Los empates no son ruido: poseen firma topológica clara y son separables.")
        print("Recomendación: Tratar 'Tie' como un régimen de baja complejidad para ahorrar tiempo de cómputo.")
    else:
        print("[RECHAZADA] Los empates son estocásticos e inseparables del resto.")
    print("======================================================================\n")

if __name__ == '__main__':
    run_phase2_tie_characterization()
