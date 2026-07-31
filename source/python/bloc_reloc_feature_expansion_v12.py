import sys
import warnings
import time
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import skew, kurtosis

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass(slots=True, frozen=True)
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

@dataclass(slots=True)
class RefinementResult:
    final_cut: int
    execution_time: float
    blocks_accepted: int
    absolute_improvement: int
    relative_improvement: float

def compute_cut(csr: GraphCSR, partition: np.ndarray) -> int:
    """Calcula el corte real de la partición sobre la matriz de adyacencia CSR."""
    indptr = csr.indptr
    indices = csr.indices
    cut = 0
    N = csr.N
    for u in range(N):
        p_u = partition[u]
        for idx in range(indptr[u], indptr[u+1]):
            v = indices[idx]
            if u < v and p_u != partition[v]:
                cut += 1
    return cut

class FrontierOperator:
    """
    Operador Frontier Real: Refinamiento basado en nodos de frontera.
    Evalúa la ganancia de intercambiar nodos en el límite entre particiones.
    """
    def refine(self, csr: GraphCSR, partition: np.ndarray) -> RefinementResult:
        start_time = time.time()
        part = partition.copy()
        N = csr.N
        indptr = csr.indptr
        indices = csr.indices
        
        initial_cut = compute_cut(csr, part)
        current_cut = initial_cut
        blocks_accepted = 0
        
        for _ in range(5):
            improved = False
            frontier_nodes = []
            for u in range(N):
                p_u = part[u]
                ext_deg = sum(1 for idx in range(indptr[u], indptr[u+1]) if part[indices[idx]] != p_u)
                int_deg = (indptr[u+1] - indptr[u]) - ext_deg
                gain = ext_deg - int_deg
                if ext_deg > 0:
                    frontier_nodes.append((gain, u))
            
            frontier_nodes.sort(key=lambda x: x[0], reverse=True)
            for gain, u in frontier_nodes:
                if gain > 0:
                    part[u] = 1 - part[u]
                    current_cut -= gain
                    blocks_accepted += 1
                    improved = True
            if not improved:
                break
                
        elapsed = time.time() - start_time
        abs_imp = initial_cut - current_cut
        rel_imp = float(abs_imp / (initial_cut + 1e-6))
        return RefinementResult(
            final_cut=current_cut,
            execution_time=elapsed,
            blocks_accepted=blocks_accepted,
            absolute_improvement=abs_imp,
            relative_improvement=rel_imp
        )

class CohesionOperator:
    """
    Operador Cohesion Real: Refinamiento por densidad interna de bloques.
    Reasigna nodos priorizando la cohesión de los clusters formados.
    """
    def refine(self, csr: GraphCSR, partition: np.ndarray) -> RefinementResult:
        start_time = time.time()
        part = partition.copy()
        N = csr.N
        indptr = csr.indptr
        indices = csr.indices
        
        initial_cut = compute_cut(csr, part)
        current_cut = initial_cut
        blocks_accepted = 0
        
        for pass_idx in range(5):
            improved = False
            nodes = np.arange(N)
            np.random.seed(42 + pass_idx)
            np.random.shuffle(nodes)
            for u in nodes:
                p_u = part[u]
                deg_u = indptr[u+1] - indptr[u]
                if deg_u == 0:
                    continue
                ext_deg = sum(1 for idx in range(indptr[u], indptr[u+1]) if part[indices[idx]] != p_u)
                int_deg = deg_u - ext_deg
                if ext_deg >= int_deg and ext_deg > 0:
                    part[u] = 1 - p_u
                    new_cut = compute_cut(csr, part)
                    if new_cut <= current_cut:
                        if new_cut < current_cut:
                            improved = True
                        current_cut = new_cut
                        blocks_accepted += 1
                    else:
                        part[u] = p_u
            if not improved:
                break
                
        elapsed = time.time() - start_time
        abs_imp = initial_cut - current_cut
        rel_imp = float(abs_imp / (initial_cut + 1e-6))
        return RefinementResult(
            final_cut=current_cut,
            execution_time=elapsed,
            blocks_accepted=blocks_accepted,
            absolute_improvement=abs_imp,
            relative_improvement=rel_imp
        )

class PartitionEngine:
    @staticmethod
    def get_spectral_partition(G: nx.Graph, csr: GraphCSR) -> np.ndarray:
        try:
            adj = nx.adjacency_matrix(G).astype(float)
            deg = np.array(adj.sum(axis=1)).flatten()
            D_inv_sqrt = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
            L_norm = sp.eye(csr.N) - D_inv_sqrt @ adj @ D_inv_sqrt
            _, evals = spla.eigsh(L_norm, k=2, which="SM", tol=1e-4)
            fiedler_vec = evals[:, 1]
            return (fiedler_vec >= 0).astype(int)
        except Exception:
            return np.random.choice([0, 1], size=csr.N)

class AdvancedDescriptorExtractor:
    @staticmethod
    def _safe_val(val: Any, default: float = 0.0) -> float:
        try:
            fval = float(val)
            if not np.isfinite(fval):
                return default
            return fval
        except Exception:
            return default

    @staticmethod
    def extract(G: nx.Graph, csr: GraphCSR) -> Dict[str, float]:
        N = csr.N
        E = G.number_of_edges()
        degrees = csr.degrees
        
        deg_mean = AdvancedDescriptorExtractor._safe_val(np.mean(degrees))
        deg_std = AdvancedDescriptorExtractor._safe_val(np.std(degrees))
        deg_het = AdvancedDescriptorExtractor._safe_val(deg_std / (deg_mean + 1e-6))
        
        try:
            deg_skew = float(skew(degrees)) if N > 2 else 0.0
        except Exception:
            deg_skew = 0.0
        deg_skew = AdvancedDescriptorExtractor._safe_val(deg_skew)
        
        try:
            deg_kurt = float(kurtosis(degrees)) if N > 3 else 0.0
        except Exception:
            deg_kurt = 0.0
        deg_kurt = AdvancedDescriptorExtractor._safe_val(deg_kurt)
        
        try:
            clustering = float(nx.average_clustering(G))
        except Exception:
            clustering = 0.0
        clustering = AdvancedDescriptorExtractor._safe_val(clustering)
        
        density = AdvancedDescriptorExtractor._safe_val((2.0 * E) / (N * (N - 1)) if N > 1 else 0.0)
        
        try:
            assortativity = float(nx.degree_assortativity_coefficient(G))
        except Exception:
            assortativity = 0.0
        assortativity = AdvancedDescriptorExtractor._safe_val(assortativity)

        fiedler_value = 0.0
        spectral_gap = 0.0
        try:
            adj = nx.adjacency_matrix(G).astype(float)
            deg = np.array(adj.sum(axis=1)).flatten()
            D_inv_sqrt = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
            L_norm = sp.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt
            evals, _ = spla.eigsh(L_norm, k=2, which="SM", tol=1e-4)
            evals = np.sort(evals)
            if len(evals) > 1:
                fiedler_value = AdvancedDescriptorExtractor._safe_val(evals[1])
                spectral_gap = AdvancedDescriptorExtractor._safe_val(evals[1] - evals[0])
        except Exception:
            pass

        modularity = 0.0
        num_communities = 1.0
        try:
            communities = nx.community.greedy_modularity_communities(G)
            modularity = AdvancedDescriptorExtractor._safe_val(nx.community.modularity(G, communities))
            num_communities = AdvancedDescriptorExtractor._safe_val(len(communities), 1.0)
        except Exception:
            pass

        return {
            "N": float(N),
            "E": float(E),
            "density": density,
            "deg_mean": deg_mean,
            "deg_std": deg_std,
            "deg_het": deg_het,
            "deg_skew": deg_skew,
            "deg_kurt": deg_kurt,
            "clustering": clustering,
            "assortativity": assortativity,
            "fiedler_value": fiedler_value,
            "spectral_gap": spectral_gap,
            "modularity": modularity,
            "num_communities": num_communities
        }

class ParametricCorpusGenerator:
    @staticmethod
    def generate_corpus(base_seed: int = 42) -> List[Tuple[str, nx.Graph, Dict[str, Any]]]:
        np.random.seed(base_seed)
        corpus = []
        
        families = [
            ("erdos_renyi", lambda i: nx.erdos_renyi_graph(50 + i*5, 0.1 + (i%5)*0.02, seed=base_seed+i)),
            ("watts_strogatz", lambda i: nx.watts_strogatz_graph(60 + i*4, 6 + (i%3)*2, 0.1 + (i%4)*0.05, seed=base_seed+i)),
            ("barabasi_albert", lambda i: nx.barabasi_albert_graph(60 + i*5, 3 + (i%3), seed=base_seed+i)),
            ("powerlaw_cluster", lambda i: nx.powerlaw_cluster_graph(60 + i*4, 3, 0.2, seed=base_seed+i)),
            ("random_regular", lambda i: nx.random_regular_graph(4 + (i%4)*2, 60 + i*4, seed=base_seed+i))
        ]
        
        for family_name, builder in families:
            for i in range(12):
                try:
                    G = builder(i)
                    gen_params = {"family_index": i, "generator": family_name}
                    corpus.append((family_name, G, gen_params))
                except Exception:
                    continue
        return corpus

def main():
    print("[*] Generando corpus y ejecutando refinamiento real con operadores dinamicos...")
    corpus = ParametricCorpusGenerator.generate_corpus(base_seed=42)
    records = []
    operators = {"frontier": FrontierOperator(), "cohesion": CohesionOperator()}

    for idx, (family, G_raw, gen_params) in enumerate(corpus):
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
        
        csr = nx_to_csr(G)
        descriptors = AdvancedDescriptorExtractor.extract(G, csr)
        
        np.random.seed(42 + idx)
        part_init = PartitionEngine.get_spectral_partition(G, csr)
        
        results = {}
        for op_name, op_obj in operators.items():
            results[op_name] = op_obj.refine(csr, part_init)
            
        sorted_ops = sorted(results.items(), key=lambda item: item[1].final_cut)
        best_op_name, best_res = sorted_ops[0]
        worst_op_name, worst_res = sorted_ops[-1]
        is_tie = (best_res.final_cut == sorted_ops[1][1].final_cut) if len(sorted_ops) > 1 else False
        
        record = {"Instance_ID": f"{family}_{idx+1}", "Family": family, **gen_params, **descriptors}
        for op_name, res in results.items():
            record[f"Cut_{op_name}"] = res.final_cut
            record[f"Time_{op_name}"] = res.execution_time
            record[f"Blocks_{op_name}"] = res.blocks_accepted
            record[f"AbsImp_{op_name}"] = res.absolute_improvement
            record[f"RelImp_{op_name}"] = res.relative_improvement
            
        record["Optimal_Operator"] = "tie" if is_tie else best_op_name
        record["Is_Tie"] = int(is_tie)
        record["Cut_Delta_Worst_Best"] = worst_res.final_cut - best_res.final_cut
        records.append(record)

    df_v12 = pd.DataFrame(records)

    # Limpieza rigurosa de datos numericos
    numeric_cols = df_v12.select_dtypes(include=[np.number]).columns
    df_v12[numeric_cols] = (
        df_v12[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
    )

    assert np.isfinite(df_v12[numeric_cols].to_numpy()).all(), "Error critico: El dataset contiene valores no finitos."

    # Impresion de la distribucion real de clases
    print("\n================ DISTRIBUTION DE CLASES EN OPTIMAL_OPERATOR ================")
    class_counts = df_v12["Optimal_Operator"].value_counts()
    print(class_counts)
    print("===========================================================================")

    df_train = df_v12[df_v12["Optimal_Operator"] != "tie"].copy()
    unique_classes = df_train["Optimal_Operator"].unique()
    
    # Validacion metodologica explicita
    if len(unique_classes) < 2:
        print("\n[!] ERROR CRITICO METODOLOGICO: El dataset contiene una sola clase activa.")
        print("El meta-selector requiere al menos dos clases (Frontier y Cohesion) para poder entrenar.")
        sys.exit(1)

    output_filename = "meta_statistical_dataset_v12.csv"
    df_v12.to_csv(output_filename, index=False)
    print(f"\n[✓] Dataset v12 generado y validado en '{output_filename}' con {df_v12.shape[0]} instancias.")

    feature_cols = [
        'N', 'E', 'density', 'deg_mean', 'deg_std', 'deg_het', 
        'deg_skew', 'deg_kurt', 'clustering', 'assortativity', 
        'fiedler_value', 'spectral_gap', 'modularity', 'num_communities'
    ]
    
    X = df_train[feature_cols]
    y = (df_train['Optimal_Operator'] == 'cohesion').astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n==================== EVALUACION MODELO V12 (OOF) ====================")
    gb = GradientBoostingClassifier(random_state=42)
    y_pred_oof = cross_val_predict(gb, X_scaled, y, cv=cv)
    
    print("Matriz de Confusion Out-of-Fold (Gradient Boosting - v12):")
    print(confusion_matrix(y, y_pred_oof))
    print("\nReporte de Clasificacion v12:")
    print(classification_report(y, y_pred_oof, target_names=['Frontier', 'Cohesion']))

    gb.fit(X_scaled, y)
    importances = pd.Series(gb.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop Importancia de Variables (Gradient Boosting v12):")
    print(importances)
    print("=====================================================================")

if __name__ == "__main__":
    main()
