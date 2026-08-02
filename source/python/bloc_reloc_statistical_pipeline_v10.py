# ==============================================================================
# bloc_reloc_statistical_pipeline_v10.py
# BLOC-RELOC v10.0: Pipeline Estadístico Masivo y Meta-Aprendizaje Paramétrico
# Propietario: Laboratorio TCF
# ==============================================================================

import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

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

@dataclass(slots=True)
class ExecutionResult:
    operator_name: str
    initial_cut: int
    final_cut: int
    absolute_improvement: int
    relative_improvement: float
    execution_time: float
    blocks_accepted: int
    partition: np.ndarray


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


class ParametricCorpusGenerator:
    """Generador sistemático de corpus con variación paramétrica para análisis estadístico."""
    
    @staticmethod
    def generate_corpus(base_seed: int = 42) -> List[Tuple[str, nx.Graph, Dict[str, Any]]]:
        corpus = []
        instance_counter = 0

        # 1. Erdos-Renyi: Variación de probabilidad p y tamaño n
        n_values = [600, 800, 1000]
        p_values = [0.015, 0.02, 0.03, 0.04]
        for n in n_values:
            for p in p_values:
                for rep in range(3):
                    seed = base_seed + instance_counter
                    G = nx.erdos_renyi_graph(n, p, seed=seed)
                    corpus.append(("Erdos_Renyi", G, {"family": "ER", "n": n, "param_p": p, "rep": rep}))
                    instance_counter += 1

        # 2. Barabasi-Albert: Variación de grado m y tamaño n
        m_values = [2, 3, 4, 6]
        for n in n_values:
            for m in m_values:
                for rep in range(3):
                    seed = base_seed + instance_counter
                    G = nx.barabasi_albert_graph(n, m, seed=seed)
                    corpus.append(("Barabasi_Albert", G, {"family": "BA", "n": n, "param_m": m, "rep": rep}))
                    instance_counter += 1

        # 3. Watts-Strogatz: Variación de vecinos k y probabilidad p_ws
        k_values = [4, 8, 12]
        p_ws_values = [0.05, 0.1, 0.2, 0.3]
        for n in n_values:
            for k in k_values:
                for p_ws in p_ws_values:
                    for rep in range(2):
                        seed = base_seed + instance_counter
                        G = nx.newman_watts_strogatz_graph(n, k, p_ws, seed=seed)
                        corpus.append(("Watts_Strogatz", G, {"family": "WS", "n": n, "param_k": k, "param_p_ws": p_ws, "rep": rep}))
                        instance_counter += 1

        # 4. Powerlaw Cluster: Variación de m y p_pc
        pc_m_values = [2, 4]
        pc_p_values = [0.02, 0.05, 0.1]
        for n in n_values:
            for m in pc_m_values:
                for p_pc in pc_p_values:
                    for rep in range(3):
                        seed = base_seed + instance_counter
                        G = nx.powerlaw_cluster_graph(n, m, p_pc, seed=seed)
                        corpus.append(("Powerlaw_Cluster", G, {"family": "PC", "n": n, "param_m": m, "param_p_pc": p_pc, "rep": rep}))
                        instance_counter += 1

        # 5. Random Geometric: Variación de radio
        radius_values = [0.09, 0.12, 0.15, 0.18]
        for n in n_values:
            for radius in radius_values:
                for rep in range(3):
                    seed = base_seed + instance_counter
                    G = nx.random_geometric_graph(n, radius, seed=seed)
                    corpus.append(("Random_Geometric", G, {"family": "RG", "n": n, "param_radius": radius, "rep": rep}))
                    instance_counter += 1

        return corpus


class DescriptorExtractor:
    @staticmethod
    def extract(G: nx.Graph, csr: GraphCSR) -> Dict[str, float]:
        N = csr.N
        E = G.number_of_edges()
        degrees = csr.degrees
        
        clustering = float(nx.average_clustering(G))
        deg_mean = float(np.mean(degrees))
        deg_std = float(np.std(degrees))
        deg_het = float(deg_std / (deg_mean + 1e-6))
        density = float((2.0 * E) / (N * (N - 1)) if N > 1 else 0.0)
        assortativity = float(nx.degree_assortativity_coefficient(G))
        if np.isnan(assortativity):
            assortativity = 0.0
            
        return {
            "N": float(N),
            "E": float(E),
            "clustering": clustering,
            "deg_mean": deg_mean,
            "deg_std": deg_std,
            "deg_het": deg_het,
            "density": density,
            "assortativity": assortativity
        }


class PartitionEngine:
    @staticmethod
    def get_spectral_partition(G: nx.Graph, csr: GraphCSR) -> np.ndarray:
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


class BaseOperator(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def refine(self, csr: GraphCSR, part_init: np.ndarray, tolerance: float, max_iters: int, max_escapes: int) -> ExecutionResult:
        pass


class FrontierOperator(BaseOperator):
    @property
    def name(self) -> str:
        return "frontier"

    def refine(self, csr: GraphCSR, part_init: np.ndarray, tolerance: float = 0.05, max_iters: int = 15, max_escapes: int = 3) -> ExecutionResult:
        start_time = time.perf_counter()
        N = csr.N
        min_size = int((0.5 - tolerance) * N)
        max_size = int((0.5 + tolerance) * N)
        
        part = part_init.copy()
        n0 = np.sum(part == 0)
        n1 = N - n0

        def compute_cut(p_arr):
            return int(np.sum(p_arr[csr.row_indices] != p_arr[csr.indices])) // 2

        initial_cut = compute_cut(part)
        best_cut = initial_cut
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

                if float(int_deg - ext_deg) < 0:
                    can_move = (curr_p == 0 and n0 - 1 >= min_size and n1 + 1 <= max_size) or \
                               (curr_p == 1 and n1 - 1 >= min_size and n0 + 1 <= max_size)
                    if can_move:
                        part[u] = 1 - curr_p
                        n0, n1 = (n0 - 1, n1 + 1) if curr_p == 0 else (n0 + 1, n1 - 1)
                        improved = True

            current_cut = compute_cut(part)
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
                        same_color_neighbors = neighbors[neighbor_parts == curr_p]
                        if len(same_color_neighbors) == 0:
                            continue
                        dynamic_limit = min(8, max(2, len(same_color_neighbors) // 2))
                        nodes_to_move = [u] + list(same_color_neighbors[:dynamic_limit])
                        block_size = len(nodes_to_move)
                        
                        can_shift = (curr_p == 0 and n0 - block_size >= min_size and n1 + block_size <= max_size) or \
                                    (curr_p == 1 and n1 - block_size >= min_size and n0 + block_size <= max_size)
                        if can_shift:
                            part[nodes_to_move] = 1 - curr_p
                            cut_candidate = compute_cut(part)
                            if cut_candidate < current_cut:
                                n0, n1 = (n0 - block_size, n1 + block_size) if curr_p == 0 else (n0 + block_size, n1 - block_size)
                                current_cut = cut_candidate
                                mutated += 1
                                blocks_accepted += 1
                            else:
                                part[nodes_to_move] = curr_p
                if mutated > 0:
                    escapes_performed += 1
                    best_cut = min(best_cut, current_cut)
                    continue
                else:
                    break
            elif not improved:
                break

        elapsed = time.perf_counter() - start_time
        abs_imp = initial_cut - best_cut
        rel_imp = abs_imp / (initial_cut + 1e-6)
        
        return ExecutionResult(
            operator_name=self.name,
            initial_cut=initial_cut,
            final_cut=best_cut,
            absolute_improvement=abs_imp,
            relative_improvement=rel_imp,
            execution_time=elapsed,
            blocks_accepted=blocks_accepted,
            partition=part
        )


class CohesionOperator(BaseOperator):
    @property
    def name(self) -> str:
        return "cohesion"

    def refine(self, csr: GraphCSR, part_init: np.ndarray, tolerance: float = 0.05, max_iters: int = 15, max_escapes: int = 3) -> ExecutionResult:
        start_time = time.perf_counter()
        N = csr.N
        min_size = int((0.5 - tolerance) * N)
        max_size = int((0.5 + tolerance) * N)
        
        part = part_init.copy()
        n0 = np.sum(part == 0)
        n1 = N - n0

        def compute_cut(p_arr):
            return int(np.sum(p_arr[csr.row_indices] != p_arr[csr.indices])) // 2

        initial_cut = compute_cut(part)
        best_cut = initial_cut
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

                if float(int_deg - ext_deg) < 0:
                    can_move = (curr_p == 0 and n0 - 1 >= min_size and n1 + 1 <= max_size) or \
                               (curr_p == 1 and n1 - 1 >= min_size and n0 + 1 <= max_size)
                    if can_move:
                        part[u] = 1 - curr_p
                        n0, n1 = (n0 - 1, n1 + 1) if curr_p == 0 else (n0 + 1, n1 - 1)
                        improved = True

            current_cut = compute_cut(part)
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
                        same_color_neighbors = neighbors[neighbor_parts == curr_p]
                        if len(same_color_neighbors) == 0:
                            continue
                        u_neighbors_set = set(neighbors)
                        cohesion_scores = []
                        for v in same_color_neighbors:
                            v_start, v_end = csr.indptr[v], csr.indptr[v + 1]
                            shared_count = len(u_neighbors_set.intersection(csr.indices[v_start:v_end]))
                            cohesion_scores.append((shared_count, v))
                        cohesion_scores.sort(key=lambda x: x[0], reverse=True)
                        dynamic_limit = min(6, max(1, len(cohesion_scores)))
                        nodes_to_move = [u] + [v for _, v in cohesion_scores[:dynamic_limit]]
                        block_size = len(nodes_to_move)
                        
                        can_shift = (curr_p == 0 and n0 - block_size >= min_size and n1 + block_size <= max_size) or \
                                    (curr_p == 1 and n1 - block_size >= min_size and n0 + block_size <= max_size)
                        if can_shift:
                            part[nodes_to_move] = 1 - curr_p
                            cut_candidate = compute_cut(part)
                            if cut_candidate < current_cut:
                                n0, n1 = (n0 - block_size, n1 + block_size) if curr_p == 0 else (n0 + block_size, n1 - block_size)
                                current_cut = cut_candidate
                                mutated += 1
                                blocks_accepted += 1
                            else:
                                part[nodes_to_move] = curr_p
                if mutated > 0:
                    escapes_performed += 1
                    best_cut = min(best_cut, current_cut)
                    continue
                else:
                    break
            elif not improved:
                break

        elapsed = time.perf_counter() - start_time
        abs_imp = initial_cut - best_cut
        rel_imp = abs_imp / (initial_cut + 1e-6)
        
        return ExecutionResult(
            operator_name=self.name,
            initial_cut=initial_cut,
            final_cut=best_cut,
            absolute_improvement=abs_imp,
            relative_improvement=rel_imp,
            execution_time=elapsed,
            blocks_accepted=blocks_accepted,
            partition=part
        )


class StatisticalOrchestrator:
    def __init__(self):
        self.operators: Dict[str, BaseOperator] = {}
        self.register_operator(FrontierOperator())
        self.register_operator(CohesionOperator())

    def register_operator(self, operator: BaseOperator) -> None:
        self.operators[operator.name] = operator

    def run(self, seed: int = 42) -> pd.DataFrame:
        corpus = ParametricCorpusGenerator.generate_corpus(base_seed=seed)
        records = []

        print(f"[*] Iniciando Pipeline Estadístico BLOC-RELOC v10.0 | Total Instancias: {len(corpus)}")
        
        for idx, (family, G_raw, gen_params) in enumerate(corpus):
            G = nx.convert_node_labels_to_integers(G_raw)
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
                G = nx.convert_node_labels_to_integers(G)
                
            csr = nx_to_csr(G)
            descriptors = DescriptorExtractor.extract(G, csr)
            
            np.random.seed(seed + idx)
            part_init = PartitionEngine.get_spectral_partition(G, csr)

            results: Dict[str, ExecutionResult] = {}
            for op_name, op_obj in self.operators.items():
                results[op_name] = op_obj.refine(csr, part_init)

            sorted_ops = sorted(results.items(), key=lambda item: item[1].final_cut)
            best_op_name, best_res = sorted_ops[0]
            worst_op_name, worst_res = sorted_ops[-1]
            
            is_tie = (best_res.final_cut == sorted_ops[1][1].final_cut) if len(sorted_ops) > 1 else False

            record = {
                "Instance_ID": f"{family}_{idx+1}",
                "Family": family,
                **gen_params,
                **descriptors,
            }

            for op_name, res in results.items():
                record[f"Cut_{op_name}"] = res.final_cut
                record[f"Time_{op_name}"] = res.execution_time
                record[f"Blocks_{op_name}"] = res.blocks_accepted
                record[f"AbsImp_{op_name}"] = res.absolute_improvement
                record[f"RelImp_{op_name}"] = res.relative_improvement

            record["Optimal_Operator"] = best_op_name
            record["Is_Tie"] = int(is_tie)
            record["Cut_Delta_Worst_Best"] = worst_res.final_cut - best_res.final_cut

            records.append(record)
            if (idx + 1) % 10 == 0 or (idx + 1) == len(corpus):
                print(f"    [+] Progreso: {idx+1}/{len(corpus)} instancias procesadas...")

        df = pd.DataFrame(records)
        return df


if __name__ == "__main__":
    start_total = time.perf_counter()
    orchestrator = StatisticalOrchestrator()
    df_meta = orchestrator.run(seed=42)
    
    output_filename = "meta_statistical_dataset_v10.csv"
    df_meta.to_csv(output_filename, index=False)
    
    elapsed_total = time.perf_counter() - start_total
    print(f"\n[✓] Campaña estadística finalizada en {elapsed_total:.2f} segundos.")
    print(f"[✓] Dataset guardado en: {output_filename}")
    
    print("\n==================== RESUMEN ESTADÍSTICO ====================")
    print(df_meta.groupby("Family")["Optimal_Operator"].value_counts(normalize=True) * 100)
    print("=============================================================")
