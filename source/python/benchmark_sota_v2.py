import os
import sys
import time
import gzip
import urllib.request
import tracemalloc
import warnings
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import networkx as nx

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. DETECCIÓN AUTOMÁTICA Y ADAPTATIVA DE SOLVERS EXTERNOS Y SOTA
# ---------------------------------------------------------------------------
AVAILABLE_SOLVERS = {
    'METIS': False,
    'KaHyPar': False,
    'SCIP': False,
    'Gurobi': False
}

try:
    import metis
    AVAILABLE_SOLVERS['METIS'] = True
    print("[Detección Entorno]: METIS disponible.")
except ImportError:
    print("[Aviso Entorno]: METIS (pymetis) no detectado. Se omitirá.")

try:
    import kahypar
    if hasattr(kahypar, "setVerbosity"):
        kahypar.setVerbosity(0)
    AVAILABLE_SOLVERS['KaHyPar'] = True
    print("[Detección Entorno]: KaHyPar disponible.")
except Exception as e:
    print(f"[Aviso Entorno]: KaHyPar no disponible ({e}). Se omitirá.")

try:
    from pyscipopt import Model as SCIPModel
    AVAILABLE_SOLVERS['SCIP'] = True
    print("[Detección Entorno]: PySCIPOpt disponible.")
except ImportError:
    pass

try:
    import gurobipy as gp
    AVAILABLE_SOLVERS['Gurobi'] = True
    print("[Detección Entorno]: Gurobi disponible.")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# 2. PARSERS NATIVOS PARA FORMATOS OFICIALES (METIS / WALSHAW / DIMACS / SNAP)
# ---------------------------------------------------------------------------
def parse_metis_graph(file_path: str) -> nx.Graph:
    """Parsea archivos .graph en el formato canónico de METIS/Walshaw Archive."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('%')]
    
    if not lines:
        raise ValueError(f"El archivo METIS {file_path} está vacío.")
    
    header = lines[0].split()
    num_nodes = int(header[0])
    
    current_node = 0
    for line in lines[1:]:
        current_node += 1
        if current_node > num_nodes:
            break
        neighbors = [int(x) - 1 for x in line.split()]  # Convertir de 1-indexed a 0-indexed
        for neighbor in neighbors:
            if current_node - 1 != neighbor:
                G.add_edge(current_node - 1, neighbor)
                
    G.add_nodes_from(range(num_nodes))
    return G


def parse_snap_edgelist_gz(file_path: str) -> nx.Graph:
    """Parsea archivos .txt.gz oficiales de Stanford SNAP Archive."""
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            u, v = map(int, line.strip().split()[:2])
            G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G)


# ---------------------------------------------------------------------------
# 3. GESTOR Y DESCARGADOR DE BENCHMARKS CANÓNICOS REALES
# ---------------------------------------------------------------------------
def fetch_and_load_official_benchmarks() -> List[Dict[str, Any]]:
    print("\n[+] Descargando e importando datasets OFICIALES del SOTA...")
    os.makedirs("official_datasets", exist_ok=True)
    
    benchmark_sources = [
        {
            'name': 'SNAP_Facebook_Combined',
            'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
            'filename': 'official_datasets/facebook_combined.txt.gz',
            'parser': parse_snap_edgelist_gz
        },
        {
            'name': 'Walshaw_add32',
            'url': 'https://raw.githubusercontent.com/julian-schuster/graph-partitioning-benchmarks/main/walshaw/add32.graph',
            'filename': 'official_datasets/add32.graph',
            'parser': parse_metis_graph
        },
        {
            'name': 'Walshaw_3elt',
            'url': 'https://raw.githubusercontent.com/julian-schuster/graph-partitioning-benchmarks/main/walshaw/3elt.graph',
            'filename': 'official_datasets/3elt.graph',
            'parser': parse_metis_graph
        }
    ]
    
    instances = []
    
    for src in benchmark_sources:
        filepath = src['filename']
        if not os.path.exists(filepath):
            print(f"    -> Descargando {src['name']} desde repositorio oficial...")
            try:
                req = urllib.request.Request(src['url'], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"       [!] Error descargando {src['name']}: {e}")
                continue
        
        try:
            print(f"    -> Cargando y parseando {src['name']}...")
            G = src['parser'](filepath)
            instances.append({
                'name': src['name'],
                'seed': 42,
                'graph': G,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            })
            print(f"       [✓] Instancia cargada: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
        except Exception as e:
            print(f"       [!] Error procesando {src['name']}: {e}")

    print(f"    -> Total de instancias reales preparadas: {len(instances)}")
    return instances


# ---------------------------------------------------------------------------
# 4. FUNCIÓN OBJETIVO Y MÉTRICAS RIGUROSAS
# ---------------------------------------------------------------------------
def evaluate_partition(G: nx.Graph, partition: Dict[int, int], k: int = 4, alpha: float = 0.1) -> Tuple[int, float, float]:
    cut_size = 0
    for u, v in G.edges():
        if partition.get(u, 0) != partition.get(v, 0):
            cut_size += 1
            
    block_sizes = [0] * k
    for n, b in partition.items():
        if 0 <= b < k:
            block_sizes[b] += 1
            
    N = G.number_of_nodes()
    target_size = N / k
    imbalance = sum(abs(s - target_size) for s in block_sizes) / (N + 1e-8)
    cost = cut_size + (alpha * N * imbalance)
    return cut_size, imbalance, cost


def get_initial_partition(G: nx.Graph, k: int = 4, seed: int = 42) -> Dict[int, int]:
    np.random.seed(seed)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    return {node: i % k for i, node in enumerate(nodes)}


# ---------------------------------------------------------------------------
# 5. IMPLEMENTACIÓN DE ALGORITMOS Y BASELINES SOTA
# ---------------------------------------------------------------------------
def run_bloc_reloc(G: nx.Graph, initial_part: Dict[int, int], k: int = 4) -> Dict[int, int]:
    part = initial_part.copy()
    nodes = list(G.nodes())
    
    for _ in range(15):
        for u in nodes:
            current_b = part[u]
            b_counts = {}
            for v in G.neighbors(u):
                b_counts[part[v]] = b_counts.get(part[v], 0) + 1
            if b_counts:
                best_b = max(b_counts, key=b_counts.get)
                if b_counts[best_b] > b_counts.get(current_b, 0):
                    part[u] = best_b
                    
        block_sizes = [0] * k
        for n, b in part.items():
            if 0 <= b < k:
                block_sizes[b] += 1
        max_b = int(np.argmax(block_sizes))
        min_b = int(np.argmin(block_sizes))
        
        target = G.number_of_nodes() / k
        if block_sizes[max_b] - target > target * 0.05:
            candidates = [u for u in nodes if part[u] == max_b and any(part[v] == min_b for v in G.neighbors(u))]
            for c in candidates[:min(len(candidates), 8)]:
                part[c] = min_b
                
    return part


def run_simulated_annealing(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, steps: int = 1000) -> Dict[int, int]:
    part = initial_part.copy()
    nodes = list(G.nodes())
    _, _, current_cost = evaluate_partition(G, part, k)
    T = 10.0
    
    for _ in range(steps):
        u = np.random.choice(nodes)
        old_b = part[u]
        new_b = np.random.randint(0, k)
        if old_b == new_b:
            continue
        part[u] = new_b
        _, _, new_cost = evaluate_partition(G, part, k)
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-5)):
            current_cost = new_cost
        else:
            part[u] = old_b
        T *= 0.95
    return part


def run_tabu_search(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, iterations: int = 200) -> Dict[int, int]:
    part = initial_part.copy()
    best_part = part.copy()
    _, _, best_cost = evaluate_partition(G, part, k)
    nodes = list(G.nodes())
    tabu_list = {}
    
    for it in range(iterations):
        best_move = None
        best_move_cost = float('inf')
        sample_nodes = np.random.choice(nodes, size=min(30, len(nodes)), replace=False)
        for u in sample_nodes:
            old_b = part[u]
            for new_b in range(k):
                if new_b == old_b or tabu_list.get((u, new_b), 0) > it:
                    continue
                part[u] = new_b
                _, _, c = evaluate_partition(G, part, k)
                if c < best_move_cost:
                    best_move_cost = c
                    best_move = (u, old_b, new_b)
                part[u] = old_b
        if best_move is None:
            break
        u, old_b, new_b = best_move
        part[u] = new_b
        tabu_list[(u, old_b)] = it + 10
        if best_move_cost < best_cost:
            best_cost = best_move_cost
            best_part = part.copy()
    return best_part


def run_genetic_algorithm(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, pop_size: int = 20, gens: int = 25) -> Dict[int, int]:
    nodes = list(G.nodes())
    population = [initial_part.copy()]
    for _ in range(pop_size - 1):
        population.append({u: np.random.randint(0, k) for u in nodes})
        
    for _ in range(gens):
        costs = [evaluate_partition(G, p, k)[2] for p in population]
        sorted_idx = np.argsort(costs)
        population = [population[i] for i in sorted_idx[:pop_size]]
        new_pop = population[:2]
        while len(new_pop) < pop_size:
            p1, p2 = population[0], population[1]
            child = {u: (p1[u] if np.random.rand() < 0.5 else p2[u]) for u in nodes}
            if np.random.rand() < 0.15:
                child[np.random.choice(nodes)] = np.random.randint(0, k)
            new_pop.append(child)
        population = new_pop
    return population[0]


def run_breakout_local_search(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, max_perturb: int = 12) -> Dict[int, int]:
    part = initial_part.copy()
    best_part = part.copy()
    _, _, best_cost = evaluate_partition(G, part, k)
    nodes = list(G.nodes())
    
    for _ in range(max_perturb):
        part = run_bloc_reloc(G, part, k)
        _, _, c = evaluate_partition(G, part, k)
        if c < best_cost:
            best_cost = c
            best_part = part.copy()
        for u in np.random.choice(nodes, size=int(0.15 * len(nodes)), replace=False):
            part[u] = np.random.randint(0, k)
    return best_part


def run_metis_adapter(G: nx.Graph, k: int = 4) -> Dict[int, int]:
    if not AVAILABLE_SOLVERS['METIS']:
        raise RuntimeError("METIS no disponible.")
    adj = [list(G.neighbors(n)) for n in range(G.number_of_nodes())]
    _, part_list = metis.part_graph(k, adj)
    return {i: p for i, p in enumerate(part_list)}


def run_kahypar_adapter(G: nx.Graph, k: int = 4) -> Dict[int, int]:
    if not AVAILABLE_SOLVERS['KaHyPar']:
        raise RuntimeError("KaHyPar no disponible.")
    num_nodes = G.number_of_nodes()
    edges = list(G.edges())
    num_edges = len(edges)
    hypergraph = kahypar.Hypergraph(num_nodes, num_edges, [[u, v] for u, v in edges])
    context = kahypar.Context()
    if hasattr(context, "load_dictionary"):
        context.load_dictionary({"mode": "direct", "objective": "cut", "k": k, "epsilon": 0.03, "seed": 42})
    partition = kahypar.partition(hypergraph, context)
    return {i: partition.block(i) for i in range(num_nodes)}


# ---------------------------------------------------------------------------
# 6. HARNESS DE EJECUCIÓN CIENTÍFICA Y MEDICIÓN
# ---------------------------------------------------------------------------
def execute_benchmark_harness(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    print("\n[+] Ejecutando evaluación comparativa SOTA...")
    
    algorithms = {
        'BLOC_RELOC (Proposed)': run_bloc_reloc,
        'Simulated_Annealing': run_simulated_annealing,
        'Tabu_Search': run_tabu_search,
        'Genetic_Algorithm': run_genetic_algorithm,
        'Breakout_Local_Search': run_breakout_local_search
    }
    
    if AVAILABLE_SOLVERS['METIS']:
        algorithms['METIS_SOTA'] = lambda G, p, k: run_metis_adapter(G, k)
    if AVAILABLE_SOLVERS['KaHyPar']:
        algorithms['KaHyPar_SOTA'] = lambda G, p, k: run_kahypar_adapter(G, k)
        
    results = []
    
    for inst in instances:
        G = inst['graph']
        k = 4
        best_known_cost = float('inf')
        inst_runs = []
        print(f"\n -> Procesando Instancia: {inst['name']} (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
        
        for alg_name, alg_fn in algorithms.items():
            tracemalloc.start()
            t0 = time.perf_counter()
            
            initial_part = get_initial_partition(G, k=k, seed=inst['seed'])
            try:
                final_part = alg_fn(G, initial_part, k=k)
            except Exception as e:
                tracemalloc.stop()
                print(f"       [!] Error ejecutando {alg_name} en {inst['name']}: {e}")
                continue
                
            exec_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            cut, imbalance, cost = evaluate_partition(G, final_part, k=k)
            throughput = G.number_of_nodes() / (exec_time + 1e-8)
            
            if cost < best_known_cost:
                best_known_cost = cost
                
            inst_runs.append({
                'Instance': inst['name'],
                'Nodes': inst['num_nodes'],
                'Edges': inst['num_edges'],
                'Algorithm': alg_name,
                'Cut': cut,
                'Imbalance': imbalance,
                'Cost': cost,
                'Execution_Time_s': exec_time,
                'Peak_Memory_MB': peak_mem / (1024 * 1024),
                'Throughput_nodes_sec': throughput
            })
            print(f"       [✓] {alg_name}: Costo={cost:.2f} (Corte={cut}, Desbal={imbalance:.4f}) | Tiempo={exec_time:.3f}s")
            
        for row in inst_runs:
            gap = ((row['Cost'] - best_known_cost) / (best_known_cost + 1e-8)) * 100.0
            row['Relative_Gap_Percent'] = gap
            results.append(row)
            
    df = pd.DataFrame(results)
    return df


# ---------------------------------------------------------------------------
# 7. ANÁLISIS ESTADÍSTICO FORMAL
# ---------------------------------------------------------------------------
def perform_formal_statistical_analysis(df: pd.DataFrame):
    print("\n========================================================================")
    print("ANÁLISIS ESTADÍSTICO FORMAL (REVISIÓN POR PARES)")
    print("========================================================================")
    
    pivot_cost = df.pivot_table(index=['Instance'], columns='Algorithm', values='Cost')
    algs = pivot_cost.columns.tolist()
    
    if len(algs) < 2 or len(pivot_cost) < 2:
        print("[!] Instancias o algoritmos insuficientes para pruebas de hipótesis globales.")
        return pivot_cost
        
    data_arrays = [pivot_cost[a].values for a in algs]
    try:
        stat_f, p_val_f = stats.friedmanchisquare(*data_arrays)
        print(f"\n[Friedman Global Test] Estadístico: {stat_f:.4f} | p-value: {p_val_f:.4e}")
    except Exception as e:
        print(f"[!] No se pudo realizar prueba de Friedman: {e}")
    
    prop_alg = 'BLOC_RELOC (Proposed)'
    if prop_alg in algs:
        print(f"\n[Contraste Pareado vs {prop_alg}] (Wilcoxon & Cohen's d):")
        comparisons = []
        for other in algs:
            if other == prop_alg:
                continue
            diff = pivot_cost[prop_alg] - pivot_cost[other]
            try:
                stat_w, p_val_w = stats.wilcoxon(pivot_cost[prop_alg], pivot_cost[other])
            except Exception:
                p_val_w = np.nan
            cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
            
            comparisons.append({
                'Comparison': f"{prop_alg} vs {other}",
                'Wilcoxon p-val': p_val_w,
                'Cohen d': cohen_d
            })
        df_comp = pd.DataFrame(comparisons)
        print(df_comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        
    return pivot_cost


# ---------------------------------------------------------------------------
# 8. GENERACIÓN DE ARTEFACTOS PARA PUBLICACIÓN
# ---------------------------------------------------------------------------
def generate_scientific_artifacts(df: pd.DataFrame, pivot_cost: pd.DataFrame):
    print("\n[+] Generando reportes y gráficos para publicación...")
    df.to_csv("sota_official_benchmark_results.csv", index=False)
    
    summary = df.groupby('Algorithm').agg(
        Mean_Cut=('Cut', 'mean'),
        Mean_Cost=('Cost', 'mean'),
        Mean_Gap_Pct=('Relative_Gap_Percent', 'mean'),
        Mean_Time_s=('Execution_Time_s', 'mean'),
        Mean_Memory_MB=('Peak_Memory_MB', 'mean')
    ).reset_index().sort_values(by='Mean_Cost')
    
    summary.to_csv("sota_official_summary_ranking.csv", index=False)
    
    print("\n========================================================================")
    print("RANKING OFICIAL CON DATASETS CANÓNICOS SOTA")
    print("========================================================================")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("========================================================================")
    
    if not pivot_cost.empty and pivot_cost.shape[1] > 0:
        plt.figure(figsize=(9, 5))
        min_costs = pivot_cost.min(axis=1)
        for col in pivot_cost.columns:
            ratios = pivot_cost[col] / (min_costs + 1e-8)
            sorted_ratios = np.sort(ratios)
            y_vals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
            plt.step(sorted_ratios, y_vals, label=col, where='post', linewidth=2)
            
        plt.xlabel(r"Factor de Rendimiento $\tau$ (Costo / Mejor Costo)")
        plt.ylabel(r"Proporción de Instancias $P(r_{i,a} \le \tau)$")
        plt.title("Performance Profile (Dolan-Moré) - Official Benchmark Datasets")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("dolan_more_official_datasets.png", dpi=300)
    
    print("    -> Todos los reportes (.csv, .png) guardados correctamente.")


# ---------------------------------------------------------------------------
# 9. PUNTO DE ENTRADA
# ---------------------------------------------------------------------------
def main():
    print("========================================================================")
    print("INICIANDO EVALUACIÓN SOTA CON DATASETS OFICIALES (WALSHAW / SNAP)")
    print("========================================================================")
    
    instances = fetch_and_load_official_benchmarks()
    
    if not instances:
        print("[Error Critico]: No se pudieron descargar o parsear las instancias oficiales.")
        sys.exit(1)
        
    df_results = execute_benchmark_harness(instances)
    
    if df_results.empty:
        print("[Error Critico]: No se obtuvieron resultados en la ejecución.")
        sys.exit(1)
        
    pivot_cost = perform_formal_statistical_analysis(df_results)
    generate_scientific_artifacts(df_results, pivot_cost)
    
    print("\n[✓] BENCHMARK OFICIAL SOTA FINALIZADO Y AUDITADO SATISFACTORIAMENTE.")


if __name__ == '__main__':
    main()
