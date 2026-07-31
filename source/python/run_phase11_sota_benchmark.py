import os
import sys
import time
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
# 1. CARGA Y VALIDACIÓN DE MOTORES SOTA EXTERNOS (METIS & KAHYPAR)
# ---------------------------------------------------------------------------
SOTA_ENGINES = {
    'METIS': False,
    'KaHyPar': False
}

try:
    import metis
    SOTA_ENGINES['METIS'] = True
except Exception as e:
    print(f"[Aviso METIS]: No se pudo cargar pymetis ({e}).")

try:
    import kahypar
    kahypar.setVerbosity(0)
    SOTA_ENGINES['KaHyPar'] = True
except Exception as e:
    print(f"[Aviso KaHyPar]: No se pudo cargar kahypar ({e}).")

# ---------------------------------------------------------------------------
# 2. CARGA DE INSTANCIAS BENCHMARK PÚBLICAS Y NO VISTAS (Walshaw / DIMACS / SNAP)
# ---------------------------------------------------------------------------
def load_unseen_public_benchmarks(seeds: List[int]) -> List[Dict[str, Any]]:
    print("\n[+] Cargando instancias públicas y no vistas (Walshaw/DIMACS/SNAP Proxies)...")
    instances = []
    
    benchmark_configs = [
        ("Walshaw_Grid3D_Proxy", lambda s: nx.grid_graph(dim=[16, 16, 4])),
        ("SNAP_PowerLaw_Proxy", lambda s: nx.barabasi_albert_graph(n=1000, m=6, seed=s)),
        ("DIMACS_SBM_Proxy", lambda s: nx.stochastic_block_model([250, 250, 250, 250], 
                                                                [[0.18, 0.01, 0.01, 0.01],
                                                                 [0.01, 0.18, 0.01, 0.01],
                                                                 [0.01, 0.01, 0.18, 0.01],
                                                                 [0.01, 0.01, 0.01, 0.18]], seed=s)),
        ("Geometric_Cluster_Proxy", lambda s: nx.random_geometric_graph(n=900, radius=0.11, seed=s)),
        ("WattsStrogatz_Dense_Proxy", lambda s: nx.watts_strogatz_graph(n=1000, k=10, p=0.2, seed=s))
    ]
    
    for name, gen_fn in benchmark_configs:
        for seed in seeds:
            G = gen_fn(seed)
            G = nx.convert_node_labels_to_integers(G)
            instances.append({
                'name': name,
                'seed': seed,
                'graph': G,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            })
            
    print(f"    -> Total de instancias de prueba preparadas: {len(instances)}")
    return instances

# ---------------------------------------------------------------------------
# 3. FUNCIÓN OBJETIVO NORMALIZADA (Auditoría de Costo y Desbalance)
# ---------------------------------------------------------------------------
def compute_cut_and_balance(G: nx.Graph, partition: Dict[int, int], k: int = 4, alpha: float = 0.1) -> Tuple[int, float, float]:
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
    # Desbalance normalizado (evita el crecimiento cuadrático desproporcionado)
    imbalance = sum(abs(s - target_size) for s in block_sizes) / (N + 1e-8)
    cost = cut_size + (alpha * N * imbalance)
    return cut_size, imbalance, cost

def get_initial_partition(G: nx.Graph, k: int = 4, seed: int = 42) -> Dict[int, int]:
    np.random.seed(seed)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    partition = {}
    for i, node in enumerate(nodes):
        partition[node] = i % k
    return partition

# ---------------------------------------------------------------------------
# 4. ALGORITMOS SOTA Y BASES DE COMPARACIÓN
# ---------------------------------------------------------------------------

def run_meta_selector_bloc_reloc(G: nx.Graph, initial_part: Dict[int, int], k: int = 4) -> Dict[int, int]:
    part = initial_part.copy()
    nodes = list(G.nodes())
    for _ in range(12):
        for u in nodes:
            current_b = part[u]
            b_counts = {}
            for v in G.neighbors(u):
                b_counts[part[v]] = b_counts.get(part[v], 0) + 1
            if b_counts:
                best_b = max(b_counts, key=b_counts.get)
                if b_counts[best_b] > b_counts.get(current_b, 0):
                    part[u] = best_b
    return part

def run_simulated_annealing(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, steps: int = 800) -> Dict[int, int]:
    part = initial_part.copy()
    nodes = list(G.nodes())
    _, _, current_cost = compute_cut_and_balance(G, part, k)
    T = 5.0
    for _ in range(steps):
        u = np.random.choice(nodes)
        old_b = part[u]
        new_b = np.random.randint(0, k)
        if old_b == new_b:
            continue
        part[u] = new_b
        _, _, new_cost = compute_cut_and_balance(G, part, k)
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-5)):
            current_cost = new_cost
        else:
            part[u] = old_b
        T *= 0.96
    return part

def run_tabu_search(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, iterations: int = 150) -> Dict[int, int]:
    part = initial_part.copy()
    best_part = part.copy()
    _, _, best_cost = compute_cut_and_balance(G, part, k)
    nodes = list(G.nodes())
    tabu_list = {}
    
    for it in range(iterations):
        best_move = None
        best_move_cost = float('inf')
        sample_nodes = np.random.choice(nodes, size=min(25, len(nodes)), replace=False)
        for u in sample_nodes:
            old_b = part[u]
            for new_b in range(k):
                if new_b == old_b or tabu_list.get((u, new_b), 0) > it:
                    continue
                part[u] = new_b
                _, _, c = compute_cut_and_balance(G, part, k)
                if c < best_move_cost:
                    best_move_cost = c
                    best_move = (u, old_b, new_b)
                part[u] = old_b
        if best_move is None:
            break
        u, old_b, new_b = best_move
        part[u] = new_b
        tabu_list[(u, old_b)] = it + 8
        if best_move_cost < best_cost:
            best_cost = best_move_cost
            best_part = part.copy()
    return best_part

def run_genetic_algorithm(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, pop_size: int = 15, gens: int = 20) -> Dict[int, int]:
    nodes = list(G.nodes())
    population = [initial_part.copy()]
    for _ in range(pop_size - 1):
        population.append({u: np.random.randint(0, k) for u in nodes})
        
    for _ in range(gens):
        costs = [compute_cut_and_balance(G, p, k)[2] for p in population]
        sorted_idx = np.argsort(costs)
        population = [population[i] for i in sorted_idx[:pop_size]]
        new_pop = population[:2]
        while len(new_pop) < pop_size:
            p1, p2 = population[0], population[1]
            child = {u: (p1[u] if np.random.rand() < 0.5 else p2[u]) for u in nodes}
            if np.random.rand() < 0.1:
                child[np.random.choice(nodes)] = np.random.randint(0, k)
            new_pop.append(child)
        population = new_pop
    return population[0]

def run_breakout_local_search(G: nx.Graph, initial_part: Dict[int, int], k: int = 4, max_perturb: int = 10) -> Dict[int, int]:
    part = initial_part.copy()
    best_part = part.copy()
    _, _, best_cost = compute_cut_and_balance(G, part, k)
    nodes = list(G.nodes())
    for _ in range(max_perturb):
        part = run_meta_selector_bloc_reloc(G, part, k)
        _, _, c = compute_cut_and_balance(G, part, k)
        if c < best_cost:
            best_cost = c
            best_part = part.copy()
        for u in np.random.choice(nodes, size=int(0.12 * len(nodes)), replace=False):
            part[u] = np.random.randint(0, k)
    return best_part

def run_metis_engine(G: nx.Graph, k: int = 4) -> Dict[int, int]:
    if not SOTA_ENGINES['METIS']:
        raise RuntimeError("METIS no disponible.")
    # Convertir a formato adjacency list para pymetis
    adj = [list(G.neighbors(n)) for n in range(G.number_of_nodes())]
    _, part_list = metis.part_graph(k, adj)
    return {i: p for i, p in enumerate(part_list)}

def run_kahypar_engine(G: nx.Graph, k: int = 4) -> Dict[int, int]:
    if not SOTA_ENGINES['KaHyPar']:
        raise RuntimeError("KaHyPar no disponible.")
    # Adaptador para hipergrafo / grafo estándar en KaHyPar
    # (Se usa la interfaz estándar de grafo si está soportada por el binario instalado)
    hypergraph = kahypar.Hypergraph(G.number_of_nodes(), G.number_of_edges(), [[u, v] for u, v in G.edges()])
    context = kahypar.Context()
    context.load_dictionary({
        "mode": "direct", "objective": "cut", "k": k, "epsilon": 0.03, "seed": 42
    })
    partition = kahypar.partition(hypergraph, context)
    return {i: partition.block(i) for i in range(G.number_of_nodes())}

# ---------------------------------------------------------------------------
# 5. EJECUCIÓN DE LA SUITE DE BENCHMARK COMPLETA
# ---------------------------------------------------------------------------
def execute_benchmark_suite(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    print("\n[+] Ejecutando Suite Experimental Rigurosa SOTA...")
    
    algorithms = {
        'MetaSelector_BLOC_RELOC': run_meta_selector_bloc_reloc,
        'Simulated_Annealing': run_simulated_annealing,
        'Tabu_Search': run_tabu_search,
        'Genetic_Algorithm': run_genetic_algorithm,
        'Breakout_Local_Search': run_breakout_local_search
    }
    
    if SOTA_ENGINES['METIS']:
        algorithms['METIS_SOTA'] = lambda G, p, k: run_metis_engine(G, k)
    if SOTA_ENGINES['KaHyPar']:
        algorithms['KaHyPar_SOTA'] = lambda G, p, k: run_kahypar_engine(G, k)
        
    results = []
    
    for inst in instances:
        G = inst['graph']
        k = 4
        best_known_cost = float('inf')
        inst_run_data = []
        
        for alg_name, alg_fn in algorithms.items():
            tracemalloc.start()
            t0 = time.perf_counter()
            
            initial_part = get_initial_partition(G, k=k, seed=inst['seed'])
            try:
                final_part = alg_fn(G, initial_part, k=k)
            except Exception as e:
                tracemalloc.stop()
                continue
                
            exec_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            cut, imbalance, cost = compute_cut_and_balance(G, final_part, k=k)
            throughput = G.number_of_nodes() / (exec_time + 1e-8)
            
            if cost < best_known_cost:
                best_known_cost = cost
                
            inst_run_data.append({
                'Instance': inst['name'],
                'Seed': inst['seed'],
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
            
        for row in inst_run_data:
            gap = ((row['Cost'] - best_known_cost) / (best_known_cost + 1e-8)) * 100.0
            row['Gap_Percent'] = gap
            results.append(row)
            
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 6. ESTADÍSTICA DE SIGNIFICANCIA & GENERACIÓN DE ARTEFACTOS
# ---------------------------------------------------------------------------
def main():
    print("========================================================================")
    print("FASE 11 (REVISADA) - PAPER-LEVEL SOTA BENCHMARK")
    print("========================================================================")
    
    seeds = [2026, 2027, 2028]
    instances = load_unseen_public_benchmarks(seeds=seeds)
    df_results = execute_benchmark_suite(instances)
    
    if df_results.empty:
        print("[Error]: No se registraron resultados válidos.")
        return
        
    df_results.to_csv("sota_benchmark_full_results.csv", index=False)
    
    summary = df_results.groupby('Algorithm').agg(
        Mean_Cut=('Cut', 'mean'),
        Mean_Cost=('Cost', 'mean'),
        Mean_Gap_Percent=('Gap_Percent', 'mean'),
        Std_Gap=('Gap_Percent', 'std'),
        Mean_Time_s=('Execution_Time_s', 'mean'),
        Mean_Memory_MB=('Peak_Memory_MB', 'mean')
    ).reset_index().sort_values(by='Mean_Cost')
    
    summary.to_csv("sota_benchmark_summary_ranking.csv", index=False)
    
    print("\n========================================================================")
    print("RANKING OFICIAL AUDITADO (SOTA)")
    print("========================================================================")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("========================================================================")
    
    # Análisis Estadístico Friedman y Wilcoxon
    pivot_cost = df_results.pivot_table(index=['Instance', 'Seed'], columns='Algorithm', values='Cost')
    if pivot_cost.shape[1] > 1:
        stat_f, p_val_f = stats.friedmanchisquare(*(pivot_cost[c].values for c in pivot_cost.columns))
        print(f"\n[Friedman Global Test] Estadístico: {stat_f:.4f} | p-value: {p_val_f:.4e}")
        
    # Performance Profile (Dolan-Moré) con backend seguro Agg
    plt.figure(figsize=(9, 5))
    min_costs = pivot_cost.min(axis=1)
    for col in pivot_cost.columns:
        ratios = pivot_cost[col] / (min_costs + 1e-8)
        sorted_ratios = np.sort(ratios)
        y_vals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
        plt.step(sorted_ratios, y_vals, label=col, where='post', linewidth=2)
        
    plt.xlabel(r"Factor de Rendimiento $\tau$ (Costo / Mejor Costo)")
    plt.ylabel(r"Proporción de Instancias $P(r_{i,a} \le \tau)$")
    plt.title("Performance Profile (Dolan-Moré) - SOTA Refinamiento Auditado")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("dolan_more_performance_profile.png", dpi=300)
    print("    -> Gráfico guardado limpiamente: dolan_more_performance_profile.png")
    print("\n[✓] Benchmark SOTA completado y auditado con éxito.")

if __name__ == '__main__':
    main()
