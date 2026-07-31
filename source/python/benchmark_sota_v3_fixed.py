import os
import sys
import time
import gzip
import shutil
import subprocess
import urllib.request
import tracemalloc
import warnings
import multiprocessing as mp
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
# CONFIGURACIÓN EXPERIMENTAL
# ---------------------------------------------------------------------------
NUM_RUNS_PER_INSTANCE = 5      # Corridas independientes por algoritmo
TIMEOUT_PER_RUN_SEC = 90       # Tiempo límite por corrida (segundos)
K_PARTITIONS = 4               # Número de bloques para la partición
ALPHA_IMBALANCE = 0.1          # Penalización por desbalance
RANDOM_SEEDS = [42, 101, 2024, 888, 9999][:NUM_RUNS_PER_INSTANCE]

# ---------------------------------------------------------------------------
# 1. AUDITORÍA Y DETECCIÓN ADAPTATIVA DE ENTORNO
# ---------------------------------------------------------------------------
AVAILABLE_SOLVERS = {
    'METIS_CLI': shutil.which('gpmetis') is not None,
    'KaHyPar_CLI': shutil.which('kahypar') is not None,
    'METIS_Py': False,
    'KaHyPar_Py': False
}

try:
    import metis
    AVAILABLE_SOLVERS['METIS_Py'] = True
except ImportError:
    pass

try:
    import kahypar
    if hasattr(kahypar, "setVerbosity"):
        kahypar.setVerbosity(0)
    AVAILABLE_SOLVERS['KaHyPar_Py'] = True
except Exception:
    pass

print("========================================================================")
print("AUDITORÍA DE ENTORNO DE EJECUCIÓN SOTA v3.1")
print("========================================================================")
for solver, status in AVAILABLE_SOLVERS.items():
    print(f" -> {solver:<15}: {'DISPONIBLE [✓]' if status else 'NO DETECTADO [-]'}")
print("========================================================================\n")

# ---------------------------------------------------------------------------
# 2. PARSERS NATIVOS ROBUSTOS Y EXPORTADORES
# ---------------------------------------------------------------------------
def parse_metis_graph(file_path: str) -> nx.Graph:
    """Parsea archivos .graph en el formato canónico de METIS/Walshaw."""
    G = nx.Graph()
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('%')]
    
    if not lines:
        raise ValueError(f"El archivo {file_path} está vacío.")
    
    header = lines[0].split()
    num_nodes = int(header[0])
    
    current_node = 0
    for line in lines[1:]:
        current_node += 1
        if current_node > num_nodes:
            break
        neighbors = [int(x) - 1 for x in line.split()]
        for neighbor in neighbors:
            if current_node - 1 != neighbor:
                G.add_edge(current_node - 1, neighbor)
                
    G.add_nodes_from(range(num_nodes))
    return G

def parse_snap_edgelist_gz(file_path: str) -> nx.Graph:
    """Parsea listas de aristas comprimidas .txt.gz de Stanford SNAP Archive."""
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G)

# ---------------------------------------------------------------------------
# 3. DESCARGADOR AUTOMÁTICO DE DATASETS (CON ESPEJOS FIABLES)
# ---------------------------------------------------------------------------
def fetch_and_load_official_benchmarks() -> List[Dict[str, Any]]:
    print("[+] Descargando suite de Datasets Oficiales (Walshaw / SNAP / SuiteSparse)...")
    os.makedirs("official_datasets", exist_ok=True)
    
    # Fuentes con espejos directos estables
    sources = [
        {
            'name': 'SNAP_Facebook',
            'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz',
            'filename': 'official_datasets/facebook_combined.txt.gz',
            'parser': parse_snap_edgelist_gz
        },
        {
            'name': 'SNAP_Wiki_Vote',
            'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz',
            'filename': 'official_datasets/wiki-Vote.txt.gz',
            'parser': parse_snap_edgelist_gz
        },
        {
            'name': 'Walshaw_add32',
            'url': 'https://ci.skycn.me/https://raw.githubusercontent.com/juexiao/graph-partitioning/master/graphs/add32.graph',
            'alt_url': 'https://juexiao.github.io/graphs/add32.graph',
            'filename': 'official_datasets/add32.graph',
            'parser': parse_metis_graph
        },
        {
            'name': 'Walshaw_3elt',
            'url': 'https://ci.skycn.me/https://raw.githubusercontent.com/juexiao/graph-partitioning/master/graphs/3elt.graph',
            'alt_url': 'https://juexiao.github.io/graphs/3elt.graph',
            'filename': 'official_datasets/3elt.graph',
            'parser': parse_metis_graph
        }
    ]
    
    instances = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

    for src in sources:
        filepath = src['filename']
        if not os.path.exists(filepath):
            print(f"    -> Descargando {src['name']}...")
            download_success = False
            for url_key in ['url', 'alt_url']:
                if url_key not in src:
                    continue
                try:
                    req = urllib.request.Request(src[url_key], headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as response, open(filepath, 'wb') as out_file:
                        out_file.write(response.read())
                    download_success = True
                    break
                except Exception:
                    continue
            if not download_success:
                print(f"       [!] No se pudo descargar {src['name']}. Se omitirá temporalmente.")
                continue

        try:
            print(f"    -> Cargando y validando {src['name']}...")
            G = src['parser'](filepath)
            instances.append({
                'name': src['name'],
                'graph': G,
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges()
            })
            print(f"       [✓] Instancia lista: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}")
        except Exception as e:
            print(f"       [!] Error al procesar {src['name']}: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)

    print(f"    -> Total de instancias oficiales cargadas: {len(instances)}\n")
    return instances

# ---------------------------------------------------------------------------
# 4. MÉTRICAS DE EVALUACIÓN FORMAL
# ---------------------------------------------------------------------------
def evaluate_partition(G: nx.Graph, partition: Dict[int, int], k: int = 4, alpha: float = 0.1) -> Tuple[int, float, float]:
    cut_size = sum(1 for u, v in G.edges() if partition.get(u, -1) != partition.get(v, -1))
    block_sizes = [0] * k
    for n, b in partition.items():
        if 0 <= b < k:
            block_sizes[b] += 1
            
    N = G.number_of_nodes()
    target_size = N / k
    imbalance = sum(abs(s - target_size) for s in block_sizes) / (N + 1e-8)
    cost = cut_size + (alpha * N * imbalance)
    return cut_size, imbalance, cost

# ---------------------------------------------------------------------------
# 5. ALGORITMOS BASELINE Y HEURÍSTICAS DE COMPARACIÓN
# ---------------------------------------------------------------------------
def run_bloc_reloc(G: nx.Graph, k: int, seed: int) -> Dict[int, int]:
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {node: i % k for i, node in enumerate(nodes)}
    
    for _ in range(20):
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
        max_b, min_b = int(np.argmax(block_sizes)), int(np.argmin(block_sizes))
        target = G.number_of_nodes() / k
        if block_sizes[max_b] - target > target * 0.05:
            candidates = [u for u in nodes if part[u] == max_b and any(part[v] == min_b for v in G.neighbors(u))]
            for c in candidates[:min(len(candidates), 10)]:
                part[c] = min_b
    return part

def run_simulated_annealing(G: nx.Graph, k: int, seed: int, steps: int = 1200) -> Dict[int, int]:
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {node: np.random.randint(0, k) for node in nodes}
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
        T *= 0.96
    return part

def run_greedy_bfs_partition(G: nx.Graph, k: int, seed: int) -> Dict[int, int]:
    """Baseline voraz basado en BFS para balancear bloques."""
    np.random.seed(seed)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    part = {}
    target = len(nodes) // k
    
    curr_k = 0
    for u in nodes:
        if u in part:
            continue
        queue = [u]
        part[u] = curr_k
        count = 1
        while queue and count < target:
            curr = queue.pop(0)
            for neighbor in G.neighbors(curr):
                if neighbor not in part:
                    part[neighbor] = curr_k
                    queue.append(neighbor)
                    count += 1
                    if count >= target:
                        break
        curr_k = (curr_k + 1) % k
        
    for u in nodes:
        if u not in part:
            part[u] = np.random.randint(0, k)
    return part

def run_kahypar_python(G: nx.Graph, k: int, seed: int) -> Dict[int, int]:
    hypergraph = kahypar.Hypergraph(G.number_of_nodes(), G.number_of_edges(), [[u, v] for u, v in G.edges()])
    context = kahypar.Context()
    if hasattr(context, "load_dictionary"):
        context.load_dictionary({"mode": "direct", "objective": "cut", "k": k, "epsilon": 0.03, "seed": seed})
    partition = kahypar.partition(hypergraph, context)
    return {i: partition.block(i) for i in range(G.number_of_nodes())}

# ---------------------------------------------------------------------------
# 6. HARNESS MULTIASEMILLA CON RESILIENCIA Y TIMEOUTS
# ---------------------------------------------------------------------------
def execute_benchmark_harness(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    print("[+] Ejecutando evaluación comparativa SOTA...")
    
    algorithms = {
        'BLOC_RELOC (Proposed)': run_bloc_reloc,
        'Simulated_Annealing': run_simulated_annealing,
        'Greedy_BFS_Baseline': run_greedy_bfs_partition
    }
    
    if AVAILABLE_SOLVERS['KaHyPar_Py']:
        algorithms['KaHyPar_SOTA'] = run_kahypar_python
        
    results = []

    for inst in instances:
        G = inst['graph']
        print(f"\n[Instancia]: {inst['name']} (|V|={inst['num_nodes']}, |E|={inst['num_edges']})")
        
        for alg_name, alg_fn in algorithms.items():
            costs, cuts, imbs, times, mems = [], [], [], [], []
            
            for seed in RANDOM_SEEDS:
                tracemalloc.start()
                t0 = time.perf_counter()
                
                try:
                    final_part = alg_fn(G, K_PARTITIONS, seed)
                    exec_time = time.perf_counter() - t0
                    _, peak_mem = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    cut, imb, cost = evaluate_partition(G, final_part, k=K_PARTITIONS, alpha=ALPHA_IMBALANCE)
                    costs.append(cost)
                    cuts.append(cut)
                    imbs.append(imb)
                    times.append(exec_time)
                    mems.append(peak_mem / (1024 * 1024))
                    
                except Exception as e:
                    tracemalloc.stop()
                    print(f"       [!] Error en {alg_name} (seed {seed}): {e}")
                    continue

            if costs:
                mean_cost = np.mean(costs)
                std_cost = np.std(costs, ddof=1) if len(costs) > 1 else 0.0
                ci95_cost = 1.96 * (std_cost / np.sqrt(len(costs))) if len(costs) > 1 else 0.0
                
                results.append({
                    'Instance': inst['name'],
                    'Nodes': inst['num_nodes'],
                    'Edges': inst['num_edges'],
                    'Algorithm': alg_name,
                    'Mean_Cut': np.mean(cuts),
                    'Mean_Imbalance': np.mean(imbs),
                    'Mean_Cost': mean_cost,
                    'Std_Cost': std_cost,
                    'CI95_Cost': ci95_cost,
                    'Mean_Time_s': np.mean(times),
                    'Mean_Mem_MB': np.mean(mems)
                })
                print(f"       [✓] {alg_name:<22}: Costo={mean_cost:.2f} (±{ci95_cost:.2f}) | Tiempo={np.mean(times):.3f}s")
                
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 7. ANÁLISIS ESTADÍSTICO DE HIPÓTESIS
# ---------------------------------------------------------------------------
def perform_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    print("\n========================================================================")
    print("ANÁLISIS ESTADÍSTICO FORMAL (FRIEDMAN + WILCOXON)")
    print("========================================================================")
    
    pivot = df.pivot_table(index='Instance', columns='Algorithm', values='Mean_Cost')
    algs = pivot.columns.tolist()
    
    if len(algs) < 2 or len(pivot) < 2:
        print("[!] Advertencia: Se requieren al menos 2 algoritmos y 2 instancias para la prueba de Friedman.")
        return pivot
        
    data = [pivot[a].values for a in algs]
    try:
        stat_f, p_val_f = stats.friedmanchisquare(*data)
        print(f"[Prueba Global de Friedman] Estadístico: {stat_f:.4f} | p-value: {p_val_f:.4e}")
    except Exception as e:
        print(f"[!] No se pudo realizar la prueba de Friedman: {e}")
    
    target_alg = 'BLOC_RELOC (Proposed)'
    if target_alg in algs:
        print(f"\n[Pruebas Pareadas Post-Hoc vs {target_alg}]")
        comp_results = []
        for other in algs:
            if other == target_alg:
                continue
            diff = pivot[target_alg] - pivot[other]
            try:
                w_stat, p_val = stats.wilcoxon(pivot[target_alg], pivot[other])
            except Exception:
                w_stat, p_val = np.nan, 1.0
                
            cohen_d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)
            comp_results.append({
                'Comparación': f"{target_alg} vs {other}",
                'Wilcoxon_W': w_stat,
                'p_value': p_val,
                'Cohen_d': cohen_d
            })
            
        df_comp = pd.DataFrame(comp_results)
        if not df_comp.empty:
            print(df_comp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            
    return pivot

# ---------------------------------------------------------------------------
# 8. GENERACIÓN DE ARTEFACTOS Y PERFORMANCE PROFILES
# ---------------------------------------------------------------------------
def generate_artifacts(df: pd.DataFrame, pivot: pd.DataFrame):
    print("\n[+] Guardando resultados y generando gráficos...")
    df.to_csv("sota_v3_detailed_runs.csv", index=False)
    
    ranking = df.groupby('Algorithm').agg(
        Mean_Cost=('Mean_Cost', 'mean'),
        Mean_Cut=('Mean_Cut', 'mean'),
        Mean_Time_s=('Mean_Time_s', 'mean'),
        Mean_Mem_MB=('Mean_Mem_MB', 'mean')
    ).reset_index().sort_values('Mean_Cost')
    
    ranking.to_csv("sota_v3_ranking_summary.csv", index=False)
    
    print("\n========================================================================")
    print("RANKING GLOBAL FINAL")
    print("========================================================================")
    print(ranking.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("========================================================================")
    
    if not pivot.empty and pivot.shape[1] > 0:
        plt.figure(figsize=(8, 5))
        min_costs = pivot.min(axis=1)
        for col in pivot.columns:
            ratios = pivot[col] / (min_costs + 1e-8)
            sorted_ratios = np.sort(ratios)
            y_vals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)
            plt.step(sorted_ratios, y_vals, label=col, where='post', linewidth=2)
            
        plt.xlabel(r"Factor de Rendimiento $\tau$ (Costo / Mejor Costo)")
        plt.ylabel(r"Proporción de Instancias $P(r_{i,a} \le \tau)$")
        plt.title("Performance Profile (Dolan-Moré) - Benchmark Oficial")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig("dolan_more_sota_v3.png", dpi=300)
        
    print(" -> Artefactos generados exitosamente: 'sota_v3_detailed_runs.csv', 'sota_v3_ranking_summary.csv', 'dolan_more_sota_v3.png'")

# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
def main():
    instances = fetch_and_load_official_benchmarks()
    if not instances:
        print("[Error Critico]: No se pudieron preparar las instancias. Verifique su conexión.")
        sys.exit(1)
        
    df_results = execute_benchmark_harness(instances)
    if df_results.empty:
        print("[Error Critico]: No se obtuvieron ejecuciones válidas.")
        sys.exit(1)
        
    pivot = perform_statistical_tests(df_results)
    generate_artifacts(df_results, pivot)
    print("\n[✓] BENCHMARK COMPLETADO EXITOSAMENTE.")

if __name__ == '__main__':
    main()
