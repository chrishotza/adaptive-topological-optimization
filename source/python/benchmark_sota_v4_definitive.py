import os
import sys
import time
import gzip
import shutil
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
# CONFIGURACIÓN EXPERIMENTAL SOTA
# ---------------------------------------------------------------------------
NUM_RUNS_PER_INSTANCE = 5      # Corridas independientes
TIMEOUT_PER_RUN_SEC = 60       # Timeout ESTRICTO por proceso
K_PARTITIONS = 4               # Bloques
ALPHA_IMBALANCE = 0.1          # Penalización
RANDOM_SEEDS = [42, 101, 2024, 888, 9999][:NUM_RUNS_PER_INSTANCE]

# ---------------------------------------------------------------------------
# 1. AUDITORÍA DE ENTORNO
# ---------------------------------------------------------------------------
AVAILABLE_SOLVERS = {
    'KaHyPar_Py': False,
    'METIS_Py': False
}
try:
    import kahypar
    if hasattr(kahypar, "setVerbosity"): kahypar.setVerbosity(0)
    AVAILABLE_SOLVERS['KaHyPar_Py'] = True
except Exception: pass
try:
    import metis
    AVAILABLE_SOLVERS['METIS_Py'] = True
except Exception: pass

print("========================================================================")
print("AUDITORÍA SOTA v4: RENDIMIENTO EXTREMO Y TIMEOUTS AISLADOS")
print("========================================================================")
for solver, status in AVAILABLE_SOLVERS.items():
    print(f" -> {solver:<15}: {'DISPONIBLE [✓]' if status else 'NO DETECTADO [-]'}")
print("========================================================================\n")

# ---------------------------------------------------------------------------
# 2. GESTIÓN ROBUSTA DE DATASETS (CON FALLBACK SINTÉTICO)
# ---------------------------------------------------------------------------
def parse_snap_edgelist_gz(file_path: str) -> nx.Graph:
    G = nx.Graph()
    with gzip.open(file_path, 'rt') as f:
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            u, v = map(int, line.strip().split()[:2])
            if u != v: G.add_edge(u, v)
    return nx.convert_node_labels_to_integers(G)

def fetch_datasets() -> List[Dict[str, Any]]:
    print("[+] Preparando suite de Datasets (con fallback sintético)...")
    os.makedirs("official_datasets", exist_ok=True)
    instances = []
    
    sources = [
        {'name': 'SNAP_Facebook', 'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz'},
        {'name': 'SNAP_Wiki_Vote', 'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'}
    ]
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    for src in sources:
        filepath = f"official_datasets/{src['name']}.txt.gz"
        if not os.path.exists(filepath):
            try:
                req = urllib.request.Request(src['url'], headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp, open(filepath, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                pass
        
        if os.path.exists(filepath):
            print(f"    -> Cargando {src['name']}...")
            G = parse_snap_edgelist_gz(filepath)
            instances.append({'name': src['name'], 'graph': G, 'nodes': G.number_of_nodes(), 'edges': G.number_of_edges()})
    
    # FALLBACK AUTOMÁTICO: Si no hay suficientes instancias, generar modelos canónicos
    if len(instances) < 4:
        print("    -> Generando datasets sintéticos SOTA para asegurar rigor estadístico...")
        # 1. Barabási-Albert (Scale-Free)
        G_ba = nx.barabasi_albert_graph(3000, 15, seed=42)
        instances.append({'name': 'Synth_ScaleFree_BA', 'graph': G_ba, 'nodes': G_ba.number_of_nodes(), 'edges': G_ba.number_of_edges()})
        
        # 2. Random Partition Graph (Planted Partition Model)
        G_pp = nx.random_partition_graph([1000, 1000, 1000, 1000], 0.05, 0.005, seed=42)
        instances.append({'name': 'Synth_PlantedPartition', 'graph': G_pp, 'nodes': G_pp.number_of_nodes(), 'edges': G_pp.number_of_edges()})

    for inst in instances:
        print(f"       [✓] Lista: {inst['name']} (|V|={inst['nodes']}, |E|={inst['edges']})")
    return instances

# ---------------------------------------------------------------------------
# 3. ALGORITMOS OPTIMIZADOS (Cálculos de Delta Costo para evitar cuelgues)
# ---------------------------------------------------------------------------
def run_bloc_reloc_fast(G: nx.Graph, k: int, seed: int) -> Dict[int, int]:
    """Versión optimizada con updates locales."""
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {node: i % k for i, node in enumerate(nodes)}
    
    for _ in range(10):  # Reducido para convergencia rápida
        for u in nodes:
            current_b = part[u]
            b_counts = {i: 0 for i in range(k)}
            for v in G.neighbors(u):
                b_counts[part[v]] += 1
            
            best_b = max(b_counts, key=b_counts.get)
            if b_counts[best_b] > b_counts[current_b]:
                part[u] = best_b
    return part

def run_simulated_annealing_fast(G: nx.Graph, k: int, seed: int, steps: int = 2000) -> Dict[int, int]:
    """SA con actualización de Delta de Costo O(grado) en lugar de O(|E|)."""
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {node: np.random.randint(0, k) for node in nodes}
    
    block_sizes = [0] * k
    for b in part.values():
        block_sizes[b] += 1
        
    T = 5.0
    for _ in range(steps):
        u = np.random.choice(nodes)
        old_b = part[u]
        new_b = np.random.randint(0, k)
        if old_b == new_b: continue
        
        # Delta Cut (Solo miramos vecinos, MUCHO más rápido)
        delta_cut = 0
        for v in G.neighbors(u):
            v_b = part[v]
            if v_b == old_b: delta_cut += 1   # Rompemos arista interna
            elif v_b == new_b: delta_cut -= 1 # Creamos arista interna
            
        # Delta Imbalance (Aproximado)
        target = len(nodes) / k
        old_imb = abs(block_sizes[old_b] - target) + abs(block_sizes[new_b] - target)
        new_imb = abs((block_sizes[old_b]-1) - target) + abs((block_sizes[new_b]+1) - target)
        delta_imb = (new_imb - old_imb) / len(nodes)
        
        delta_cost = delta_cut + (ALPHA_IMBALANCE * len(nodes) * delta_imb)
        
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(T, 1e-5)):
            part[u] = new_b
            block_sizes[old_b] -= 1
            block_sizes[new_b] += 1
        T *= 0.98
    return part

def run_kahypar_python(G: nx.Graph, k: int, seed: int) -> Dict[int, int]:
    hypergraph = kahypar.Hypergraph(G.number_of_nodes(), G.number_of_edges(), [[u, v] for u, v in G.edges()])
    context = kahypar.Context()
    if hasattr(context, "load_dictionary"):
        context.load_dictionary({"mode": "direct", "objective": "cut", "k": k, "epsilon": 0.03, "seed": seed})
    partition = kahypar.partition(hypergraph, context)
    return {i: partition.block(i) for i in range(G.number_of_nodes())}

# ---------------------------------------------------------------------------
# 4. AISLAMIENTO DE PROCESOS (TIMEOUT REAL)
# ---------------------------------------------------------------------------
def worker_wrapper(func, G, k, seed, queue):
    """Ejecuta el algoritmo y devuelve el resultado a través del Queue."""
    try:
        t0 = time.perf_counter()
        tracemalloc.start()
        part = func(G, k, seed)
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        exec_time = time.perf_counter() - t0
        queue.put(('SUCCESS', part, exec_time, peak_mem / (1024*1024)))
    except Exception as e:
        queue.put(('ERROR', str(e), 0, 0))

def run_with_timeout(func, G, k, seed, timeout):
    """Fuerza la detención si el proceso excede el límite de tiempo."""
    queue = mp.Queue()
    p = mp.Process(target=worker_wrapper, args=(func, G, k, seed, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return 'TIMEOUT', None, timeout, 0.0
        
    if not queue.empty():
        status, part_or_err, ex_time, mem = queue.get()
        return status, part_or_err, ex_time, mem
    return 'ERROR', "Queue Empty", 0.0, 0.0

def evaluate_final(G: nx.Graph, partition: Dict[int, int], k: int) -> Tuple[int, float, float]:
    cut_size = sum(1 for u, v in G.edges() if partition.get(u, -1) != partition.get(v, -1))
    block_sizes = [0] * k
    for b in partition.values():
        if 0 <= b < k: block_sizes[b] += 1
    N = G.number_of_nodes()
    imb = sum(abs(s - (N/k)) for s in block_sizes) / N
    cost = cut_size + (ALPHA_IMBALANCE * N * imb)
    return cut_size, imb, cost

# ---------------------------------------------------------------------------
# 5. HARNESS PRINCIPAL
# ---------------------------------------------------------------------------
def execute_benchmark(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    print("\n[+] Iniciando Harness Multiproceso (Timeout: 60s/run)...")
    
    algorithms = {
        'BLOC_RELOC (Prop)': run_bloc_reloc_fast,
        'SA_Fast': run_simulated_annealing_fast
    }
    if AVAILABLE_SOLVERS['KaHyPar_Py']:
        algorithms['KaHyPar_SOTA'] = run_kahypar_python

    results = []
    
    for inst in instances:
        G = inst['graph']
        print(f"\n[Dataset]: {inst['name']} (|V|={inst['nodes']}, |E|={inst['edges']})")
        
        for alg_name, alg_fn in algorithms.items():
            costs, times, mems = [], [], []
            timeouts = 0
            
            for seed in RANDOM_SEEDS:
                status, part, ex_time, mem = run_with_timeout(alg_fn, G, K_PARTITIONS, seed, TIMEOUT_PER_RUN_SEC)
                
                if status == 'SUCCESS':
                    cut, imb, cost = evaluate_final(G, part, K_PARTITIONS)
                    costs.append(cost)
                    times.append(ex_time)
                    mems.append(mem)
                elif status == 'TIMEOUT':
                    timeouts += 1
                    
            if costs:
                mean_cost = np.mean(costs)
                ci95 = 1.96 * (np.std(costs, ddof=1) / np.sqrt(len(costs))) if len(costs)>1 else 0.0
                results.append({
                    'Instance': inst['name'], 'Algorithm': alg_name,
                    'Mean_Cost': mean_cost, 'CI95_Cost': ci95,
                    'Mean_Time_s': np.mean(times), 'Timeouts': timeouts
                })
                print(f"    -> {alg_name:<16}: Costo={mean_cost:.2f} (±{ci95:.2f}) | Tiempo={np.mean(times):.2f}s | T/O: {timeouts}")
            else:
                print(f"    -> {alg_name:<16}: [FAILED / TIMEOUT ALL RUNS]")
                
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# MAIN Y EXPORTACIÓN
# ---------------------------------------------------------------------------
def main():
    mp.set_start_method('spawn', force=True)
    instances = fetch_datasets()
    if not instances:
        sys.exit(1)
        
    df = execute_benchmark(instances)
    if not df.empty:
        df.to_csv("sota_v4_results.csv", index=False)
        print("\n========================================================================")
        print("RANKING FINAL DE COSTOS PROMEDIO")
        print("========================================================================")
        ranking = df.groupby('Algorithm')['Mean_Cost'].mean().sort_values()
        print(ranking.to_string())
        print("========================================================================")
        print("[✓] COMPLETO. Resultados exportados a 'sota_v4_results.csv'.")

if __name__ == '__main__':
    main()
