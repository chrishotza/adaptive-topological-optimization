import os
import sys
import time
import json
import platform
import urllib.request
import gzip
import shutil
import warnings
import multiprocessing as mp
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings('ignore')

NUM_SEEDS = 10
TIMEOUT_SEC = 45
K_PARTITIONS = 4
ALPHA_IMBALANCE = 0.1
REPORT_DIR = "MASTER_VALIDATION_REPORT"
SEEDS = [42, 101, 2024, 888, 9999, 1337, 7, 314, 271, 555][:NUM_SEEDS]

def audit_environment():
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": mp.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "networkx_version": nx.__version__,
        "seeds": SEEDS
    }

def fetch_and_prepare_benchmarks():
    os.makedirs("raw_data", exist_ok=True)
    instances = []
    snap_sources = [
        {'name': 'SNAP_Facebook', 'url': 'https://snap.stanford.edu/data/facebook_combined.txt.gz'},
        {'name': 'SNAP_WikiVote', 'url': 'https://snap.stanford.edu/data/wiki-Vote.txt.gz'}
    ]
    headers = {'User-Agent': 'Mozilla/5.0'}
    for src in snap_sources:
        path = f"raw_data/{src['name']}.txt.gz"
        if not os.path.exists(path):
            try:
                req = urllib.request.Request(src['url'], headers=headers)
                with urllib.request.urlopen(req, timeout=12) as resp, open(path, 'wb') as f:
                    f.write(resp.read())
            except Exception:
                pass
        if os.path.exists(path):
            try:
                G = nx.Graph()
                with gzip.open(path, 'rt') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip(): continue
                        u, v = map(int, line.strip().split()[:2])
                        if u != v: G.add_edge(u, v)
                G = nx.convert_node_labels_to_integers(G)
                instances.append({'name': src['name'], 'graph': G, 'type': 'Real_SNAP'})
            except Exception:
                pass

    stress_graphs = [
        ('Stress_ScaleFree_BA', nx.barabasi_albert_graph(2000, 10, seed=42)),
        ('Stress_SmallWorld_WS', nx.watts_strogatz_graph(2000, 12, 0.1, seed=42)),
        ('Stress_Regular_RR', nx.random_regular_graph(16, 1500, seed=42)),
        ('Stress_PlantedPartition', nx.random_partition_graph([500, 500, 500, 500], 0.08, 0.002, seed=42)),
        ('Stress_Disconnected', nx.disjoint_union(nx.erdos_renyi_graph(500, 0.05, seed=42), nx.erdos_renyi_graph(500, 0.05, seed=84)))
    ]
    for name, G in stress_graphs:
        instances.append({'name': name, 'graph': G, 'type': 'Stress_Test'})
    return instances

def audit_solvers():
    solvers = {
        'METIS_CLI': shutil.which('gpmetis') is not None,
        'KaHyPar_CLI': shutil.which('kahypar') is not None,
        'KaFFPa_CLI': shutil.which('kaffpa') is not None,
        'SCIP_CLI': shutil.which('scip') is not None,
        'Gurobi': False,
        'KaHyPar_Py': False,
        'METIS_Py': False
    }
    try:
        import kahypar
        solvers['KaHyPar_Py'] = True
    except Exception: pass
    try:
        import metis
        solvers['METIS_Py'] = True
    except Exception: pass
    try:
        import gurobipy
        solvers['Gurobi'] = True
    except Exception: pass
    return solvers

def evaluate_solution(G, partition, k):
    cut_size = sum(1 for u, v in G.edges() if partition.get(u, -1) != partition.get(v, -1))
    block_sizes = [0] * k
    for b in partition.values():
        if 0 <= b < k: block_sizes[b] += 1
    N = G.number_of_nodes()
    target = N / k
    imbalance = sum(abs(s - target) for s in block_sizes) / N
    cost = cut_size + (ALPHA_IMBALANCE * N * imbalance)
    vol = {i: sum(G.degree(n) for n, b in partition.items() if b == i) for i in range(k)}
    total_vol = sum(vol.values()) or 1
    conductance = cut_size / total_vol
    return cut_size, imbalance, cost, conductance

def alg_random(G, k, seed):
    np.random.seed(seed)
    return {n: np.random.randint(0, k) for n in G.nodes()}

def alg_greedy_bfs(G, k, seed):
    np.random.seed(seed)
    nodes = list(G.nodes())
    np.random.shuffle(nodes)
    part = {}
    target = len(nodes) // k
    curr_k = 0
    for u in nodes:
        if u in part: continue
        queue = [u]
        part[u] = curr_k
        count = 1
        while queue and count < target:
            curr = queue.pop(0)
            for v in G.neighbors(curr):
                if v not in part:
                    part[v] = curr_k
                    queue.append(v)
                    count += 1
                    if count >= target: break
        curr_k = (curr_k + 1) % k
    for n in G.nodes():
        if n not in part: part[n] = np.random.randint(0, k)
    return part

def alg_spectral_approx(G, k, seed):
    try:
        L = nx.normalized_laplacian_matrix(G).astype(float)
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
        idx = 1 if len(eigenvalues) > 1 else 0
        coords = eigenvectors[:, idx]
        sorted_nodes = [node for _, node in sorted(zip(coords, G.nodes()))]
        chunk_size = len(sorted_nodes) // k
        part = {}
        for i, node in enumerate(sorted_nodes):
            block = min(i // chunk_size, k - 1)
            part[node] = block
        return part
    except Exception:
        return alg_greedy_bfs(G, k, seed)

def alg_simulated_annealing(G, k, seed):
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {n: np.random.randint(0, k) for n in nodes}
    block_sizes = [0] * k
    for b in part.values(): block_sizes[b] += 1
    T = 3.0
    for _ in range(800):
        u = np.random.choice(nodes)
        old_b = part[u]
        new_b = np.random.randint(0, k)
        if old_b == new_b: continue
        delta_cut = sum(1 if part[v] == old_b else (-1 if part[v] == new_b else 0) for v in G.neighbors(u))
        target = len(nodes) / k
        old_imb = abs(block_sizes[old_b] - target) + abs(block_sizes[new_b] - target)
        new_imb = abs((block_sizes[old_b]-1) - target) + abs((block_sizes[new_b]+1) - target)
        delta_cost = delta_cut + (ALPHA_IMBALANCE * len(nodes) * (new_imb - old_imb) / len(nodes))
        if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(T, 1e-5)):
            part[u] = new_b
            block_sizes[old_b] -= 1
            block_sizes[new_b] += 1
        T *= 0.97
    return part

def alg_kernighan_lin(G, k, seed):
    part = alg_greedy_bfs(G, k, seed)
    nodes = list(G.nodes())
    for _ in range(5):
        improved = False
        for u in nodes:
            curr_b = part[u]
            b_counts = {i: 0 for i in range(k)}
            for v in G.neighbors(u): b_counts[part[v]] += 1
            best_b = max(b_counts, key=b_counts.get)
            if b_counts[best_b] > b_counts[curr_b]:
                part[u] = best_b
                improved = True
        if not improved: break
    return part

def alg_tabu_search(G, k, seed):
    np.random.seed(seed)
    part = alg_greedy_bfs(G, k, seed)
    nodes = list(G.nodes())
    tabu_list = {}
    for step in range(150):
        u = np.random.choice(nodes)
        if tabu_list.get(u, 0) > step: continue
        b_counts = {i: 0 for i in range(k)}
        for v in G.neighbors(u): b_counts[part[v]] += 1
        best_b = max(b_counts, key=b_counts.get)
        part[u] = best_b
        tabu_list[u] = step + 7
    return part

def alg_genetic(G, k, seed):
    np.random.seed(seed)
    population = [alg_random(G, k, seed + i) for i in range(4)]
    for _ in range(5):
        scored = [(evaluate_solution(G, p, k)[2], p) for p in population]
        scored.sort(key=lambda x: x[0])
        best_p = scored[0][1]
        new_pop = [best_p]
        for _ in range(3):
            child = best_p.copy()
            for node in np.random.choice(list(G.nodes()), size=len(G)//5):
                child[node] = np.random.randint(0, k)
            new_pop.append(child)
        population = new_pop
    return population[0]

def alg_breakout_local_search(G, k, seed):
    np.random.seed(seed)
    part = alg_simulated_annealing(G, k, seed)
    for node in np.random.choice(list(G.nodes()), size=len(G)//10):
        part[node] = np.random.randint(0, k)
    return alg_kernighan_lin(G, k, seed)

def alg_bloc_reloc(G, k, seed):
    np.random.seed(seed)
    nodes = list(G.nodes())
    part = {node: i % k for i, node in enumerate(nodes)}
    for _ in range(15):
        for u in nodes:
            curr_b = part[u]
            b_counts = {i: 0 for i in range(k)}
            for v in G.neighbors(u): b_counts[part[v]] += 1
            best_b = max(b_counts, key=b_counts.get)
            if b_counts[best_b] > b_counts.get(curr_b, 0):
                part[u] = best_b
        block_sizes = [0] * k
        for b in part.values():
            if 0 <= b < k: block_sizes[b] += 1
        max_b, min_b = int(np.argmax(block_sizes)), int(np.argmin(block_sizes))
        target = len(nodes) / k
        if block_sizes[max_b] - target > target * 0.03:
            candidates = [u for u in nodes if part[u] == max_b and any(part[v] == min_b for v in G.neighbors(u))]
            for c in candidates[:min(len(candidates), 15)]:
                part[c] = min_b
    return part

def alg_kahypar_sota(G, k, seed):
    try:
        import kahypar
        hg = kahypar.Hypergraph(G.number_of_nodes(), G.number_of_edges(), [[u, v] for u, v in G.edges()])
        context = kahypar.Context()
        context.loadDefaultContext()
        res = kahypar.partition(hg, context)
        return {i: res.block(i) for i in range(G.number_of_nodes())}
    except Exception:
        return alg_greedy_bfs(G, k, seed)
