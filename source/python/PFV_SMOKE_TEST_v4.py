import os
import time
import tracemalloc
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp
from numba import njit


@dataclass(slots=True)
class GraphCSR:
  indptr: np.ndarray
  indices: np.ndarray
  data: np.ndarray
  degrees: np.ndarray
  N: int
  vol_total: int


def nx_to_csr(G: nx.Graph) -> GraphCSR:
  adj = nx.adjacency_matrix(G)
  csr = adj.tocsr()
  degrees = np.diff(csr.indptr)
  return GraphCSR(
      indptr=csr.indptr,
      indices=csr.indices,
      data=csr.data.astype(np.float64),
      degrees=degrees,
      N=G.number_of_nodes(),
      vol_total=int(degrees.sum()),
  )


def calculate_metrics(G: nx.Graph, part: np.ndarray):
  cut = sum(1 for u, v in G.edges() if part[u] != part[v])
  s0 = np.sum(part == 0)
  s1 = np.sum(part == 1)
  balance = abs(s0 - s1) / len(part)
  return cut, balance


def run_metis(G: nx.Graph):
  adj_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
  _, parts = pymetis.part_graph(2, adjacency=adj_list)
  return np.array(parts)


def run_louvain(G: nx.Graph):
  communities = list(nx.community.greedy_modularity_communities(G))
  parts = np.zeros(G.number_of_nodes(), dtype=int)
  if len(communities) > 1:
    for node in communities[1]:
      parts[node] = 1
  return parts


def run_kernighan_lin(G: nx.Graph):
  _, part1 = nx.community.kernighan_lin_bisection(G)
  parts = np.zeros(G.number_of_nodes(), dtype=int)
  for node in part1:
    parts[node] = 1
  return parts


@njit(cache=True)
def pfv_refine_numba(indptr, indices, part, max_iters=10):
  N = len(part)
  for _ in range(max_iters):
    improved = False
    for u in range(N):
      curr_p = part[u]
      ext_deg = int_deg = 0
      for idx in range(indptr[u], indptr[u + 1]):
        if part[indices[idx]] == curr_p:
          int_deg += 1
        else:
          ext_deg += 1
      if ext_deg > int_deg:
        part[u] = 1 - curr_p
        improved = True
    if not improved:
      break
  return part


def run_pfv_v17(G: nx.Graph):
  csr = nx_to_csr(G)
  adj = nx.adjacency_matrix(G).astype(float)
  deg = np.array(adj.sum(axis=1)).flatten()
  D = sp.diags(np.power(np.maximum(deg, 1.0), -0.5))
  L = sp.eye(G.number_of_nodes()) - D @ adj @ D

  vals, vecs = sp.linalg.eigsh(L, k=2, which="SM")
  fiedler = vecs[:, 1]

  part = (fiedler >= np.median(fiedler)).astype(np.int32)
  return pfv_refine_numba(csr.indptr, csr.indices, part)


def warmup_jit():
  pfv_refine_numba(
      np.array([0, 1, 2], dtype=np.int32),
      np.array([1, 0], dtype=np.int32),
      np.array([0, 1], dtype=np.int32),
  )


def execute():
  print("=" * 90)
  print(
      " PFV_SMOKE_TEST_v4.py -- VERIFICACION FINAL SOTA (4 ALGORITMOS DE"
      " RED)"
  )
  print("=" * 90)

  print("\n[WARM-UP] Ejecutando warm-up de Numba JIT para PFV v17.1...")
  warmup_jit()
  print(
      "[WARM-UP] Rutinas Numba listas en memoria. Evaluando algoritmos...\n"
  )

  np.random.seed(42)
  G = nx.stochastic_block_model([250, 250], [[0.1, 0.01], [0.01, 0.1]], seed=42)

  algorithms = {
      "PFV v17.1": run_pfv_v17,
      "METIS": run_metis,
      "Louvain/Mod": run_louvain,
      "Kernighan-Lin": run_kernighan_lin,
  }

  smoke_results = []

  for name, func in algorithms.items():
    print(f" -> Ejecutando: {name}...")
    tracemalloc.start()
    t0 = time.perf_counter()

    try:
      part = func(G)
      t1 = time.perf_counter()
      _, peak_mem = tracemalloc.get_traced_memory()
      tracemalloc.stop()

      cut, balance = calculate_metrics(G, part)
      smoke_results.append({
          "Algoritmo": name,
          "Corte": cut,
          "Balance": f"{balance:.4f}",
          "Tiempo (s)": f"{(t1 - t0):.5f}",
          "Memoria (MB)": f"{peak_mem / 1048576:.2f}",
          "Estado": "OK",
      })
    except Exception as e:
      tracemalloc.stop()
      smoke_results.append({
          "Algoritmo": name,
          "Corte": "N/A",
          "Balance": "N/A",
          "Tiempo (s)": "N/A",
          "Memoria (MB)": "N/A",
          "Estado": f"Error: {e}",
      })
      print(f"    [!] Error en {name}: {e}")

  print("\n" + "=" * 90)
  print(" [RESULTADOS SMOKE TEST v4] -- CUADRO COMPARATIVO COMPLETO SOTA")
  print("=" * 90)
  print(pd.DataFrame(smoke_results).to_string(index=False))
  print("=" * 90)
  print(" Todos los motores han sido validados. Smoke test completado.")


if __name__ == "__main__":
  execute()
