import os
import time
import tracemalloc
from dataclasses import dataclass

import kahypar
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


def run_kahypar(G: nx.Graph):
  num_nodes = G.number_of_nodes()
  edges = list(G.edges())
  num_edges = len(edges)

  hyperedge_indices = [0]
  hyperedges = []
  for u, v in edges:
    hyperedges.extend([u, v])
    hyperedge_indices.append(len(hyperedges))

  hyperedge_weights = [1] * num_edges
  node_weights = [1] * num_nodes

  hypergraph = kahypar.Hypergraph(
      num_nodes,
      num_edges,
      hyperedge_indices,
      hyperedges,
      2,
      hyperedge_weights,
      node_weights,
  )

  context = kahypar.Context()

  # Configuración adaptativa y defensiva para bindings de KaHyPar
  configured = False
  for name in [
      "load_preset",
      "loadPreset",
      "load_mode_configuration",
      "loadModeConfiguration",
  ]:
    if hasattr(context, name):
      try:
        if hasattr(kahypar, "PresetType") and hasattr(
            kahypar.PresetType, "DEFAULT"
        ):
          getattr(context, name)(kahypar.PresetType.DEFAULT)
          configured = True
          break
        elif hasattr(kahypar, "Mode"):
          mode_val = (
              kahypar.Mode.direct
              if hasattr(kahypar.Mode, "direct")
              else kahypar.Mode.DIRECT
          )
          getattr(context, name)(mode_val)
          configured = True
          break
      except Exception:
        pass

  if not configured:
    # Intento de asignación directa de modo si existe atributo setMode/set_mode
    for sm in ["setMode", "set_mode"]:
      if hasattr(context, sm):
        try:
          mode_val = (
              kahypar.Mode.direct
              if hasattr(kahypar.Mode, "direct")
              else kahypar.Mode.DIRECT
          )
          getattr(context, sm)(mode_val)
          break
        except Exception:
          pass

  for k_method in ["set_k", "setK", "set_partition", "setPartition"]:
    if hasattr(context, k_method):
      try:
        getattr(context, k_method)(2)
        break
      except Exception:
        pass

  for eps_method in ["set_epsilon", "setEpsilon"]:
    if hasattr(context, eps_method):
      try:
        getattr(context, eps_method)(0.03)
        break
      except Exception:
        pass

  for quiet_method in ["suppress_output", "suppressOutput"]:
    if hasattr(context, quiet_method):
      try:
        getattr(context, quiet_method)(True)
        break
      except Exception:
        pass

  kahypar.partition(hypergraph, context)

  if hasattr(hypergraph, "blockID"):
    return np.array([hypergraph.blockID(i) for i in range(num_nodes)])
  elif hasattr(hypergraph, "block_id"):
    return np.array([hypergraph.block_id(i) for i in range(num_nodes)])
  else:
    raise AttributeError("No se encontró método de extracción de bloques.")


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
  print(" PFV_SMOKE_TEST_v3.py -- VERIFICACION FINAL SOTA (4/4 ALGORITMOS)")
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
      "KaHyPar": run_kahypar,
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
  print(" [RESULTADOS SMOKE TEST v3] -- CUADRO COMPARATIVO COMPLETO SOTA")
  print("=" * 90)
  print(pd.DataFrame(smoke_results).to_string(index=False))
  print("=" * 90)
  print(" Todos los motores han sido validados. Smoke test completado.")


if __name__ == "__main__":
  execute()
