import warnings
import networkx as nx
import numpy as np
import pandas as pd
import pymetis
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=UserWarning)

def calculate_conductance(G: nx.Graph, part: np.ndarray):
    """Calcula la conductancia del corte (aristas entre particiones / volumen menor)."""
    cut_edges = sum(1 for u, v in G.edges() if part[u] != part[v])
    vol0 = sum(G.degree(u) for u in range(G.number_of_nodes()) if part[u] == 0)
    vol1 = sum(G.degree(u) for u in range(G.number_of_nodes()) if part[u] == 1)
    min_vol = min(vol0, vol1)
    return cut_edges / min_vol if min_vol > 0 else 0.0

def get_graph_topological_profile(G: nx.Graph, name: str):
    degrees = [d for _, d in G.degree()]
    deg_array = np.array(degrees, dtype=float)
    
    profile = {
        "Instancia": name,
        "N": G.number_of_nodes(),
        "M": G.number_of_edges(),
        "Densidad": nx.density(G),
        "Clustering Prom.": nx.average_clustering(G),
        "Asortatividad": nx.degree_assortativity_coefficient(G),
        "Varianza Grados": np.var(deg_array),
        "Max Grado": int(np.max(deg_array)),
    }
    return profile

def run_profiling_campaign():
    print("=" * 130)
    print(" CARPAÑA DE CARACTERIZACIÓN TOPOLÓGICA Y DOMINIO DE APLICABILIDAD")
    print("=" * 130)
    
    # Generación de los escenarios representativos evaluados
    datasets = {
        "Barabasi_Albert": nx.barabasi_albert_graph(2000, 5, seed=42),
        "Erdos_Renyi": nx.erdos_renyi_graph(2000, 0.02, seed=42),
        "LesMis": nx.les_miserables_graph() if hasattr(nx, "les_miserables_graph") else nx.erdos_renyi_graph(77, 0.15, seed=42),
        "PowerGrid_Sim": nx.powerlaw_cluster_graph(2000, 3, 0.05, seed=42),
        "Social_Network": nx.newman_watts_strogatz_graph(1500, 10, 0.1, seed=42)
    }
    
    profiles = []
    
    for name, G_raw in datasets.items():
        G = nx.convert_node_labels_to_integers(G_raw)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            G = nx.convert_node_labels_to_integers(G)
            
        prof = get_graph_topological_profile(G, name)
        
        # Ejecución rápida de METIS vs PFV (tol=5%) para registrar el gap y conductancia
        adj_list = [list(G.neighbors(i)) for i in range(G.number_of_nodes())]
        _, parts_metis = pymetis.part_graph(2, adjacency=adj_list)
        cut_metis = sum(1 for u, v in G.edges() if parts_metis[u] != parts_metis[v])
        
        prof["Corte METIS"] = cut_metis
        prof["Conductancia METIS"] = calculate_conductance(G, np.array(parts_metis))
        
        profiles.append(prof)
        
    df_profiles = pd.DataFrame(profiles)
    print("\n" + "=" * 130)
    print(" [MATRIZ DE PERFILADO TOPOLÓGICO Y ESTRUCTURAL]")
    print("=" * 130)
    print(df_profiles.to_string(index=False))
    print("=" * 130)

if __name__ == "__main__":
    run_profiling_campaign()