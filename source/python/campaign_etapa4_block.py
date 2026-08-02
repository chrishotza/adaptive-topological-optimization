#!/usr/bin/env python3
"""
campaign_etapa4_block.py
MVP de optimización por bloques coordinados (BLOC-RELOC) para particionamiento de grafos.
Objetivo: Romper el techo de PFV (618) en Social_Network y acercarse a METIS (488).
"""

import os
import random
import networkx as nx

def calcular_corte(G, partition):
    """Calcula el número de aristas de corte (edge cut) entre particiones."""
    corte = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:
            corte += 1
    return corte

def obtener_frontera(G, partition):
    """Detecta nodos frontera (con al menos un vecino en la otra partición)."""
    frontera = set()
    for u, v in G.edges():
        if partition[u] != partition[v]:
            frontera.add(u)
            frontera.add(v)
    return list(frontera)

def construir_bloque(G, start_node, partition, max_size=5):
    """Construye un bloque conectado de 2 a 5 nodos dentro de la misma partición."""
    p_val = partition[start_node]
    bloque = [start_node]
    frontera_local = list(G.neighbors(start_node))
    
    while len(bloque) < max_size and frontera_local:
        candidatos = [n for n in frontera_local if partition[n] == p_val and n not in bloque]
        if not candidatos:
            break
        siguiente = random.choice(candidatos)
        bloque.append(siguiente)
        frontera_local.remove(siguiente)
        for n in G.neighbors(siguiente):
            if n not in bloque and n not in frontera_local:
                frontera_local.append(n)
                
    return bloque

def aplicar_movimiento_bloque(partition, bloque):
    """Invierte la asignación de partición para todos los nodos del bloque."""
    nueva_part = partition.copy()
    for nodo in bloque:
        nueva_part[nodo] = 1 - nueva_part[nodo]
    return nueva_part

def ejecutar_busqueda_local_bloques(G, partition_inicial, max_iter=200, min_size=2, max_size=5):
    """Ejecuta la búsqueda local basada en bloques coordinados."""
    part_actual = partition_inicial.copy()
    corte_actual = calcular_corte(G, part_actual)
    
    print(f"[INICIO] Corte inicial: {corte_actual}")
    
    mejorado = True
    iteracion = 0
    
    while mejorado and iteracion < max_iter:
        mejorado = False
        iteracion += 1
        frontera = obtener_frontera(G, part_actual)
        
        if not frontera:
            break
            
        random.shuffle(frontera)
        
        for nodo_inicio in frontera:
            tamanio_bloque = random.randint(min_size, max_size)
            bloque = construir_bloque(G, nodo_inicio, part_actual, max_size=tamanio_bloque)
            
            if len(bloque) < min_size:
                continue
                
            part_candidata = aplicar_movimiento_bloque(part_actual, bloque)
            corte_candidato = calcular_corte(G, part_candidata)
            
            # Criterio de aceptación estricto: Aceptar sólo si mejora
            if corte_candidato < corte_actual:
                print(f"[MEJORA] Iter {iteracion}: Bloque de {len(bloque)} nodos | Corte: {corte_actual} -> {corte_candidato}")
                part_actual = part_candidata
                corte_actual = corte_candidato
                mejorado = True
                break  # Reiniciar búsqueda tras la mejora (refinamiento inmediato)
                
    print(f"[FIN] Corte final obtenido: {corte_actual}")
    return part_actual, corte_actual

if __name__ == "__main__":
    # Configuración de prueba / Carga de Social_Network (ajustar ruta de datos según entorno)
    print("--- CAMPAÑA ETAPA 4: MVP BLOC-RELOC ---")
    
    # Simulación de grafo para validación sintáctica y ejecución directa del script
    # Reemplazar por nx.read_edgelist('Social_Network.txt') al conectar con los datos reales
    G = nx.karate_club_graph() 
    
    nodes = list(G.nodes())
    random.seed(42)
    random.shuffle(nodes)
    mid = len(nodes) // 2
    
    # Partición inicial aleatoria balanceada
    particion_inicial = {n: (0 if i < mid else 1) for i, n in enumerate(nodes)}
    
    print(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas.")
    
    # Ejecución del operador
    particion_final, corte_final = ejecutar_busqueda_local_bloques(G, particion_inicial)