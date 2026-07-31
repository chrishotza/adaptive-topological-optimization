import os
import sys
import time
import tracemalloc
import multiprocessing as mp
import pandas as pd
import master_validation_core as core

def worker_runner(func, G, k, seed, queue):
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        part = func(G, k, seed)
        t_elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        queue.put(('SUCCESS', part, t_elapsed, peak_mem / (1024*1024)))
    except Exception as e:
        queue.put(('ERROR', str(e), 0, 0))

def execute_with_timeout(func, G, k, seed, timeout):
    queue = mp.Queue()
    p = mp.Process(target=worker_runner, args=(func, G, k, seed, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return 'TIMEOUT', None, timeout, 0.0
    if not queue.empty():
        return queue.get()
    return 'ERROR', None, 0.0, 0.0

def execute_all_experiments(instances, solvers_avail):
    algorithms = {
        'Random': core.alg_random,
        'Greedy': core.alg_greedy_bfs,
        'Spectral': core.alg_spectral_approx,
        'Simulated_Annealing': core.alg_simulated_annealing,
        'Kernighan_Lin': core.alg_kernighan_lin,
        'Tabu_Search': core.alg_tabu_search,
        'Genetic': core.alg_genetic,
        'Breakout_Local_Search': core.alg_breakout_local_search,
        'BLOC_RELOC (Proposed)': core.alg_bloc_reloc
    }
    if solvers_avail.get('KaHyPar_Py'):
        algorithms['KaHyPar_SOTA'] = core.alg_kahypar_sota

    os.makedirs(core.REPORT_DIR, exist_ok=True)
    csv_path = f"{core.REPORT_DIR}/detailed_runs.csv"
    
    header_df = pd.DataFrame(columns=[
        'Instance', 'Algorithm', 'Seed', 'Cut', 'Conductance', 
        'Balance', 'Cost', 'Gap', 'Time_s', 'RAM_MB', 'Timeout', 'Error'
    ])
    header_df.to_csv(csv_path, index=False)

    for inst in instances:
        G = inst['graph']
        inst_name = inst['name']
        print(f"\n[>>>] Procesando Instancia: {inst_name}", flush=True)
        
        baseline_costs = []
        for alg_name, alg_fn in algorithms.items():
            try:
                status, part, _, _ = execute_with_timeout(alg_fn, G, core.K_PARTITIONS, core.SEEDS[0], core.TIMEOUT_SEC)
                if status == 'SUCCESS':
                    _, _, c, _ = core.evaluate_solution(G, part, core.K_PARTITIONS)
                    baseline_costs.append(c)
            except Exception:
                pass
        min_global_cost = min(baseline_costs) if baseline_costs else 1.0

        for alg_name, alg_fn in algorithms.items():
            for seed in core.SEEDS:
                print(f"  [->] Iniciando: {inst_name} | Algoritmo: {alg_name} | Semilla: {seed}", flush=True)
                t_start = time.perf_counter()
                try:
                    status, part, t_elapsed, mem_mb = execute_with_timeout(alg_fn, G, core.K_PARTITIONS, seed, core.TIMEOUT_SEC)
                    
                    if status == 'SUCCESS':
                        cut, imb, cost, cond = core.evaluate_solution(G, part, core.K_PARTITIONS)
                        gap = (cost - min_global_cost) / (min_global_cost + 1e-8)
                        record = {
                            'Instance': inst_name, 'Algorithm': alg_name, 'Seed': seed,
                            'Cut': cut, 'Conductance': cond, 'Balance': imb, 'Cost': cost, 'Gap': gap,
                            'Time_s': t_elapsed, 'RAM_MB': mem_mb, 'Timeout': 0, 'Error': 0
                        }
                        print(f"  [✔] OK: {alg_name} (Seed {seed}) en {t_elapsed:.2f}s | Cost: {cost}", flush=True)
                    elif status == 'TIMEOUT':
                        record = {
                            'Instance': inst_name, 'Algorithm': alg_name, 'Seed': seed,
                            'Cut': 999999, 'Conductance': 1.0, 'Balance': 1.0, 'Cost': 999999.0, 'Gap': 1.0,
                            'Time_s': core.TIMEOUT_SEC, 'RAM_MB': 0.0, 'Timeout': 1, 'Error': 0
                        }
                        print(f"  [⏱] TIMEOUT: {alg_name} (Seed {seed})", flush=True)
                    else:
                        record = {
                            'Instance': inst_name, 'Algorithm': alg_name, 'Seed': seed,
                            'Cut': 999999, 'Conductance': 1.0, 'Balance': 1.0, 'Cost': 999999.0, 'Gap': 1.0,
                            'Time_s': core.TIMEOUT_SEC, 'RAM_MB': 0.0, 'Timeout': 0, 'Error': 1
                        }
                        print(f"  [✖] ERROR: {alg_name} (Seed {seed})", flush=True)
                except Exception as e:
                    t_elapsed = time.perf_counter() - t_start
                    record = {
                        'Instance': inst_name, 'Algorithm': alg_name, 'Seed': seed,
                        'Cut': 999999, 'Conductance': 1.0, 'Balance': 1.0, 'Cost': 999999.0, 'Gap': 1.0,
                        'Time_s': t_elapsed, 'RAM_MB': 0.0, 'Timeout': 0, 'Error': 1
                    }
                    print(f"  [!] EXCEPTION en {alg_name} (Seed {seed}): {e}", flush=True)

                df_row = pd.DataFrame([record])
                df_row.to_csv(csv_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    print("[*] Auditoría de solvers y datasets a través de core...", flush=True)
    solvers_avail = core.audit_solvers()
    instances = core.fetch_and_prepare_benchmarks()
    if not instances:
        print("[!] No hay instancias disponibles.", flush=True)
        sys.exit(1)
    print("[*] Iniciando ejecución instrumentada con guardado incremental...", flush=True)
    execute_all_experiments(instances, solvers_avail)
    print(f"\n[✓] EJECUCIÓN COMPLETADA. Resultados incrementales en: {core.REPORT_DIR}/detailed_runs.csv", flush=True)
