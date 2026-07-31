/

* ==============================================================================
* BLOC-RELOC v2: Motor de Particionamiento con Hub-Shielded Affinity y D-Shock
* Entorno: C++20 / OpenMP / High-Performance Workspace
* ==============================================================================
*/

#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include 
#include <unordered_set>
#include 

#ifdef _OPENMP
#include <omp.h>
#endif

struct Graph {
int n;
long long m;
std::vector row_ptr;
std::vector col_idx;
std::vector weights;
std::vector degrees;
double mean_degree;
double std_degree;
};

Graph generate_test_graph(int n, int avg_deg, uint64_t seed) {
Graph g;
g.n = n;
g.row_ptr.push_back(0);
std::mt19937_64 rng(seed);
std::vector<std::unordered_set> adj(n);

```
for (int i = 0; i < n; ++i) {
    int edges = std::max(1, (int)(rng() % (avg_deg * 2)));
    for (int e = 0; e < edges; ++e) {
        int target = rng() % n;
        if (target != i) {
            adj[i].insert(target);
            adj[target].insert(i);
        }
    }
}

g.degrees.resize(n, 0.0);
long long total_edges = 0;
for (int i = 0; i < n; ++i) {
    g.degrees[i] = static_cast<double>(adj[i].size());
    total_edges += adj[i].size();
    for (int neighbor : adj[i]) {
        g.col_idx.push_back(neighbor);
        g.weights.push_back(1.0);
    }
    g.row_ptr.push_back(g.col_idx.size());
}
g.m = total_edges / 2;

double sum = std::accumulate(g.degrees.begin(), g.degrees.end(), 0.0);
g.mean_degree = sum / n;
double sq_sum = 0.0;
for (double d : g.degrees) sq_sum += (d - g.mean_degree) * (d - g.mean_degree);
g.std_degree = std::sqrt(sq_sum / n);

return g;

```

}

class BlocRelocV2 {
private:
Graph g;
int k_blocks;
double alpha;
double k_sigma;
std::vector hubs;
std::vector shielded_weights;

```
void compute_hub_shielding() {
    double threshold = g.mean_degree + k_sigma * g.std_degree;
    hubs.clear();
    shielded_weights = g.weights;

    for (int u = 0; u < g.n; ++u) {
        if (g.degrees[u] > threshold) {
            hubs.push_back(u);
        }
    }

    #pragma omp parallel for
    for (size_t idx = 0; idx < g.col_idx.size(); ++idx) {
        int u = 0;
        while (u < g.n && g.row_ptr[u + 1] <= static_cast<int>(idx)) u++;
        int v = g.col_idx[idx];

        bool u_is_hub = (g.degrees[u] > threshold);
        bool v_is_hub = (g.degrees[v] > threshold);

        if (u_is_hub || v_is_hub) {
            double factor_u = u_is_hub ? (g.mean_degree / g.degrees[u]) : 1.0;
            double factor_v = v_is_hub ? (g.mean_degree / g.degrees[v]) : 1.0;
            shielded_weights[idx] *= std::pow(factor_u * factor_v, alpha);
        }
    }
}

```

public:
BlocRelocV2(const Graph& graph, int k, float a = 0.75f, float ks = 1.5f)
: g(graph), k_blocks(k), alpha(a), k_sigma(ks) {
compute_hub_shielding();
}

```
struct Result {
    double cut_cost;
    double execution_time_ms;
    int stalls_escaped;
    bool success;
};

Result optimize(uint64_t seed, int max_iter = 150, int max_stall = 15) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::mt19937_64 rng(seed);

    std::vector<int> partition(g.n);
    for (int i = 0; i < g.n; ++i) {
        partition[i] = rng() % k_blocks;
    }

    double current_cut = evaluate_cut(partition);
    double best_cut = current_cut;
    int stall_count = 0;
    int d_shocks_applied = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool improved = false;

        for (int u = 0; u < g.n; ++u) {
            int current_block = partition[u];
            int best_target_block = current_block;
            double min_delta = 0.0;

            for (int b = 0; b < k_blocks; ++b) {
                if (b == current_block) continue;
                double delta = evaluate_marginal_gain(u, current_block, b, partition);
                if (delta < min_delta) {
                    min_delta = delta;
                    best_target_block = b;
                }
            }

            if (best_target_block != current_block) {
                partition[u] = best_target_block;
                current_cut += min_delta;
                improved = true;
            }
        }

        if (current_cut < best_cut) {
            best_cut = current_cut;
            stall_count = 0;
        } else {
            stall_count++;
        }

        if (stall_count >= max_stall) {
            d_shocks_applied++;
            apply_d_shock(partition, rng);
            current_cut = evaluate_cut(partition);
            stall_count = 0;
        }

        if (!improved && stall_count == 0) break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return {best_cut, duration, d_shocks_applied, true};
}

```

private:
double evaluate_cut(const std::vector& partition) const {
double cut = 0.0;
#pragma omp parallel for reduction(+:cut)
for (int u = 0; u < g.n; ++u) {
for (int idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
int v = g.col_idx[idx];
if (u < v && partition[u] != partition[v]) {
cut += g.weights[idx];
}
}
}
return cut;
}

```
double evaluate_marginal_gain(int u, int from_b, int to_b, const std::vector<int>& partition) const {
    double gain = 0.0;
    for (int idx = g.row_ptr[u]; idx < g.row_ptr[u + 1]; ++idx) {
        int v = g.col_idx[idx];
        int v_block = partition[v];
        double w = shielded_weights[idx];

        if (v_block == from_b) gain += w;
        if (v_block == to_b)   gain -= w;
    }
    return gain;
}

void apply_d_shock(std::vector<int>& partition, std::mt19937_64& rng) {
    int perturbations = std::max(1, g.n / 20);
    for (int p = 0; p < perturbations; ++p) {
        int node = rng() % g.n;
        partition[node] = rng() % k_blocks;
    }
}

```

};

int main() {
std::cout << "===\n";
std::cout << "   BLOC-RELOC v2: EJECUCIÓN DE BENCHMARK AUTOMATIZADO  \n";
std::cout << "===\n\n";

```
#ifdef _OPENMP
std::cout << "[INFO] OpenMP activado. Hilos máximos: " << omp_get_max_threads() << "\n";
#endif

struct TestInstance {
    std::string name;
    int n;
    int avg_deg;
    int k;
};

std::vector<TestInstance> suite = {
    {"BA_Synthetic_Small", 2000, 10, 4},
    {"BA_Synthetic_Medium", 10000, 15, 8},
    {"ScaleFree_Large_Sim", 50000, 20, 16}
};

std::ofstream report("bloc_reloc_v2_metrics_report.csv");
report << "Instance,Nodes,Blocks,Seed,CutCost,TimeMs,DShocks\n";

const int runs = 5;
for (const auto& inst : suite) {
    std::cout << "Ejecutando suite en instancia: " << inst.name << " (N=" << inst.n << ")\n";
    Graph g = generate_test_graph(inst.n, inst.avg_deg, 4242);

    for (int r = 0; r < runs; ++r) {
        uint64_t seed = 1337 + r * 101;
        BlocRelocV2 solver(g, inst.k);
        auto res = solver.optimize(seed);

        std::cout << "  -> Corrida " << r + 1 << "/" << runs 
                  << " | Corte: " << res.cut_cost 
                  << " | Tiempo: " << res.execution_time_ms << " ms"
                  << " | D-Shocks: " << res.stalls_escaped << "\n";

        report << inst.name << "," << inst.n << "," << inst.k << "," 
               << seed << "," << res.cut_cost << "," << res.execution_time_ms << "," 
               << res.stalls_escaped << "\n";
    }
}

report.close();
std::cout << "\n[EXITO] Benchmark completado. Métricas guardadas en 'bloc_reloc_v2_metrics_report.csv'.\n";
return 0;

```

}
