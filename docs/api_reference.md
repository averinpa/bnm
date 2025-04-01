# API Reference: BNM (Bayesian Network Metrics)

## `BNMetrics.compare_df`

```python
BNMetrics.compare_df(descriptive_metrics='All', comparison_metrics='All')
```

Compile and merge descriptive and comparative metrics into a single table.

### 🔢 Descriptive Metrics

| Metric                  | Calculation Function             | Description                                                                 |
|------------------------|----------------------------------|-----------------------------------------------------------------------------|
| `n_edges`              | `count_edges`                    | Total number of edges                                                       |
| `n_nodes`              | `count_nodes`                    | Total number of nodes                                                       |
| `n_colliders`          | `count_colliders`                | Number of collider structures (X → Z ← Y)                                   |
| `n_root_nodes`         | `count_root_nodes_strict`        | Nodes with no directed or undirected parents                                |
| `n_leaf_nodes`         | `count_leaf_nodes_strict`        | Nodes with no directed or undirected children                               |
| `n_isolated_nodes`     | `count_isolated_nodes_strict`    | Nodes with no incident directed or undirected edges                         |
| `n_directed_arcs`      | `count_directed_arcs`            | Number of edges marked as directed                                          |
| `n_undirected_arcs`    | `count_undirected_arcs`          | Number of edges marked as undirected                                        |
| `n_reversible_arcs`    | `count_reversible_arcs`          | Directed edges not part of any collider                                     |
| `n_in_degree`          | `count_in_degree_directed`       | Number of directed parents of a node                                        |
| `n_out_degree`         | `count_out_degree_directed`      | Number of directed children of a node                                       |

### ⚖️ Comparative Metrics

| Metric         | Calculation Function           | Description                                                                 |
|----------------|--------------------------------|-----------------------------------------------------------------------------|
| `additions`    | `count_additions`              | Edges present in G2 but not in G1 (ignoring direction)                      |
| `deletions`    | `count_deletions`              | Edges present in G1 but not in G2                                           |
| `reversals`    | `count_reversals`              | Directed edges that were reversed or became undirected                      |
| `shd`          | `shd`                           | Structural Hamming Distance (additions + deletions + reversals)            |
| `hd`           | `hd`                            | Hamming Distance (additions + deletions only)                              |
| `tp`           | `count_true_positives`         | Edges correctly predicted in G2 with same direction/type                    |
| `fp`           | `count_false_positives`        | Extra edges in G2 not in G1                                                 |
| `fn`           | `count_false_negatives`        | Missing edges in G2 that were in G1                                         |
| `precision`    | `precision`                    | TP / (TP + FP)                                                              |
| `recall`       | `recall`                       | TP / (TP + FN)                                                              |
| `f1_score`     | `f1_score`                     | Harmonic mean of precision and recall                                       |

### Parameters

- `descriptive_metrics` : `list[str]` or `'All'`, default=`'All'`
- `comparison_metrics` : `list[str]` or `'All'`, default=`'All'`

### Returns

- `pandas.DataFrame` — Combined table with metrics.

### Example

```python
bn = BNMetrics(G1, G2)
df = bn.compare_df()
```

---

## `BNMetrics.compare_two_bn`

```python
BNMetrics.compare_two_bn(nodes, option=1, name1='DAG1', name2='DAG2')
```

Visualize two networks side-by-side, highlighting shared edges and selected nodes.

### Parameters
- `nodes`: List of node names to highlight.
- `option`: 1, 2, or 3 (controls subgraph selection strategy)
- `name1`, `name2`: Graph titles.

### Example
```python
bn.compare_two_bn(nodes=['X_1', 'X_2'])
```

---

## `BNMetrics.plot_bn`

```python
BNMetrics.plot_bn(nodes, layer="d1", title="DAG")
```

Display a single DAG from one of the layers (`d1`, `d2`, or `d3`).

### Parameters
- `nodes`: Nodes to highlight
- `layer`: One of `'d1'`, `'d2'`, or `'d3'`
- `title`: Title shown on the plot

### Example
```python
bn.plot_bn(nodes=['X_1', 'X_5'], layer='d1', title='Markov Blanket')
```

---

## `generate_random_dag`

```python
from bnm.utils import generate_random_dag
G = generate_random_dag(n_nodes=50, edge_prob=0.1, seed=42)
```

Generate a random DAG based on topological ordering.

### Parameters
- `n_nodes`: Number of nodes
- `edge_prob`: Probability of edge creation (upper-triangle only)
- `seed`: Random seed

### Returns
- `networkx.DiGraph`: Random DAG with nodes named `X_1`, `X_2`, ...

---

## `dag_to_cpdag`

```python
from bnm.utils import dag_to_cpdag
cpdag = dag_to_cpdag(G)
```

Convert a DAG to its CPDAG skeleton (preserving colliders and undirected edges).

### Parameters
- `G`: `networkx.DiGraph`

### Returns
- `networkx.DiGraph` CPDAG-like structure
