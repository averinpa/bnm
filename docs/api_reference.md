# API Reference: BNM (Bayesian Network Metrics)<a href="https://github.com/averinpa/bnm/blob/cd7e82a77dfd69c1890687318ae32e37e2188192/bnm/core.py#L14" style="float: right; font-weight: normal;">[source]</a>

```python
class BNMetrics(G1, G2=None, node_names=None)
```

The `BNMetrics` class computes and compares descriptive and comparative metrics between one or two Bayesian networks (DAGs), with support for visualization.

Initialize a BNMetrics object with one or two DAGs. This class supports flexible input formats for causal structure comparison. The graphs can be provided either as `networkx.DiGraph` objects or as adjacency matrices (NumPy arrays or list-of-lists). If matrices are passed, `node_names` must also be provided to assign names to the nodes.

All edges are processed to detect and mark bidirected edges as "undirected". Bidirected edges are collapsed into one edge. Directed edges are marked with "directed". Subgraphs for each node’s Markov blanket are computed and stored for downstream metric calculations and visualizations.

## Parameters

**G1** : `nx.DiGraph`, `np.ndarray`, or `list of lists`  
: The first graph (base DAG). If not a DiGraph, it must be a square adjacency matrix.

**G2** : `nx.DiGraph`, `np.ndarray`, or `list of lists`, default=`None`  
: The second graph (comparison DAG). Must have the same node names and structure 
as `G1`. If not provided, BNMetrics operates in single-graph mode.

**node_names** : `list of str`, optional  
: Required only when `G1` or `G2` is given as a NumPy array or list of lists.
Length must match number of nodes.

## Raises

**ValueError**  
- If `G1` or `G2` is not square when passed as a matrix.  
- If `node_names` are missing or mismatched.  
- If `G1` and `G2` have different node sets.

## Examples

```python
# Using networkx graphs
import networkx as nx
from bnm import BNMetrics

G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("C", "B")])

G2 = nx.DiGraph()
G2.add_edges_from([("A", "B"), ("B", "C")])

bnm = BNMetrics(G1, G2)
```

```python
# Using NumPy adjacency matrices
import numpy as np

mat1 = np.array([[0, 1], [0, 0]])
mat2 = np.array([[0, 0], [1, 0]])

bnm = BNMetrics(mat1, mat2, node_names=["X1", "X2"])
```

## `BNMetrics.compare_df`

```python
BNMetrics.compare_df(descriptive_metrics='All', comparison_metrics='All')
```

Compile and merge descriptive and comparative metrics into a single table.

### 🔢 Descriptive Metrics

| Metric | Description
|----------------------------------|--------------------------------------|
| `n_edges`              | Total number of edges|
| `n_nodes`              | Total number of nodes|
| `n_colliders`          | Number of collider structures (X → Z ← Y)|
| `n_root_nodes`         | Nodes with no parents or connected undirected edges|
| `n_leaf_nodes`         | Nodes with no children or connected undirected edges|
| `n_isolated_nodes`     | Nodes with no connected edges|
| `n_directed_arcs`      | Number of directed edges|
| `n_undirected_arcs`    | Number of undirected edges|
| `n_reversible_arcs`    | Directed edges not part of any collider|
| `n_in_degree`          | Number of incoming edges|
| `n_out_degree`         | Number of outgoing edges|
### ⚖️ Comparative Metrics

| Metric         | Description|
|----------------|------------|
| `additions`    | Edges present in G2 but not in G1 (ignoring direction)|
| `deletions`    | Edges present in G1 but not in G2 (ignoring direction)|
| `reversals`    | Directed edges that were reversed or became undirected|
| `shd`          | Structural Hamming Distance (additions + deletions + reversals)|
| `hd`           | Hamming Distance (additions + deletions only)|
| `tp`           | Edges presented in two graphs|
| `fp`           | Edges in G2 not in G1|
| `fn`           | Missing edges in G2 that were in G1|
| `precision`    | TP / (TP + FP)|
| `recall`       | TP / (TP + FN)|
| `f1_score`     | Harmonic mean of precision and recall
|

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
