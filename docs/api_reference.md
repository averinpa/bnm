# API Reference: BNM (Bayesian Network Metrics)
## Class BNMetrics                      <a href="https://github.com/averinpa/bnm/blob/main/bnm/core.py#L14" style="float: right; font-weight: normal;">[source]</a>



The `BNMetrics` class calculates and compares descriptive and comparative metrics for one or two DAGs, with support for visualizations. It supports flexible input types including NetworkX DiGraph and adjacency matrices.

---

### Initialization

```python
from bnm import BNMetrics

bnm_obj = BNMetrics(G1, G2=None, node_names=None, mb_nodes='All')
```
### Parameters

**G1** : `nx.DiGraph`, `np.ndarray`, or `list of lists`  
: The first DAG. If not a DiGraph, it must be a square adjacency matrix.

**G2** : `nx.DiGraph`, `np.ndarray`, or `list of lists`, default=`None`  
: The second DAG. Must have the same node names. If not provided, BNMetrics operates in single-DAG mode.

**node_names** : `list of str`, optional  
: Required only when `G1`, `G2` or both are given as a NumPy array or list of lists.
Length must match number of nodes.
**mb_nodes**: `str` or `list`, default = `'All'`  
  Nodes for which Markov blanket-based metrics and subgraphs will be computed.

### Raises

**ValueError**  
- If `G1` or `G2` is not square when passed as a matrix.  
- If `node_names` are missing or mismatched.  
- If `G1` and `G2` have different node sets.  

**TypeError**  
- If unsupported input types are provided for `G1` or `G2`.

### Internal Behavior

- Edges are processed to detect bidirected edges and bidirected pairs into undirected edges.
- All edges are labeled as `"directed"` or `"undirected"` accordingly.
- Subgraphs representing each node’s Markov blanket are computed and stored.
- These subgraphs are used for calculating local and global metrics and for visualization.

---
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

## `BNMetrics.compare_df`                       <a href="https://github.com/averinpa/bnm/blob/main/bnm/core.py#L347" style="float: right; font-weight: normal;">[source]</a>

```python
BNMetrics.compare_df(descriptive_metrics='All', comparison_metrics='All')
```

Calculates and returns a DataFrame of descriptive and/or comparison metrics for the Markov blanket of each node, as well as global metrics under the label 'All'.  

If only one graph is provided (G2 is None), only descriptive metrics can be computed.

This method supports flexible inclusion of:  
- Descriptive metrics computed from one or both DAGs  
- Comparative metrics when both DAGs are provided

### Descriptive Metrics

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
### Comparative Metrics

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
| `f1_score`     | Harmonic mean of precision and recall|


### Parameters

**descriptive_metrics** : `list[str]`, `'All'` or `None`, default=`'All'`  
: List of descriptive metric names to compute, or "All" to include all available.
If None, descriptive metrics are not included.  
**comparison_metrics** : `list[str]`, `'All'` or `None`, default=`'All'`  
:  List of comparison metric names to compute, or "All" to include all available.
If None, comparison metrics are not included.

### Returns

**pd.DataFrame** or **None**  
: A DataFrame with one row per node (including 'All' for global metrics).
Columns will depend on the selected metrics. Returns None if no valid metrics were specified.

### Example

```python
from bnm import BNMetrics
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
G2 = nx.DiGraph()
G2.add_edges_from([("A", "B"), ("C", "B")])
bn = BNMetrics(G1, G2)
df = bn.compare_df(
    descriptive_metrics=["n_edges", "n_colliders"],
    comparison_metrics=["shd", "tp", "fp"])
```
| node_name | n_edges_base | n_edges | n_colliders_base | n_colliders | shd | tp | fp |
|-----------|--------------|---------|------------------|-------------|-----|----|----|
| All       | 2.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
| A         | 1.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
| B         | 2.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
| C         | 1.0          | 2.0     | 0.0              | 1.0         | 2   | 0  | 2  |

---

## `BNMetrics.compare_two_bn`                       <a href="https://github.com/averinpa/bnm/blob/main/bnm/core.py#L562" style="float: right; font-weight: normal;">[source]</a>

```python
BNMetrics.compare_two_bn(nodes, option=1, name1='DAG1', name2='DAG2')
```

 Visually compare two DAGs (G1 vs. G2) side-by-side using a subset of nodes.

This method highlights:
- True positive edges (present in both graphs) in green.
- Selected nodes in green.
- Edge types (directed or undirected) are preserved visually.

### Parameters
**nodes**: `list[str]`  
: List of node names to include in the visualization. These must match node names in the DAGs. The subgraph containing these nodes and their Markov blankets will be extracted and visualized.  
**option**: `int`, `default=1`   
: Controls the visualization strategy:
  - 1: Shows the Markov blankets from G1 and G2 side-by-side.  
  - 2: Shows the Merkov blanket from G1 and the same set of nodes in G2.

**name1**: `str`, `default="DAG1"`  
: Title label for the first graph.  
**name2**: `str`, `default="DAG2"`  
: Title label for the second graph.

### Returns

- `None`  
  Displays two DAGs side-by-side using Graphviz within a Jupyter notebook environment.

### Example
```python
from bnm import BNMetrics
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
G2 = nx.DiGraph()
G2.add_edges_from([("A", "B"), ("C", "B")])
bn = BNMetrics(G1, G2)
bn.compare_two_bn(nodes=['A', 'B'], option=1, name1='Original', name2='Modified')
```

---

## `BNMetrics.plot_bn`                      <a href="https://github.com/averinpa/bnm/blob/main/bnm/core.py#L667" style="float: right; font-weight: normal;">[source]</a>

```python
BNMetrics.plot_bn(nodes, layer="d1", title="DAG")
```

 Plot a single DAG composed of merged Markov Blanket subgraphs for the specified nodes.  
This method constructs a graph by merging subgraphs from a specific layer ('d1', 'd2', or 'd3') for each node in the list, highlights the selected nodes in green, and renders the network using Graphviz.

### Parameters
**nodes**: `list[str]`  
: A list of node names (e.g., ['X_1', 'X_2', ...]) to extract subgraphs from `self.graph_dict`.  
**layer**: `str`, `default="d1"`  
The subgraph layer to visualize:
- 'd1' : Markov Blanket from G1 (always available)
- 'd2' : Markov Blanket from G2 (requires G2)
- 'd3' : Subgraph from G2 using nodes from G1's MB (requires G2)  
**title**: `str`, `default="DAG"`  
:  Title displayed above the plotted graph.

### Returns

- `None`  
  Displays a DAG using Graphviz within a Jupyter notebook environment.

### Example
```python
from bnm import BNMetrics
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
bn = BNMetrics(G1)
bn.plot_bn(nodes=['A', 'B'], layer='d1', title='Markov Blanket')
```

---

## `compare_models_descriptive`                                 <a href="https://github.com/averinpa/bnm/blob/main/bnm/viz.py#L81" style="float: right; font-weight: normal;">[source]</a>

```python
compare_models_descriptive(list_of_dags, 
                            model_names, 
                            node_names, 
                            mb_nodes)
```

Calculates descriptive metrics—including number of edges, colliders, root nodes, leaf nodes, isolated nodes, directed arcs, undirected arcs, and reversible arcs—for the global structure and specified Markov blankets. The results are displayed in eight interactive subplots, with a dropdown menu allowing selection of the Markov blanket of interest.

### Parameters  

**list_of_dags**: `list[nx.DiGraph]`, `list[np.ndarray]` or `list[list of lists]`  
: A list of DAGs to compare.  
**model_names**: `list`  
: A list of model names corresponding to list_of_dags.  
**node_names**: `list`
: A list of all node names in the associated with a DAG.
**mb_nodes**: `list`
: A list of nodes to compute Markov blanket-based descriptive metrics for.

### Returns

- `None`  
  Displays the descriptive metrics in eight interactive subplots

### Example
```python
from bnm import compare_models_descriptive
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
G2 = nx.DiGraph()
G2.add_edges_from([("A", "B"), ("A", "C")])
compare_models_descriptive(list_of_dags=[G1, G2], 
                            model_names=['Model1', 'Model2'], 
                            node_names=list(G1.nodes), 
                            mb_nodes=['A', 'B'])
```

---

## `compare_models_comparative`                                 <a href="https://github.com/averinpa/bnm/blob/main/bnm/viz.py#L181" style="float: right; font-weight: normal;">[source]</a>

```python
compare_models_comparative(list_of_dags, 
                            model_names, 
                            node_names,
                            metric, 
                            mb_nodes)
```

Calculates a comparative metric—selected from additions, deletions, reversals, SHD, HD, TP, FN, FP, precision, recall, or F1 score—for the global structure and specified Markov blankets. The results are visualized in a heatmap comparing all pairs of models based on the chosen metric, with a dropdown menu for selecting the Markov blanket of interest.

### Parameters  

**list_of_dags**: `list[nx.DiGraph]`, `list[np.ndarray]` or `list[list of lists]`  
: A list of DAGs to compare.  
**model_names**: `list`  
: A list of model names corresponding to list_of_dags.  
**node_names**: `list`
: A list of all node names in the associated with a DAG.
**metric**: `str`
: metric to be calculated. Choices are additions, deletions, reversals, shd, hd, tp, fn, fp, precision, recall, or f1_score
**mb_nodes**: `list`
: A list of nodes to compute Markov blanket-based comparative metric for.

### Returns

- `None`  
  Displays the heatmap comparing all pairs of models

### Example
```python
from bnm import compare_models_comparative
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
G2 = nx.DiGraph()
G2.add_edges_from([("A", "B"), ("A", "C")])
compare_models_comparative(list_of_dags=[G1, G2], 
                            model_names=['Model1', 'Model2'], 
                            node_names=list(G1.nodes), 
                            metric='shd',
                            mb_nodes=['A', 'B'])
```
## `analyse_mb`                                 <a href="https://github.com/averinpa/bnm/blob/main/bnm/viz.py#273" style="float: right; font-weight: normal;">[source]</a>

```python
analyse_mb(G1, node_names=None, mb_nodes='All')
```

Analyze the Markov Blanket space of a DAG and plot distribution of descriptive metrics.

### Parameters  

**G1** : `nx.DiGraph`, `np.ndarray`, or `list of lists`  
: The first DAG (base DAG). If not a DiGraph, it must be a square adjacency matrix.
**node_names** : `list of str`, optional  
: Required only when `G1`is a NumPy array or list of lists.
Length must match number of nodes.  
**mb_nodes**: `str` or `list`, default = `'All'`  
  Nodes for which Markov blanket-based metrics and subgraphs will be computed.

### Returns

- `None`  
  Displays the distibution of descriptive metrics in eight interactive subplots

### Example
```python
from bnm import analyse_mb
import networkx as nx
G1 = nx.DiGraph()
G1.add_edges_from([("A", "B"), ("B", "C")])
analyse_mb(G1, node_names=None, mb_nodes='All')
```
---
## Use Cases: 
- [Evaluate Single DAG](https://github.com/averinpa/bnm/blob/main/use%20cases/evaluate%20single%20DAG.ipynb)
- [Compare Two DAGs](https://github.com/averinpa/bnm/blob/main/use%20cases/compare%20two%20DAGs.ipynb)
- [Use Case: Compare Algorithms](https://github.com/averinpa/bnm/blob/main/use%20cases/compare%20algorithms.ipynb)