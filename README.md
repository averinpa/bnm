# BNM: Bayesian Network Metrics

**BNM** is a Python package for evaluating, comparing, and visualizing DAGs. It provides tools to calculate a wide range of descriptive and comparative metrics on full networks or local structures (e.g., Markov blankets), and supports visualization of these structures with true positive highlights. 

Originally developed as [DAGMetrics](https://github.com/averinpa/DAGMetrics) in R for analyzing Bayesian Networks in microbial abundance data [(Averin et al., 2025)](https://doi.org/10.20944/preprints202503.0943.v1), **BNM** is the Python implementation of that package.


---

## 📦 Installation

```bash
!pip install git+https://github.com/averinpa/bnm.git
```

---

## 🧠 Core Class: `BNMetrics`

```python
from bnm import BNMetrics
```

### 🔍 `compare_df()`
Generate a comprehensive table of descriptive and comparative metrics.

```python
bn = BNMetrics(G1, G2)
df = bn.compare_df(descriptive_metrics='All', comparison_metrics='All')
```

**Arguments:**
- `descriptive_metrics`: List of metrics or 'All'.
- `comparison_metrics`: List of metrics or 'All'.

**Descriptive metrics include:**
- `n_edges`, `n_nodes`, `n_colliders`, `n_root_nodes`, `n_leaf_nodes`
- `n_isolated_nodes`, `n_directed_arcs`, `n_undirected_arcs`
- `n_reversible_arcs`, `n_in_degree`, `n_out_degree`

**Comparative metrics include:**
- `additions`, `deletions`, `reversals`, `shd`, `hd`, `tp`, `fp`, `fn`, `precision`, `recall`, `f1_score`

---

### 📊 `compare_two_bn()`
Visualize the structure of two networks side-by-side, highlighting common edges and selected nodes.

```python
bn.compare_two_bn(nodes=['X_1', 'X_5'], option=1)
```

**Arguments:**
- `nodes`: List of node names to highlight.
- `option`: 1 (Markov blanket from d1 & d2), 2 (structure from G1 and G2), or 3 (common nodes, edges from G2).
- `name1`, `name2`: Custom labels for the two graphs.

---

### 📌 `plot_bn()`
Display a single DAG (or MB subgraph) for a set of nodes, with green highlights on selected nodes.

```python
bn.plot_bn(nodes=['X_1', 'X_5'], layer='d1', title='Markov Blanket')
```

**Arguments:**
- `nodes`: List of node names.
- `layer`: One of 'd1', 'd2', or 'd3'.
- `title`: Title for the plot.

---

## 🧪 Utility Functions

### 🔄 `generate_random_dag()`
Create a synthetic random DAG.

```python
from bnm.utils import generate_random_dag
G = generate_random_dag(n_nodes=50, edge_prob=0.1, seed=42)
```

- Ensures acyclicity via topological ordering.
- Nodes are named `X_1`, `X_2`, ..., `X_n`.

### 🔁 `dag_to_cpdag()`
Convert a DAG to a CPDAG (preserving only colliders and Y-structures).

```python
from bnm.utils import dag_to_cpdag
cpdag = dag_to_cpdag(G)
```

---

## 📬 License
MIT

## ✍️ Author
Pavel Averin
