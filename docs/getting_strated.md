# 🚀 Getting Started with BNM

BNM (**Bayesian Network Metrics**) is a Python package for evaluating and visualizing Bayesian Networks and Directed Acyclic Graphs (DAGs), originally developed for microbial network comparisons, but applicable to any causal graph analysis.

---

## 🔧 Installation

You can install BNM directly from GitHub:

```bash
pip install git+https://github.com/averinpa/bnm.git
```
# 📦 Requirements for BNM

BNM relies on the following Python packages:

| Package     | Version       | Description                                              |
|-------------|---------------|----------------------------------------------------------|
| `networkx`  | >=2.8         | For working with graph structures                        |
| `graphviz`  | >=0.20        | For visualizing DAGs               |
| `pandas`    | >=1.3         | For constructing and manipulating metrics tables         |
| `numpy`     | >=1.21        | For numerical operations and array processing            |

---

## ✅ Basic Usage

### 1. **Import the package**

```python
from bnm import BNMetrics
from bnm.utils import generate_random_dag
```
### 2. **Create or load graphs**  
#### You can generate random DAGs using the built-in utility:

```python
G1 = generate_random_dag(n_nodes=20, edge_prob=0.08, seed=42)
G2 = generate_random_dag(n_nodes=20, edge_prob=0.10, seed=99)
```
### 3. **Initialize the `BNMetrics` object**
```python
bnm = BNMetrics(G1, G2)
```
#### You can also use just one graph:
```python
bnm = BNMetrics(G1)
```
### 4. **Compare graph structures**
#### Generate metrics:
```python
df = bnm.compare_df(
    descriptive_metrics="All",
    comparison_metrics=["shd", "precision", "recall"]
)
print(df)
```
### 5. **Visualize the DAGs**
#### Compare two graphs side by side:
```python
bnm.compare_two_bn(nodes=["X_3", "X_7"], option=1)
```
#### Plot a single DAG:
```python
bnm.plot_bn(nodes=["X_1", "X_5"], layer="d1")
```
## 📚 More

- [API Reference](#) *(coming soon or link to actual API documentation)*
- [Paper (Averin et al., 2025)](https://doi.org/10.20944/preprints202503.0943.v1)
- [R Version of DAGMetrics](https://github.com/averinpa/DAGMetrics)