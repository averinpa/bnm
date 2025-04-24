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
| `plotly`    | >=5.9.0       | For interactive visualizations            |

---

## ✅ Basic Usage

### 1. **Import the package**

```python
from bnm import BNMetrics, generate_random_dag
```
### 2. **Create or load graphs**  
#### You can generate random DAGs using the built-in utility:

```python
G1 = generate_random_dag(n_nodes=20, edge_prob=0.15, seed=3)
G2 = generate_random_dag(n_nodes=20, edge_prob=0.15, seed=2)
```
### 3. **Initialize the `BNMetrics` object**
```python
bnm = BNMetrics(G1, G2)
```
#### You can also use just one graph:
```python
bnm_single = BNMetrics(G1)
```
### 4. **Compare graph structures**
#### Generate metrics:
```python
df = bnm.compare_df(
    descriptive_metrics=['n_edges'],
    comparison_metrics=['shd', 'precision', 'recall']
).query("node_name in ['All', 'X_2', 'X_3', 'X_6']")

print(df)
```
| node_name | n_edges_base | n_edges | shd | precision | recall   |
|-----------|--------------|---------|-----|-----------|----------|
| All       | 28.0         | 28.0    | 48  | 0.107143  | 0.107143 |
| X_2       | 7.0          | 8.0     | 12  | 0.125     | 0.142857 |
| X_6       | 11.0         | 11.0    | 18  | 0.181818  | 0.181818 |
| X_3       | 11.0         | 1.0     | 12  | 0.0       | 0.0      |

### 5. **Visualize the DAGs**
#### Compare two graphs side by side:
```python
bnm.compare_two_bn(
    nodes=['X_6'], 
    name1='DAG 1 Markov blanket of node X_6', 
    name2='DAG 2 Markov blanket of node X_6'
)
```
<p align="center">
  <img src="images/two_dags.png" alt="BNM Graph Comparison" />
</p>

#### Plot a single DAG:
```python
bnm.plot_bn(
    nodes=['X_2', 'X_3'], 
    title='Markov blankets of nodes X_2 and X_3'
)
```
<p align="center">
  <img src="images/one_dag.png" alt="BNM Graph Single" />
</p>

## 📚 More

- [API Reference](https://github.com/averinpa/bnm/blob/main/docs/api_reference.md)  
### Use Cases: 
- [Evaluate Single DAG](https://github.com/averinpa/bnm/blob/main/use%20cases/evaluate%20single%20DAG.ipynb)
- [Compare Two DAGs](https://github.com/averinpa/bnm/blob/main/use%20cases/compare%20two%20DAGs.ipynb)
- [Compare Algorithms](https://github.com/averinpa/bnm/blob/main/use%20cases/compare%20algorithms.ipynb)