# BNM: Bayesian Network Metrics

**BNM** is a Python package for evaluating, comparing, and visualizing DAGs. It provides an intuitive interface for exploring both global and local graph structures, offering a rich set of metrics and visual tools. 

Originally developed as [DAGMetrics](https://github.com/averinpa/DAGMetrics) in R for analyzing Bayesian Networks in microbial abundance data [(Averin et al., 2025)](https://doi.org/10.3390/agronomy15040987), **BNM** is the Python implementation that expands this functionality.


---
## 🚀 Key Features

- **Descriptive Metrics**: Analyze structural properties of individual DAGs — including number of edges, colliders, root/leaf nodes, and more.
- **Comparative Metrics**: Quantify similarity between DAGs using metrics like Structural Hamming Distance (SHD), Hamming Distance (HD), true/false positives, F1 score, and others.
- **Local Structure Analysis**: Explore and compare the Markov blankets of selected nodes to understand the structure of a system at a granular level.
- **Visual Comparisons**: Generate side-by-side visualizations of DAGs, highlighting shared edges.
- **Batch Evaluation**: Compare multiple models (e.g., from different algorithm runs or hyperparameter settings) to assess model stability and complexity.

---

## 📦 Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/averinpa/bnm.git
```

---

## 📚 Documentation

- [User Guide and API Reference](https://github.com/averinpa/bnm/blob/main/docs/index.md)
- [R Version of DAGMetrics](https://github.com/averinpa/DAGMetrics)
- [Evaluating Directed Acyclic Graphs with DAGMetrics: Insights from Tuber and Soil Microbiome Data (Averin et al., 2025)](https://doi.org/10.3390/agronomy15040987)

---

## 📬 License

This project is licensed under the [MIT License](LICENSE).

---

## ✍️ Author

**Pavel Averin**  
GitHub: [@averinpa](https://github.com/averinpa)

