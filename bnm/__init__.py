"""
BNM (Bayesian Network Metrics)

A package for evaluating, comparing, and visualizing Bayesian networks (DAGs).

Author: Pavel Averin
"""

from .core import BNMetrics
from .utils import generate_random_dag
from .utils import dag_to_cpdag