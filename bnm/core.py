import pandas as pd
import numpy as np
pd.options.display.max_columns = 999
import networkx as nx
import graphviz

from IPython.display import display, HTML

from bnm.utils import mark_and_collapse_bidirected_edges, get_markov_blanket_subgraph
from bnm import metrics as m



class BNMetrics:
    def __init__(self, G1, G2=None, node_names=None):
        """
        The `BNMetrics` class computes and compares descriptive and comparative 
        metrics between one or two Bayesian networks (DAGs), with support for 
        visualization.

        Initialize a BNMetrics object with one or two DAGs. 
        This class supports flexible input formats for causal structure comparison. 
        The graphs can be provided either as `networkx.DiGraph` objects or as 
        adjacency matrices (NumPy arrays or list-of-lists). If matrices are 
        passed, `node_names` must also be provided to assign names to the nodes.

        All edges are processed to detect and mark bidirected edges as "undirected". 
        Bidirected edges are collapsed into one edge. Directed edges are marked 
        with "directed". Subgraphs for each node’s Markov blanket are computed and 
        stored for downstream metric calculations and visualizations.

        Parameters
        ----------
        G1 : nx.DiGraph or np.ndarray or list of lists
            The first graph (base DAG). If not a DiGraph, it must be a square adjacency matrix.

        G2 : nx.DiGraph or np.ndarray or list of list, optional
            The second graph (comparison DAG). Must have the same node names and structure 
            as G1. If not provided, BNMetrics operates in single-graph mode.

        node_names : list of str, optional
            Required only when G1 and G2 is given as a NumPy array or list of lists.
            Length must match number of nodes.

        Raises
        ------
        ValueError
            If G1 or G2 is not square when provided as a matrix.
            If node_names are missing or do not match the number of nodes.
            If G1 and G2 do not share the same set of node names.

        Examples
        --------
        >>> import networkx as nx
        >>> from bnm import BNMetrics
        >>> G1 = nx.DiGraph()
        >>> G1.add_edges_from([("A", "B"), ("C", "B")])
        >>> G2 = nx.DiGraph()
        >>> G2.add_edges_from([("A", "B"), ("B", "C")])
        >>> model = BNMetrics(G1, G2)

        >>> # With NumPy arrays
        >>> import numpy as np
        >>> mat1 = np.array([[0, 1], [0, 0]])
        >>> mat2 = np.array([[0, 0], [1, 0]])
        >>> model = BNMetrics(mat1, mat2, node_names=["X1", "X2"])
        """

        self.G1_raw = G1
        self.G2_raw = G2

        # Case: G1 is DiGraph and G2 is matrix → extract node names from G1
        if isinstance(G1, nx.DiGraph) and isinstance(G2, (np.ndarray, list)):
            inferred_node_names = list(G1.nodes)
            self.G1 = mark_and_collapse_bidirected_edges(G1)
            self.G2 = self._convert_to_digraph(G2, inferred_node_names, name="G2")

        # Case: G1 is matrix and G2 is DiGraph → extract node names from G2
        elif isinstance(G1, (np.ndarray, list)) and isinstance(G2, nx.DiGraph):
            inferred_node_names = list(G2.nodes)
            self.G1 = self._convert_to_digraph(G1, inferred_node_names, name="G1")
            self.G2 = mark_and_collapse_bidirected_edges(G2)

        # Case: both are matrices → need node_names
        elif isinstance(G1, (np.ndarray, list)) and isinstance(G2, (np.ndarray, list)):
            if node_names is None:
                raise ValueError("node_names must be provided when both G1 and G2 are matrices.")
            self.G1 = self._convert_to_digraph(G1, node_names, name="G1")
            self.G2 = self._convert_to_digraph(G2, node_names, name="G2")

        # Case: G1 is matrix and G2 is None
        elif isinstance(G1, (np.ndarray, list)) and G2 is None:
            if node_names is None:
                raise ValueError("node_names must be provided when G1 is a matrix and G2 is None.")
            self.G1 = self._convert_to_digraph(G1, node_names, name="G1")
            self.G2 = None

        # Case: both are DiGraphs → just validate and process
        elif isinstance(G1, nx.DiGraph) and (G2 is None or isinstance(G2, nx.DiGraph)):
            self.G1 = mark_and_collapse_bidirected_edges(G1)
            if G2 is not None:
                if set(G1.nodes()) != set(G2.nodes()):
                    raise ValueError("G1 and G2 must have the same node names.")
                self.G2 = mark_and_collapse_bidirected_edges(G2)
            else:
                self.G2 = None

        # Unsupported input types
        else:
            raise TypeError("G1 and G2 must each be a networkx.DiGraph, numpy.ndarray, or list of lists.")

        # Build internal graph dictionary
        if self.G2 is not None:
            self.graph_dict = self._build_graph_dict_two_graphs(self.G1, self.G2)
        else:
            self.graph_dict = self._build_graph_dict_one_graph(self.G1)

    def _convert_to_digraph(self, G, node_names, name="G"):
        """
        Convert supported formats (DiGraph, ndarray, list of lists) into DiGraph.
        """
        if isinstance(G, nx.DiGraph):
            return mark_and_collapse_bidirected_edges(G)

        elif isinstance(G, (np.ndarray, list)):
            if isinstance(G, list):
                G = np.array(G)

            if G.shape[0] != G.shape[1]:
                raise ValueError(f"{name} must be a square matrix.")

            if node_names is None or len(node_names) != G.shape[0]:
                raise ValueError(f"node_names must be provided and match the shape of {name}.")

            # Convert using networkx and relabel nodes
            graph = nx.from_numpy_array(G, create_using=nx.DiGraph)
            mapping = {i: node_names[i] for i in range(len(node_names))}
            graph = nx.relabel_nodes(graph, mapping)

            return mark_and_collapse_bidirected_edges(graph)

        else:
            raise TypeError(f"{name} must be a networkx.DiGraph, numpy.ndarray, or list of lists.")

    def _build_graph_dict_two_graphs(self, G1, G2):
        """
        Internal method to construct dictionary of MB subgraphs.
        """
        graph_dict = {}

        for i in ['All'] + list(G1.nodes()):
            graph_dict[i] = {'d1': '', 'd2': '', 'd3': ''}

            if i == 'All':
                graph_dict[i]['d1'] = G1
                graph_dict[i]['d2'] = G2
                graph_dict[i]['d3'] = G2
            else:
                d1 = get_markov_blanket_subgraph(G1, i)
                d2 = get_markov_blanket_subgraph(G2, i)

                graph_dict[i]['d1'] = d1
                graph_dict[i]['d2'] = d2

                nodes_d1 = set(d1.nodes())
                d3 = G2.subgraph(nodes_d1).copy()
                graph_dict[i]['d3'] = d3

        return graph_dict
    
    def _build_graph_dict_one_graph(self, G1):
        """
        Internal method to construct dictionary of MB subgraphs.
        """
        graph_dict = {}

        for i in ['All'] + list(G1.nodes()):
            graph_dict[i] = {'d1': ''}

            if i == 'All':
                graph_dict[i]['d1'] = G1
            else:
                d1 = get_markov_blanket_subgraph(G1, i)
                
                graph_dict[i]['d1'] = d1
                
        return graph_dict
    
    def _merge_graphs_no_duplicates_clean(self, nodes, layer):
        """
        Merge subgraphs from self.graph_dict[node][layer] into one DiGraph,
        including all unique nodes and edges (no duplicates).

        Parameters:
        - nodes: list of node names (keys in self.graph_dict)
        - layer: 'd1', 'd2', or 'd3'

        Returns:
        - A merged networkx.DiGraph
        """
        merged = nx.DiGraph()
        added_edges = set()
        added_nodes = set()

        for node in nodes:
            g = self.graph_dict[node][layer]

            for n in g.nodes():
                if n not in added_nodes:
                    merged.add_node(n)
                    added_nodes.add(n)

            for u, v in g.edges():
                edge_type = g[u][v].get("type", "directed")
                edge_id = (u, v, edge_type)
                if edge_id not in added_edges:
                    merged.add_edge(u, v, type=edge_type)
                    added_edges.add(edge_id)

        return merged

    def compile_descriptive_metrics(self, metric_names="All"):
        """
        Compile a full descriptive metrics table for each node (and 'All') using d1 and d2 graphs
        from self.graph_dict.

        Parameters:
        metric_names (list[str] or "All"): Descriptive metric base names (e.g., 'count_edges').
            If "All", all supported metrics are computed.

        Returns:
        pd.DataFrame: Table of descriptive metrics with one row per node and columns for
                    both base ('_base') and comparison ('') metrics.
        """
        graph_dict = self.graph_dict

        available = {
            "n_edges": m.count_edges,
            "n_nodes": m.count_nodes,
            "n_colliders": m.count_colliders,
            "n_root_nodes": m.count_root_nodes,
            "n_leaf_nodes": m.count_leaf_nodes,
            "n_isolated_nodes": m.count_isolated_nodes,
            "n_directed_arcs": m.count_directed_arcs,
            "n_undirected_arcs": m.count_undirected_arcs,
            "n_reversible_arcs": m.count_reversible_arcs,
            "n_in_degree": m.count_in_degree,
            "n_out_degree": m.count_out_degree
        }

        if metric_names == "All":
            metric_names = list(available.keys())

        columns = ['node_name']
        for name in metric_names:
            if self.G2 is not None:
                columns.append(name + "_base")
            columns.append(name)

        df = pd.DataFrame(np.full((len(graph_dict), len(columns)), np.nan), columns=columns)
        df['node_name'] = list(graph_dict.keys())

        for node in graph_dict:
            d1 = graph_dict[node]["d1"]
            if self.G2 is not None:
                d2 = graph_dict[node]["d2"]

            for name in metric_names:
                func = available.get(name)
                if not func:
                    continue

                if "in_degree" in name or "out_degree" in name:
                    if self.G2 is not None:
                        df.loc[df["node_name"] == node, name + "_base"] = func(d1, node)
                        df.loc[df["node_name"] == node, name] = func(d2, node)
                    else:
                        df.loc[df["node_name"] == node, name] = func(d1, node)
                        
                else:
                    if self.G2 is not None:
                        df.loc[df["node_name"] == node, name + "_base"] = func(d1)
                        df.loc[df["node_name"] == node, name] = func(d2)
                    else:
                        df.loc[df["node_name"] == node, name] = func(d1)

        return df

    def compile_comparison_metrics(self, metric_names="All"):
        """
        Compile a full comparison metrics table for each node (and 'All') using d1 and d2 subgraphs.

        Parameters:
        metric_names (list[str] or "All"): Names of comparison metrics to compute.
            If "All", includes all supported metrics.

        Returns:
        pd.DataFrame: Table of comparison metrics with one row per node.
        """
        graph_dict = self.graph_dict

        available = {
            "additions": m.count_additions,
            "deletions": m.count_deletions,
            "reversals": m.count_reversals,
            "shd": m.shd,
            "hd": m.hd,
            "tp": m.count_true_positives,
            "fp": m.count_false_positives,
            "fn": m.count_false_negatives,
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score
        }

        if metric_names == "All":
            metric_names = list(available.keys())

        columns = ["node_name"] + metric_names
        df = pd.DataFrame(index=range(len(graph_dict)), columns=columns)
        df["node_name"] = list(graph_dict.keys())

        for node in graph_dict:
            d1 = graph_dict[node]["d1"]
            d2 = graph_dict[node]["d2"]

            for name in metric_names:
                func = available.get(name)
                if func:
                    df.loc[df["node_name"] == node, name] = func(d1, d2)

        return df

    def compare_df(self, descriptive_metrics="All", comparison_metrics="All"):
        """
        Compile and return a merged DataFrame containing descriptive and/or comparison metrics
        for each node (and global metrics under 'All').

        This method supports flexible inclusion of:
        - Descriptive metrics based on a single DAG or both DAGs
        - Comparative metrics when both DAGs are provided

        Parameters
        ----------
        descriptive_metrics : list[str] or "All" or None, default="All"
            List of descriptive metric names to compute, or "All" to include all available.
            If None, descriptive metrics are not included.

        comparison_metrics : list[str] or "All" or None, default="All"
            List of comparison metric names to compute, or "All" to include all available.
            If None, comparison metrics are not included.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame with one row per node (including 'All' for global metrics).
            Columns will depend on the selected metrics. Returns None if no valid metrics were specified.

        Notes
        -----
        Descriptive metrics (available with one or two graphs):
        - 'n_edges': Number of edges
        - 'n_nodes': Number of nodes
        - 'n_colliders': Number of collider structures
        - 'n_root_nodes': Root nodes (no parents)
        - 'n_leaf_nodes': Leaf nodes (no children)
        - 'n_isolated_nodes': Nodes with no edges
        - 'n_directed_arcs': Number of directed edges
        - 'n_undirected_arcs': Number of undirected edges
        - 'n_reversible_arcs': Directed edges not involved in colliders
        - 'n_in_degree': In-degree (only for individual nodes)
        - 'n_out_degree': Out-degree (only for individual nodes)

        Comparative metrics (requires both graphs):
        - 'additions': Edges in G2 but not in G1
        - 'deletions': Edges in G1 but not in G2
        - 'reversals': Reversed edges or edge type changes
        - 'shd': Structural Hamming Distance
        - 'hd': Hamming Distance (additions + deletions only)
        - 'tp': True Positives
        - 'fp': False Positives
        - 'fn': False Negatives
        - 'precision': Precision = TP / (TP + FP)
        - 'recall': Recall = TP / (TP + FN)
        - 'f1_score': F1 score = 2 * (precision * recall) / (precision + recall)

        Behavior
        --------
        - If only one graph is provided (G2 is None), only descriptive metrics can be computed.
        - If no valid metrics are specified, the method prints a warning and returns None.

        Example
        -------
        >>> from bnmetrics import BNMetrics
        >>> import networkx as nx
        >>> G1 = nx.DiGraph()
        >>> G1.add_edges_from([("A", "B"), ("B", "C")])
        >>> G2 = nx.DiGraph()
        >>> G2.add_edges_from([("A", "B"), ("C", "B")])
        >>> bn = BNMetrics(G1, G2)
        >>> df = bn.compare_df(
        ...     descriptive_metrics=["n_edges", "n_colliders"],
        ...     comparison_metrics=["shd", "tp", "fp"]
        ... )
        >>> print(df)
        """
        if self.G2 is not None:
            if descriptive_metrics is None and comparison_metrics is None:
                print('please specify metrics you want to see')
                return None
            elif descriptive_metrics is not None and comparison_metrics is None:
                desc_df = self.compile_descriptive_metrics(metric_names=descriptive_metrics)
                return desc_df
            elif descriptive_metrics is not None and comparison_metrics is not None:
                desc_df = self.compile_descriptive_metrics(metric_names=descriptive_metrics)
                comp_df = self.compile_comparison_metrics(metric_names=comparison_metrics)
                combined = pd.merge(desc_df, comp_df, on="node_name", how="left")
                return combined
            elif descriptive_metrics is None and comparison_metrics is not None:
                comp_df = self.compile_comparison_metrics(metric_names=comparison_metrics)
                return comp_df
        
        else:
            if descriptive_metrics is None and comparison_metrics is None:
                print('please specify descriptive metrics you want to see')
                return None
            elif descriptive_metrics is not None and comparison_metrics is None:
                desc_df = self.compile_descriptive_metrics(metric_names=descriptive_metrics)
                return desc_df
            elif descriptive_metrics is not None and comparison_metrics is not None:
                desc_df = self.compile_descriptive_metrics(metric_names=descriptive_metrics)
                print('comparative metrics only available when there are two graphs')
                return desc_df
            elif descriptive_metrics is None and comparison_metrics is not None:
                print('comparative metrics only available when there are two graphs')
                return None

    def _mark_true_positives_color_both(self, nodes, option=1):
        """
        Return copies of merged G1 and G2 subgraphs (from the specified layer)
        where matching edges (true positives) are marked with color='green'.

        Matching rules:
        - Directed: must match direction and type
        - Undirected: must match type and connect same nodes in any direction

        Parameters:
        - nodes: List of nodes to merge subgraphs for
        - layer: One of 'd1', 'd2', or 'd3' (defaults to 'd1'/'d2')

        Returns:
        - Tuple of colored G1 and G2 merged subgraphs
        """
        if option == 1:
            layer = "d1"
            layer2 = "d2"
        elif option == 2:
            layer = "d1"
            layer2 = "d3"

        G1 = self._merge_graphs_no_duplicates_clean(nodes, layer)
        G2 = self._merge_graphs_no_duplicates_clean(nodes, layer2)

        G1_marked = G1.copy()
        G2_marked = G2.copy()

        for u, v in G1.edges():
            type1 = G1[u][v].get("type", "directed")

            if type1 == "directed":
                if G2.has_edge(u, v) and G2[u][v].get("type") == "directed":
                    G1_marked[u][v]["color"] = "green"
                    G2_marked[u][v]["color"] = "green"

            elif type1 == "undirected":
                if (G2.has_edge(u, v) and G2[u][v].get("type") == "undirected"):
                    G1_marked[u][v]["color"] = "green"
                    G2_marked[u][v]["color"] = "green"
                elif (G2.has_edge(v, u) and G2[v][u].get("type") == "undirected"):
                    G1_marked[u][v]["color"] = "green"
                    G2_marked[v][u]["color"] = "green"

        return G1_marked, G2_marked

    def _color_selected_nodes_green(self, G1, G2, nodes):
        """
        Given two graphs and a list of node names, mark those nodes with color='green' 
        in both graphs (except if node == 'All').

        Parameters:
        - G1, G2: DiGraphs to color
        - nodes: list of node names

        Returns:
        - Tuple of updated copies (G1_colored, G2_colored)
        """
        G1_colored = G1.copy()
        G2_colored = G2.copy()

        for node in nodes:
            if node == 'All':
                continue
            if node in G1_colored.nodes:
                G1_colored.nodes[node]['color'] = 'green'
            if node in G2_colored.nodes:
                G2_colored.nodes[node]['color'] = 'green'

        return G1_colored, G2_colored

    def _display_graphs_side_by_side_highlighted(self, G1, G2, name1 = "DAG 1", name2 = "DAG 2"):
        """
        Display two networkx graphs side by side with their names above them.
        - Nodes and edges with attribute color='green' will be highlighted.
        - Edges with type='undirected' are rendered without arrowheads.
        """

        def to_graphviz(graph, name):
            dot = graphviz.Digraph(name=name)

            for node, attrs in graph.nodes(data=True):
                if attrs.get("color") == "green":
                    dot.node(str(node), style="filled", fillcolor="lightgreen")
                else:
                    dot.node(str(node))

            for u, v, attrs in graph.edges(data=True):
                edge_type = attrs.get("type", "directed")
                edge_color = "green" if attrs.get("color") == "green" else None

                if edge_type == "undirected":
                    dot.edge(str(u), str(v), dir="none", style="solid", color=edge_color)
                else:
                    dot.edge(str(u), str(v), color=edge_color)

            return dot.pipe(format='svg').decode("utf-8")

        svg1 = to_graphviz(G1, name1)
        svg2 = to_graphviz(G2, name2)

        display(HTML(f"""
        <div style="display: flex; gap: 60px;">
            <div style="text-align: center;">
                <h3>{name1}</h3>
                {svg1}
            </div>
            <div style="text-align: center;">
                <h3>{name2}</h3>
                {svg2}
            </div>
        </div>
        """))

    def compare_two_bn(self, nodes, option=1, name1='DAG1', name2='DAG2'):
        """
        Visually compare two DAGs (G1 vs. G2) side-by-side using a subset of nodes.

        This method highlights:
        - True positive edges (present in both graphs) in green.
        - Selected nodes in green.
        - Edge types (directed or undirected) are preserved visually.

        Parameters
        ----------
        nodes : list of str
            List of node names to include in the visualization. Use node names from the DAGs.

        option : int, default=1
            Determines which layer to visualize:
            - 1: Use the full Markov blanket subgraph (d1 and d2).
            - 2: Use d3 layer (nodes from d1 but edges from d2). Intended for structure comparison.

        name1 : str, default="DAG1"
            Title to display above the first graph (G1).

        name2 : str, default="DAG2"
            Title to display above the second graph (G2).

        Returns
        -------
        None
            Displays two graph visualizations side-by-side using Graphviz in Jupyter.

        Raises
        ------
        ValueError
            If no second graph (G2) was provided during initialization.

        Example
        -------
        >>> bn = BNMetrics(G1, G2)
        >>> bn.compare_two_bn(nodes=['X_1', 'X_5', 'X_10'], option=2, name1='Original', name2='Modified')
        """
        if self.G2 is not None:
            G1, G2 = self._mark_true_positives_color_both(nodes, option)
            G1, G2 = self._color_selected_nodes_green(G1, G2, nodes)
            self._display_graphs_side_by_side_highlighted(G1, G2, name1=name1, name2=name2)
        else:
            print('you need to specify the second DAG when instantiating the object BNMetrics')

    def _color_selected_nodes_green_single(self, G, nodes):
        """
        Given one graph and a list of node names, mark those nodes with color='green'
        (except if node == 'All').

        Parameters:
        - G: a NetworkX DiGraph
        - nodes: list of node names to highlight

        Returns:
        - A copy of the graph with updated node attributes
        """
        G_colored = G.copy()

        for node in nodes:
            if node == 'All':
                continue
            if node in G_colored.nodes:
                G_colored.nodes[node]['color'] = 'green'

        return G_colored

    def _display_graph_with_node_highlight(self, G, name="DAG"):
        """
        Display a single networkx graph using Graphviz.
        - Nodes with color='green' are highlighted in light green.
        - Edges use the 'type' attribute for styling:
            - 'directed': default arrow
            - 'undirected': no arrowhead (dir='none')
        - Edges are not color-highlighted.
        """

        dot = graphviz.Digraph(name=name)

        # Add nodes
        for node, attrs in G.nodes(data=True):
            if attrs.get("color") == "green":
                dot.node(str(node), style="filled", fillcolor="lightgreen")
            else:
                dot.node(str(node))

        # Add edges
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get("type", "directed")
            if edge_type == "undirected":
                dot.edge(str(u), str(v), dir="none", style="solid")
            else:
                dot.edge(str(u), str(v))

        svg = dot.pipe(format='svg').decode("utf-8")

        display(HTML(f"""
        <div style="text-align: center;">
            <h3>{name}</h3>
            {svg}
        </div>
        """))

    def plot_bn(self, nodes, layer="d1", title="DAG"):
        """
        Plot a single DAG composed of merged Markov Blanket subgraphs for the specified nodes.

        This method constructs a graph by merging subgraphs from a specific layer ('d1', 'd2', or 'd3')
        for each node in the list, highlights the selected nodes in green, and renders the network 
        using Graphviz.

        Parameters
        ----------
        nodes : list of str
            A list of node names (e.g., ['X_1', 'X_2', ...]) to extract subgraphs from `self.graph_dict`.

        layer : str, default="d1"
            The subgraph layer to visualize:
            - 'd1' : Markov Blanket from G1 (always available)
            - 'd2' : Markov Blanket from G2 (requires G2)
            - 'd3' : Subgraph from G2 using nodes from G1's MB (requires G2)

        title : str, default="DAG"
            Title displayed above the plotted graph.

        Returns
        -------
        None
            Displays the graph inline in Jupyter using Graphviz.

        Raises
        ------
        ValueError
            If `layer` is 'd2' or 'd3' but no second graph (G2) was provided during initialization.

        Example
        -------
        >>> bn = BNMetrics(G1)
        >>> bn.plot_bn(nodes=['X_1', 'X_2'], layer='d1', title="Local Structure")

        >>> bn = BNMetrics(G1, G2)
        >>> bn.plot_bn(nodes=['X_3'], layer='d3', title="Comparison MB Overlay")
        """
            
        if self.G2 is None and layer in ['d2', 'd3']:
            print('please specify the second graph when instantiating the BNMetrics object')
        else:
            G = self._merge_graphs_no_duplicates_clean(nodes, layer)
            G = self._color_selected_nodes_green_single(G, nodes)
            self._display_graph_with_node_highlight(G, name=title)

