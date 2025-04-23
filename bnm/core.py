import pandas as pd
import numpy as np
pd.options.display.max_columns = 999
import networkx as nx
import graphviz

from IPython.display import display, HTML

from bnm.utils import mark_and_collapse_bidirected_edges, get_markov_blanket_subgraph
from bnm import metrics as m



class BNMetrics:

    """
    The BNMetrics class calculates and compares descriptive and comparative metrics 
    for one or two DAGs, with support for visualizations. It supports flexible input 
    types including NetworkX DiGraph and adjacency matrices.

    Parameters
    ----------
    G1 : nx.DiGraph, np.ndarray, or list of lists
        The first DAG. If not a DiGraph, it must be a square adjacency matrix.

    G2 : nx.DiGraph, np.ndarray, or list of lists, optional (default=None)
        The second DAG. Must have the same node names. If not provided, BNMetrics 
        operates in single-DAG mode.

    node_names : list of str, optional
        Required only when G1, G2 or both are given as a NumPy array or list of lists. 
        Length must match number of nodes.

    mb_nodes : str or list, default='All'
        Nodes for which Markov blanket-based metrics and subgraphs will be computed.

    Raises
    ------
    ValueError
        - If G1 or G2 is not square when passed as a matrix.
        - If node_names are missing or mismatched.
        - If G1 and G2 have different node sets.

    TypeError
        - If unsupported input types are provided for G1 or G2.

    Internal Behavior
    -----------------
    - Edges are processed to detect bidirected edges and bidirected pairs into 
    undirected edges.
    - All edges are labeled as "directed" or "undirected" accordingly.
    - Subgraphs representing each node’s Markov blanket are computed and stored.
    - These subgraphs are used for calculating local and global metrics and for 
    visualization.

    Examples
    --------
    >>> import networkx as nx
    >>> from bnm import BNMetrics
    >>> G1 = nx.DiGraph()
    >>> G1.add_edges_from([("A", "B"), ("C", "B")])
    >>> G2 = nx.DiGraph()
    >>> G2.add_edges_from([("A", "B"), ("B", "C")])
    >>> bnm = BNMetrics(G1, G2)

    >>> import numpy as np
    >>> mat1 = np.array([[0, 1], [0, 0]])
    >>> mat2 = np.array([[0, 0], [1, 0]])
    >>> bnm = BNMetrics(mat1, mat2, node_names=["X1", "X2"])
    """

    def __init__(self, G1, G2=None, node_names=None, mb_nodes='All'):
        self.G1_raw = G1
        self.G2_raw = G2
        self.mb_nodes = mb_nodes
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
            self.graph_dict = self._build_graph_dict_two_graphs(self.G1, self.G2, self.mb_nodes)
        else:
            self.graph_dict = self._build_graph_dict_one_graph(self.G1, self.mb_nodes)

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

    def _build_graph_dict_two_graphs(self, G1, G2, mb_nodes):
        """
        Internal method to construct dictionary of MB subgraphs.
        """
        graph_dict = {}
        if mb_nodes == 'All':
            mb_nodes_to_analyze = ['All'] + list(G1.nodes())
        else:
            mb_nodes_to_analyze = ['All'] + mb_nodes

        for i in mb_nodes_to_analyze:
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
    
    def _build_graph_dict_one_graph(self, G1, mb_nodes):
        """
        Internal method to construct dictionary of MB subgraphs.
        """
        graph_dict = {}

        if mb_nodes == 'All':
            mb_nodes_to_analyze = ['All'] + list(G1.nodes())
        else:
            mb_nodes_to_analyze = ['All'] + mb_nodes

        for i in mb_nodes_to_analyze:
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
        Calculates and returns a DataFrame of descriptive and/or comparison metrics 
        for the Markov blanket of each node, as well as global metrics under the label 'All'. 
        
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

        | node_name | n_edges_base | n_edges | n_colliders_base | n_colliders | shd | tp | fp |
        |-----------|--------------|---------|------------------|-------------|-----|----|----|
        | All       | 2.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
        | A         | 1.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
        | B         | 2.0          | 2.0     | 0.0              | 1.0         | 1   | 1  | 1  |
        | C         | 1.0          | 2.0     | 0.0              | 1.0         | 2   | 0  | 2  |
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
        where matching edges (true positives) are marked with color='crimson'.

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
                    G1_marked[u][v]["color"] = "crimson"
                    G2_marked[u][v]["color"] = "crimson"

            elif type1 == "undirected":
                if (G2.has_edge(u, v) and G2[u][v].get("type") == "undirected"):
                    G1_marked[u][v]["color"] = "crimson"
                    G2_marked[u][v]["color"] = "crimson"
                elif (G2.has_edge(v, u) and G2[v][u].get("type") == "undirected"):
                    G1_marked[u][v]["color"] = "crimson"
                    G2_marked[v][u]["color"] = "crimson"

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
                edge_color = "crimson" if attrs.get("color") == "crimson" else None

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
        - True positive edges (present in both graphs) in red.
        - Selected nodes in green.
        - Edge types (directed or undirected) are preserved visually.

        Parameters
        ----------
        nodes : list of str
            List of node names to include in the visualization. These must match node names in the DAGs. 
            The subgraph containing these nodes and their Markov blankets will be extracted and visualized.

        option : int, default=1
            Controls the visualization strategy:
            - 1: Shows the Markov blankets from G1 and G2 side-by-side.  
            - 2: Shows the Merkov blanket from G1 and the same set of nodes in G2.

        name1 : str, default="DAG1"
            Title to display above the first graph (G1).

        name2 : str, default="DAG2"
            Title to display above the second graph (G2).

        Returns
        -------
        None
            Displays two graph visualizations side-by-side using Graphviz within a Jupyter notebook environment.

        Raises
        ------
        ValueError
            If no second graph (G2) was provided during initialization.

        Example
        -------
        >>> from bnm import BNMetrics
        >>> import networkx as nx
        >>> G1 = nx.DiGraph()
        >>> G1.add_edges_from([("A", "B"), ("B", "C")])
        >>> G2 = nx.DiGraph()
        >>> G2.add_edges_from([("A", "B"), ("C", "B")])
        >>> bn = BNMetrics(G1, G2)
        >>> bn.compare_two_bn(nodes=['A', 'B'], option=1, name1='Original', name2='Modified')
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
            Displays a DAG using Graphviz within a Jupyter notebook environment.

        Raises
        ------
        ValueError
            If `layer` is 'd2' or 'd3' but no second graph (G2) was provided during initialization.

        Example
        -------
        >>> from bnm import BNMetrics
        >>> import networkx as nx
        >>> G1 = nx.DiGraph()
        >>> G1.add_edges_from([("A", "B"), ("B", "C")])
        >>> bn = BNMetrics(G1)
        >>> bn.plot_bn(nodes=['A', 'B'], layer='d1', title='Markov Blanket')
        """
            
        if self.G2 is None and layer in ['d2', 'd3']:
            print('please specify the second graph when instantiating the BNMetrics object')
        else:
            G = self._merge_graphs_no_duplicates_clean(nodes, layer)
            G = self._color_selected_nodes_green_single(G, nodes)
            self._display_graph_with_node_highlight(G, name=title)

