import networkx as nx
import numpy as np

# ==========================
# Descriptive Graph Metrics
# ==========================

def count_edges(graph):
    """
    Count the total number of edges in the graph (both directed and undirected).

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A NetworkX directed graph with optional 'type' attributes on edges.

    Returns:
    int: Total number of edges.
    """
    return graph.number_of_edges()

def count_colliders(graph):
    """
    Count the number of collider structures in a directed graph.

    A collider is a node with two or more parents (connected via directed edges)
    where the parents are not connected to each other via a directed edge 
    (in either direction). Undirected edges are ignored using the 'type' attribute.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where each edge may have a 'type' attribute
                        ("directed" or "undirected").

    Returns:
    int: The number of collider nodes found in the graph.
    """
    colliders = 0

    for node in graph.nodes():
        parents = [
            u for u in graph.predecessors(node)
            if graph.has_edge(u, node) and graph[u][node].get("type") == "directed"
        ]

        if len(parents) < 2:
            continue

        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                u, v = parents[i], parents[j]

                connected_uv = graph.has_edge(u, v) and graph[u][v].get("type") == "directed"
                connected_vu = graph.has_edge(v, u) and graph[v][u].get("type") == "directed"

                if not connected_uv and not connected_vu:
                    colliders += 1

    return colliders

def count_root_nodes(graph):
    """
    Count the number of root nodes in the graph, considering only directed edges.

    A root node is defined as:
    - Having no incoming directed edges
    - Not being involved in any undirected edges (incoming or outgoing)

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute
                        ("directed" or "undirected").

    Returns:
    int: The number of strict root nodes in the graph.
    """
    count = 0

    for node in graph.nodes():
        has_directed_parent = any(
            graph[u][node].get("type", "directed") == "directed"
            for u in graph.predecessors(node)
        )

        has_undirected_connection = any(
            graph[u][node].get("type") == "undirected"
            for u in graph.predecessors(node)
            if graph.has_edge(u, node)
        ) or any(
            graph[node][v].get("type") == "undirected"
            for v in graph.successors(node)
            if graph.has_edge(node, v)
        )

        if not has_directed_parent and not has_undirected_connection:
            count += 1

    return count

def count_leaf_nodes(graph):
    """
    Count the number of leaf nodes in the graph, considering only directed edges.

    A leaf node is defined as:
    - Having no outgoing directed edges
    - Not being involved in any undirected edges (incoming or outgoing)

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute
                        ("directed" or "undirected").

    Returns:
    int: The number of strict leaf nodes in the graph.
    """
    count = 0

    for node in graph.nodes():
        has_directed_child = any(
            graph[node][v].get("type", "directed") == "directed"
            for v in graph.successors(node)
        )

        has_undirected_connection = any(
            graph[u][node].get("type") == "undirected"
            for u in graph.predecessors(node)
            if graph.has_edge(u, node)
        ) or any(
            graph[node][v].get("type") == "undirected"
            for v in graph.successors(node)
            if graph.has_edge(node, v)
        )

        if not has_directed_child and not has_undirected_connection:
            count += 1

    return count

def count_isolated_nodes(graph):
    """
    Count the number of isolated nodes in the graph, using only directed edges.

    An isolated node is defined as:
    - Having no incoming or outgoing directed edges
    - Not being involved in any undirected edges (incoming or outgoing)

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute
                        ("directed" or "undirected").

    Returns:
    int: The number of strict isolated nodes in the graph.
    """
    count = 0

    for node in graph.nodes():
        has_directed_in = any(
            graph[u][node].get("type", "directed") == "directed"
            for u in graph.predecessors(node)
        )
        has_directed_out = any(
            graph[node][v].get("type", "directed") == "directed"
            for v in graph.successors(node)
        )
        has_undirected_connection = any(
            graph[u][node].get("type") == "undirected"
            for u in graph.predecessors(node)
            if graph.has_edge(u, node)
        ) or any(
            graph[node][v].get("type") == "undirected"
            for v in graph.successors(node)
            if graph.has_edge(node, v)
        )

        if not has_directed_in and not has_directed_out and not has_undirected_connection:
            count += 1

    return count

def count_nodes(graph):
    """
    Count the total number of nodes in the graph.

    Parameters:
    graph (nx.DiGraph): A directed graph.

    Returns:
    int: The number of nodes in the graph.
    """
    return graph.number_of_nodes()

def count_directed_arcs(graph):
    """
    Count the number of directed arcs in the graph.

    Only edges where the 'type' attribute is 'directed' are included.
    If the 'type' attribute is missing, the edge is assumed to be directed by default.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute.

    Returns:
    int: The number of directed arcs.
    """
    return sum(
        1 for u, v in graph.edges()
        if graph[u][v].get("type", "directed") == "directed"
    )

def count_undirected_arcs(graph):
    """
    Count the number of undirected arcs in the graph.

    Only edges where the 'type' attribute is 'undirected' are included.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where some edges may be marked as 'undirected'.

    Returns:
    int: The number of undirected arcs.
    """
    return sum(
        1 for u, v in graph.edges()
        if graph[u][v].get("type") == "undirected"
    )

def count_reversible_arcs(graph):
    """
    Count the number of reversible arcs in the graph.

    A reversible arc is defined as:
    - A directed edge (type == 'directed')
    - That is not part of a collider structure (A → C ← B)

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute
                        ("directed" or "undirected").

    Returns:
    int: The number of reversible (non-collider) directed arcs.
    """
    collider_centers = set()
    for node in graph.nodes():
        parents = [
            u for u in graph.predecessors(node)
            if graph[u][node].get("type", "directed") == "directed"
        ]
        if len(parents) < 2:
            continue

        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                u, v = parents[i], parents[j]
                connected_uv = graph.has_edge(u, v) and graph[u][v].get("type") == "directed"
                connected_vu = graph.has_edge(v, u) and graph[v][u].get("type") == "directed"

                if not connected_uv and not connected_vu:
                    collider_centers.add(node)

    reversible_count = 0
    for u, v in graph.edges():
        if graph[u][v].get("type", "directed") == "directed" and v not in collider_centers:
            reversible_count += 1

    return reversible_count

def count_in_degree(graph, node):
    """
    Count the number of directed incoming edges to a specified node.

    If the input node is 'All', returns np.nan.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute.
    node (str): The node for which to count incoming directed edges.

    Returns:
    int or float: The number of directed incoming edges, or np.nan if node == 'All'.
    """
    if node == "All":
        return np.nan

    return sum(
        1 for u in graph.predecessors(node)
        if graph[u][node].get("type", "directed") == "directed"
    )

def count_out_degree(graph, node):
    """
    Count the number of directed outgoing edges from a specified node.

    If the input node is 'All', returns np.nan.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    graph (nx.DiGraph): A directed graph where edges may have a 'type' attribute.
    node (str): The node for which to count outgoing directed edges.

    Returns:
    int or float: The number of directed outgoing edges, or np.nan if node == 'All'.
    """
    if node == "All":
        return np.nan

    return sum(
        1 for v in graph.successors(node)
        if graph[node][v].get("type", "directed") == "directed"
    )

# ==========================
# Comparative Graph Metrics
# ==========================

def count_additions(G1, G2):
    """
    Count the number of undirected (presence-only) edges added in G2 relative to G1.

    The function treats all edges as undirected — direction is ignored.
    Only the presence of a connection between two nodes matters.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph.
    G2 (nx.DiGraph): Comparison graph.

    Returns:
    int: Number of edges that exist in G2 but not in G1 (as undirected).
    """
    g1_edges = {frozenset((u, v)) for u, v in G1.edges()}
    g2_edges = {frozenset((u, v)) for u, v in G2.edges()}

    added = g2_edges - g1_edges
    return len(added)

def count_deletions(G1, G2):
    """
    Count the number of undirected (presence-only) edges that were deleted in G2 relative to G1.

    The function treats all edges as undirected — direction is ignored.
    Only the presence of a connection between two nodes matters.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph.
    G2 (nx.DiGraph): Comparison graph.

    Returns:
    int: Number of edges that exist in G1 but not in G2 (as undirected).
    """
    g1_edges = {frozenset((u, v)) for u, v in G1.edges()}
    g2_edges = {frozenset((u, v)) for u, v in G2.edges()}

    deleted = g1_edges - g2_edges
    return len(deleted)

def count_reversals(G1, G2):
    """
    Count the number of edge reversals between two graphs.

    A reversal is defined as:
    - A directed edge in G1 becomes directed in the opposite direction in G2
    - A directed edge in G1 becomes undirected in G2
    - An undirected edge in G1 becomes directed (in either direction) in G2

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Number of reversed edges.
    """
    reversals = 0
    checked_pairs = set()

    for u, v in G1.edges():
        edge_g1 = frozenset((u, v))
        if edge_g1 in checked_pairs:
            continue
        checked_pairs.add(edge_g1)

        g1_type = G1[u][v].get("type", "directed")

        if g1_type == "undirected":
            # Undirected → Directed (in either direction)
            if (G2.has_edge(u, v) and G2[u][v].get("type") == "directed") or \
               (G2.has_edge(v, u) and G2[v][u].get("type") == "directed"):
                reversals += 1

        elif g1_type == "directed":
            # Directed → reversed Directed
            if G2.has_edge(v, u) and G2[v][u].get("type") == "directed":
                reversals += 1
            # Directed → Undirected
            elif G2.has_edge(u, v) and G2[u][v].get("type") == "undirected":
                reversals += 1

    return reversals

def shd(G1, G2):
    """
    Compute the Structural Hamming Distance (SHD) between two DAGs.

    SHD is the number of edge differences between two graphs.
    This implementation uses:
    - Undirected edge comparison for additions and deletions
    - Full reversal detection (flipped directed edges and type changes)

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Structural Hamming Distance between G1 and G2.
    """
    added = count_additions(G1, G2)
    deleted = count_deletions(G1, G2)
    reversed_ = count_reversals(G1, G2)

    return added + deleted + reversed_

def hd(G1, G2):
    """
    Compute Hamming Distance (HD) between two DAGs, ignoring edge direction.

    HD is computed as the number of:
    - Edges present in G2 but not in G1 (additions)
    - Edges present in G1 but not in G2 (deletions)

    Reversals and edge type changes are ignored.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Hamming Distance between G1 and G2.
    """
    added = count_additions(G1, G2)
    deleted = count_deletions(G1, G2)

    return added + deleted

def count_true_positives(G1, G2):
    """
    Count the number of true positive edges between G1 and G2.

    Matching rules:
    - Directed edges must exist in both graphs with the same direction and type='directed'.
    - Undirected edges must exist in either direction in both graphs with type='undirected'.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Number of matching edges (true positives) under hybrid evaluation.
    """
    tp = 0

    for u, v in G1.edges():
        type1 = G1[u][v].get("type", "directed")

        if type1 == "directed":
            if G2.has_edge(u, v) and G2[u][v].get("type") == "directed":
                tp += 1

        elif type1 == "undirected":
            if ((G2.has_edge(u, v) and G2[u][v].get("type") == "undirected") or
                (G2.has_edge(v, u) and G2[v][u].get("type") == "undirected")):
                tp += 1

    return tp

def count_false_positives(G1, G2):
    """
    Count the number of false positive edges: present in G2 but not in G1.

    Matching rules:
    - Directed: counted as false positive if G1 does not contain the same directed edge with type='directed'.
    - Undirected: counted as false positive if G1 does not contain the same undirected edge (in either direction).

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Number of false positive edges under hybrid evaluation.
    """
    fp = 0

    for u, v in G2.edges():
        type2 = G2[u][v].get("type", "directed")

        if type2 == "directed":
            if not (G1.has_edge(u, v) and G1[u][v].get("type") == "directed"):
                fp += 1

        elif type2 == "undirected":
            if not (
                (G1.has_edge(u, v) and G1[u][v].get("type") == "undirected") or
                (G1.has_edge(v, u) and G1[v][u].get("type") == "undirected")
            ):
                fp += 1

    return fp

def count_false_negatives(G1, G2):
    """
    Count the number of false negative edges: present in G1 but missing in G2.

    Matching rules:
    - Directed: counted as false negative if G2 does not contain the same directed edge with type='directed'.
    - Undirected: counted as false negative if G2 does not contain the same undirected edge (in either direction).

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph with 'type' attributes on edges.
    G2 (nx.DiGraph): Comparison graph with 'type' attributes on edges.

    Returns:
    int: Number of false negative edges under hybrid evaluation.
    """
    fn = 0

    for u, v in G1.edges():
        type1 = G1[u][v].get("type", "directed")

        if type1 == "directed":
            if not (G2.has_edge(u, v) and G2[u][v].get("type") == "directed"):
                fn += 1

        elif type1 == "undirected":
            if not (
                (G2.has_edge(u, v) and G2[u][v].get("type") == "undirected") or
                (G2.has_edge(v, u) and G2[v][u].get("type") == "undirected")
            ):
                fn += 1

    return fn

def precision(G1, G2):
    """
    Compute precision based on hybrid edge comparison.

    Precision is defined as:
        TP / (TP + FP)

    If TP + FP == 0, returns 0.0 to avoid division by zero.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph (ground truth).
    G2 (nx.DiGraph): Predicted graph.

    Returns:
    float: Precision score.
    """
    tp = count_true_positives(G1, G2)
    fp = count_false_positives(G1, G2)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)

def recall(G1, G2):
    """
    Compute recall based on hybrid edge comparison.

    Recall is defined as:
        TP / (TP + FN)

    If TP + FN == 0, returns 0.0 to avoid division by zero.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph (ground truth).
    G2 (nx.DiGraph): Predicted graph.

    Returns:
    float: Recall score.
    """
    tp = count_true_positives(G1, G2)
    fn = count_false_negatives(G1, G2)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)

def f1_score(G1, G2):
    """
    Compute F1 score based on hybrid edge comparison.

    F1 score is the harmonic mean of precision and recall:
        F1 = 2 * (P * R) / (P + R)

    If both precision and recall are zero, returns 0.0.

    Note:
    This function assumes that the graph(s) have been preprocessed using 
    `mark_and_collapse_bidirected_edges()`, so that each edge has a valid 
    'type' attribute ('directed' or 'undirected').

    Parameters:
    G1 (nx.DiGraph): Base graph (ground truth).
    G2 (nx.DiGraph): Predicted graph.

    Returns:
    float: F1 score.
    """
    p = precision(G1, G2)
    r = recall(G1, G2)

    if p + r == 0:
        return 0.0

    return 2 * (p * r) / (p + r)
