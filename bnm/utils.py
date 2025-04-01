import networkx as nx
import random

def mark_and_collapse_bidirected_edges(graph):
    """
    Modifies a DiGraph:
    - If A→B and B→A both exist, keep A→B (arbitrary), remove B→A, and mark A→B as 'undirected'
    - All other edges are marked as 'directed'

    Parameters:
    graph (nx.DiGraph): The input directed graph

    Returns:
    nx.DiGraph: A new DiGraph with 'type' attribute added to each edge
                ('directed' or 'undirected')
    """
    G = graph.copy()
    processed = set()

    for u, v in list(G.edges()):
        if (v, u) in G.edges() and ((v, u) not in processed and (u, v) not in processed):
            if G.has_edge(v, u):
                G.remove_edge(v, u)
            if G.has_edge(u, v):
                G[u][v]['type'] = 'undirected'
            processed.add((u, v))
        elif (u, v) not in processed:
            if G.has_edge(u, v):
                G[u][v]['type'] = 'directed'
            processed.add((u, v))

    return G

def get_markov_blanket_subgraph(graph, node):
    """
    Return the subgraph induced by the Markov Blanket of the given node.

    The Markov Blanket includes:
    - Directed parents
    - Directed children
    - Parents of directed children
    - Nodes connected via undirected edges
    - The node itself

    Parameters:
    graph (nx.DiGraph): The input graph
    node (str): Node of interest

    Returns:
    nx.DiGraph: Induced subgraph of the Markov Blanket
    """
    blanket = set()

    # 1. Parents of node (via directed edges)
    parents = {
        u for u in graph.predecessors(node)
        if graph[u][node].get("type", "directed") == "directed"
    }
    blanket.update(parents)

    # 2. Children of node (via directed edges)
    children = {
        v for v in graph.successors(node)
        if graph[node][v].get("type", "directed") == "directed"
    }
    blanket.update(children)

    # 3. Other parents of the node's children
    for child in children:
        co_parents = {
            u for u in graph.predecessors(child)
            if graph[u][child].get("type", "directed") == "directed" and u != node
        }
        blanket.update(co_parents)

    # 4. Nodes connected via undirected edges (in either direction)
    undirected_neighbors = {
        u for u in graph.predecessors(node)
        if graph[u][node].get("type") == "undirected"
    }.union({
        v for v in graph.successors(node)
        if graph[node][v].get("type") == "undirected"
    })
    blanket.update(undirected_neighbors)

    # Include the node itself
    blanket.add(node)

    return graph.subgraph(blanket).copy()

def generate_random_dag(n_nodes=40, edge_prob=0.1, seed=None):
    """
    Generate a random Directed Acyclic Graph (DAG) with randomized structure and node labels.

    This function ensures acyclicity by constructing edges in a random topological order.
    To avoid visual patterns (e.g., upper-triangular adjacency matrix), it also applies a final 
    random relabeling of nodes to simulate realistic DAGs.

    Parameters
    ----------
    n_nodes : int, default=40
        Number of nodes in the graph.

    edge_prob : float, default=0.1
        Probability of including a valid acyclic edge between nodes in topological order.

    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    dag : networkx.DiGraph
        A randomly generated DAG with relabeled nodes ('X_1', ..., 'X_n').

    Example
    -------
    >>> G = generate_random_dag(n_nodes=20, edge_prob=0.15, seed=42)
    >>> nx.is_directed_acyclic_graph(G)
    True
    >>> list(G.nodes)[:5]
    ['X_14', 'X_3', 'X_17', 'X_8', 'X_10']
    """
    if seed:
        random.seed(seed)

    nodes = list(range(n_nodes))
    topo_order = list(nodes)
    random.shuffle(topo_order)

    possible_edges = [(topo_order[i], topo_order[j]) for i in range(n_nodes) for j in range(i + 1, n_nodes)]

    n_edges = int(len(possible_edges) * edge_prob)
    sampled_edges = random.sample(possible_edges, n_edges)

    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)
    dag.add_edges_from(sampled_edges)

    final_order = list(nodes)
    random.shuffle(final_order)
    rename_map = {node: f"X_{i+1}" for i, node in enumerate(final_order)}
    dag = nx.relabel_nodes(dag, rename_map)

    return dag

def dag_to_cpdag(dag):
    """
    Convert a DAG into a CPDAG-like structure:
    - Retain only directed edges that form colliders: A → C ← B
    - Convert all other directed edges to bidirected (A → B and B → A)

    Returns:
    - A new DiGraph with:
        - Directed collider edges kept
        - Non-collider edges replaced with bidirectional edges
    """
    cpdag = nx.DiGraph()
    cpdag.add_nodes_from(dag.nodes())

    collider_edges = set()
    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        if len(parents) >= 2:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    a, b = parents[i], parents[j]
                    if not dag.has_edge(a, b) and not dag.has_edge(b, a):
                        collider_edges.add((a, node))
                        collider_edges.add((b, node))

    # Add edges to CPDAG
    for u, v in dag.edges():
        if (u, v) in collider_edges:
            cpdag.add_edge(u, v)  # Keep directed
        else:
            cpdag.add_edge(u, v)
            cpdag.add_edge(v, u)  # Make it bidirected

    return cpdag
