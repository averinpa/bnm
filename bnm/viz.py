import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from . import BNMetrics



def plot_descriptive(df):
    metrics = [
        "n_edges", "n_colliders", "n_root_nodes", "n_leaf_nodes",
        "n_isolated_nodes", "n_directed_arcs", "n_undirected_arcs", "n_reversible_arcs"
    ]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=metrics,
        horizontal_spacing=0.08,
        vertical_spacing=0.12
    )

    node_names = df['node_name'].unique()
    traces_dict = {}

    for node in node_names:
        node_df = df[df['node_name'] == node]
        traces = []
        for metric in metrics:
            trace = go.Scatter(
                x=node_df['model_name'],
                y=node_df[metric],
                mode='lines+markers',
                name=node,
                marker=dict(color='#1E3A8A',
                           symbol='diamond'),
                visible=(node == "All"),
                showlegend=False
            )
            traces.append(trace)
        traces_dict[node] = traces

    for i, metric in enumerate(metrics):
        row = i // 3 + 1
        col = i % 3 + 1
        for node in node_names:
            fig.add_trace(traces_dict[node][i], row=row, col=col)

    dropdown_buttons = []
    for node in node_names:
        visibility = []
        for _ in range(len(metrics)):
            for name in node_names:
                visibility.append(name == node)
        dropdown_buttons.append(
            dict(
                label=node,
                method="update",
                args=[{"visible": visibility},
                      {"title": f"Descriptive Metrics — Node: {node}"}]
            )
        )

    fig.update_layout(
        height=750,
        width=1200,
        title="Descriptive Metrics",
        margin=dict(l=20, r=20, t=80, b=40),
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            direction="down",
            x=1.01,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )]
    )
    fig.update_xaxes(tickangle=45)
    fig.show()


def compare_models_descriptive(list_of_dags, model_names, node_names, mb_nodes):
    """
    Calculates descriptive metrics—including number of edges, colliders, root nodes, 
    leaf nodes, isolated nodes, directed arcs, undirected arcs, and reversible arcs—for the 
    global structure and specified Markov blankets. The results are displayed in eight 
    interactive subplots, with a dropdown menu allowing selection of the Markov blanket of interest.
    
    Parameters
    ----------
    list_of_dags : list
        A list of DAGs (as networkx.DiGraph) to compare.

    model_names : list
        A list of model names corresponding to list_of_dags.

    node_names : list
        List of all node names in the DAGs.

    mb_nodes : list
        List of nodes to compute Markov blanket-based descriptive metrics for.
    
    Returns
        -------
        None
            Displays the descriptive metrics in eight interactive subplots

        
    Example
    -------
    >>> from bnm import compare_models_descriptive
    >>> import networkx as nx
    >>> G1 = nx.DiGraph()
    >>> G1.add_edges_from([("A", "B"), ("B", "C")])
    >>> G2 = nx.DiGraph()
    >>> G2.add_edges_from([("A", "B"), ("A", "C")])
    >>> compare_models_descriptive(list_of_dags=[G1, G2], 
        ...                model_names=['Model1', 'Model2'], 
        ...                node_names=list(G1.nodes), 
        ...                mb_nodes=['A', 'B'])
    
    """
    list_of_df = []
    for i in range(len(list_of_dags)):
        bnm_obj = BNMetrics(G1=list_of_dags[i], node_names=node_names, mb_nodes=mb_nodes)
        bnm_df = bnm_obj.compare_df(descriptive_metrics='All', comparison_metrics=None)
        del bnm_obj
        bnm_df['model_name'] = f"{model_names[i]:.4f}" if isinstance(model_names[i], float) else str(model_names[i])
        list_of_df.append(bnm_df)

    df = pd.concat(list_of_df)
    plot_descriptive(df=df)


def plot_heatmap(df, metric):
    """
    Plot an interactive heatmap comparing models based on a selected metric for each node.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns: 'node_name', 'model_name1', 'model_name2', and the metric.
    metric : str
        Name of the metric column to be visualized (e.g., 'hd', 'shd').
    """
    heatmaps = []
    node_names = df['node_name'].unique()
    visibility_list = []

    for i, node in enumerate(node_names):
        df_node = df[df['node_name'] == node]
        pivot = df_node.pivot(index='model_name2', columns='model_name1', values=metric)

        heatmap = go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='Blues',
            colorbar=dict(title=metric),
            visible=(i == 0),
            text=[[f"{v:.2f}" if pd.notna(v) else "" for v in row] for row in pivot.values],
            texttemplate="%{text}",
            textfont={"size": 8},
            zmin=df_node[metric].min(),
            zmax=df_node[metric].max(),
        )
        heatmaps.append(heatmap)
        visibility = [j == i for j in range(len(node_names))]
        visibility_list.append(visibility)

    fig = go.Figure(data=heatmaps)

    buttons = [
        dict(
            label=node,
            method="update",
            args=[
                {"visible": visibility_list[i]},
                {"title": f"{metric} Heatmap — Node: {node}"}
            ]
        )
        for i, node in enumerate(node_names)
    ]

    fig.update_layout(
        title=f"{metric} Heatmap — Node: {node_names[0]}",
        xaxis_title="Model Name",
        yaxis_title="Model Name",
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction="down",
            x=1.02,
            xanchor="left",
            y=1.1,
            yanchor="top"
        )],
        width=900,
        height=800
    )

    fig.show()


def compare_models_comparative(
    list_of_dags,
    model_names,
    node_names,
    metric,
    mb_nodes
    ):
    """
    Calculates a comparative metric—selected from additions, 
    deletions, reversals, SHD, HD, TP, FN, FP, precision, recall, or 
    F1 score—for the global structure and specified Markov blankets. 
    The results are visualized in a heatmap comparing all pairs of 
    models based on the chosen metric, with a dropdown menu for 
    selecting the Markov blanket of interest.
    
    Parameters
    ----------
    list_of_dags : list[nx.DiGraph], list[np.ndarray] or list[list of lists]
        List of DAGs to compare.
    model_names : list
        A list of model names corresponding to list_of_dags.
    node_names : list
        A list of all node names in the associated with a DAG.
    metric : str
        metric to be calculated. Choices are additions, deletions, 
        reversals, shd, hd, tp, fn, fp, precision, recall, or f1_score
    mb_nodes : list
        A list of nodes to compute Markov blanket-based comparative metric for.
    
    
    Returns
        -------
        None
            Displays the heatmap comparing all pairs of models

        
    Example
    -------
    >>> from bnm import compare_models_comparative
    >>> import networkx as nx
    >>> G1 = nx.DiGraph()
    >>> G1.add_edges_from([("A", "B"), ("B", "C")])
    >>> G2 = nx.DiGraph()
    >>> G2.add_edges_from([("A", "B"), ("A", "C")])
    >>> compare_models_comparative(list_of_dags=[G1, G2], 
        ...                        model_names=['Model1', 'Model2'], 
        ...                        node_names=list(G1.nodes), 
        ...                        metric='shd',
        ...                        mb_nodes=['A', 'B'])
    
    """
    all_comparisons = []

    for i, dag1 in enumerate(list_of_dags):
        for j, dag2 in enumerate(list_of_dags):
            bnm_obj = BNMetrics(G1=dag1, G2=dag2, node_names=node_names, mb_nodes=mb_nodes)
            comparison_df = bnm_obj.compare_df(
                descriptive_metrics=None,
                comparison_metrics=[metric]
            )
            comparison_df['model_name1'] = f"{model_names[i]:.4f}" if isinstance(model_names[i], float) else str(model_names[i])
            comparison_df['model_name2'] = f"{model_names[j]:.4f}" if isinstance(model_names[j], float) else str(model_names[j])
            all_comparisons.append(comparison_df)
            del bnm_obj

    combined_df = pd.concat(all_comparisons)
    plot_heatmap(combined_df, metric)


def plot_mb(df):
    """
    Create a grid of bar plots showing value counts for each column in the input DataFrame.
    Y-axis is labeled on the 4th subplot; X-axis on the 8th subplot.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame where each column represents a discrete metric to be visualized.
    """
    n_rows, n_cols = 3, 4
    numeric_cols = df.columns[:n_rows * n_cols]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=numeric_cols,
        vertical_spacing=0.1
    )

    for i, col in enumerate(numeric_cols):
        row = i // n_cols + 1
        col_idx = i % n_cols + 1

        counts = df[col].value_counts().sort_index()

        fig.add_trace(
            go.Bar(
                x=counts.index.astype(int).astype(str),
                y=counts.values,
                marker=dict(color='#1E3A8A'),
                name=col
            ),
            row=row,
            col=col_idx
        )

    fig.update_yaxes(title_text="Frequency", col=1)
    fig.update_xaxes(title_text="Metric Value", row=3)

    fig.update_layout(
        height=800,
        width=1200,
        title_text="Exploration of Markov Blanket Space",
        showlegend=False,
        margin=dict(t=100),
        font=dict(size=11)
    )

    fig.show()

def analyse_mb(G1, node_names=None, mb_nodes='All'):
    """
    Analyze the Markov blanket space of a DAG and plot distribution 
    of descriptive metrics.
    
    Parameters:
    -----------
    G1 : nx.DiGraph, np.ndarray, or list of lists
        The first DAG. If not a DiGraph, 
        it must be a square adjacency matrix.
    node_names : list of str, optional
        Required only when G1, G2 or both are given as a NumPy array or list of lists. 
        Length must match number of nodes.
    mb_nodes : str or list, default='All'
        Nodes for which Markov blanket-based metrics and subgraphs will be computed.
    
    Returns:
    --------
    None
        Displays the distibution of descriptive metrics in eight interactive subplots
    
    Example
    -------
    >>> from bnm import analyse_mb
    >>> import networkx as nx
    >>> G1 = nx.DiGraph()
    >>> G1.add_edges_from([("A", "B"), ("B", "C")])
    >>> analyse_mb(G1, node_names=None, mb_nodes='All')
    
    """
    bnm_obj = BNMetrics(G1=G1, node_names=node_names, mb_nodes=mb_nodes)
    
    df = bnm_obj.compare_df(descriptive_metrics='All', comparison_metrics=None)
    df = df.query("node_name != 'All'").drop(columns='node_name')
    
    plot_mb(df)