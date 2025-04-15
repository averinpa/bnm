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
    Compare multiple DAG models on descriptive metrics and plot the results.

    Parameters
    ----------
    list_of_dags : list
        A list of DAGs (as networkx.DiGraph) to compare.

    model_names : list
        A list of model names corresponding to list_of_dags (e.g., alpha values).

    node_names : list
        List of all node names in the DAGs.

    mb_nodes : list
        List of nodes to compute Markov blanket-based descriptive metrics for.
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
