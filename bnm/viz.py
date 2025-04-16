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
    Compare all pairs of models and plot heatmaps of a specific comparison metric.

    Parameters
    ----------
    list_of_dags : list
        List of learned DAGs to be compared.
    model_names : list
        List of model names corresponding to each DAG.
    node_names : list
        List of all node names in the DAGs.
    metric : str, optional
        The comparison metric to compute (default is 'hd').
    mb_nodes : list, optional
        List of nodes for Markov blanket comparison. If None, defaults to list_of_nodes.
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
