import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from torchtyping import TensorType as TT
from typing import Dict
import numpy as np
import pandas as pd
from brackets_datasets import BracketsDataset
from transformer_lens import utils, HookedTransformer
import einops
import torch as t

color_discrete_map = dict(zip(['both failures', 'just neg failure', 'balanced', 'just total elevation failure'], px.colors.qualitative.D3))
# names = ["balanced", "just total elevation failure", "just neg failure", "both failures"]
# colors = ['#2CA02C', '#1c96eb', '#b300ff', '#ff4800']
# color_discrete_map = dict(zip(names, colors))

def plot_failure_types_scatter(
    unbalanced_component_1: TT["batch"],
    unbalanced_component_2: TT["batch"],
    failure_types_dict: Dict[str, TT["batch"]],
    data: BracketsDataset
):
    failure_types = np.full(len(unbalanced_component_1), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": utils.to_numpy(unbalanced_component_1),
        "Head 2.1 contribution": utils.to_numpy(unbalanced_component_2),
        "Failure type": utils.to_numpy(failure_types),
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, color_discrete_map=color_discrete_map,
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        # category_orders={"color": failure_types_dict.keys()},
    ).update_traces(marker_size=4)
    fig.show()

def plot_contribution_vs_open_proportion(unbalanced_component: TT["batch"], title: str, failure_types_dict: Dict, data: BracketsDataset):
    failure_types = np.full(len(unbalanced_component), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(mask, name, failure_types)
    fig = px.scatter(
        x=utils.to_numpy(data.open_proportion), y=utils.to_numpy(unbalanced_component), color=failure_types, color_discrete_map=color_discrete_map,
        title=f"Head {title} contribution vs proportion of open brackets '('", template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": f"Head {title} contribution"}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()

def mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir: TT["comp", "batch"], 
    data: BracketsDataset, failure_types_dict: Dict
) -> None:
    failure_types = np.full(out_by_component_in_pre_20_unbalanced_dir.shape[-1], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(utils.to_numpy(mask), name, failure_types)
    for layer in range(2):
        mlp_output = out_by_component_in_pre_20_unbalanced_dir[3+layer*3]
        fig = px.scatter(
            x=utils.to_numpy(data.open_proportion[data.starts_open]), 
            y=utils.to_numpy(mlp_output[data.starts_open]), 
            color_discrete_map=color_discrete_map,
            color=utils.to_numpy(failure_types)[utils.to_numpy(data.starts_open)], 
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

def plot_neurons(neurons_in_unbalanced_dir: TT["batch", "neurons"], model: HookedTransformer, data: BracketsDataset, failure_types_dict: Dict, layer: int):
    
    failure_types = np.full(neurons_in_unbalanced_dir.shape[0], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(utils.to_numpy(mask[utils.to_numpy(data.starts_open)]), name, failure_types)

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = einops.repeat(t.arange(model.cfg.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = einops.repeat(failure_types, "s -> (s n)", n=model.cfg.d_model)
    data_open_proportion = einops.repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.cfg.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": utils.to_numpy(neurons_in_unbalanced_dir.flatten()),
        "Neuron number": utils.to_numpy(neuron_numbers),
        "Open-proportion": utils.to_numpy(data_open_proportion),
        "Failure type": failure_types
    })
    fig = px.scatter(
        df, 
        x="Open-proportion", y="Output in 2.0 direction", color="Failure type", animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", 
        template="simple_white", height=800, width=1100
    ).update_traces(marker_size=3).update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5])
    fig.show(renderer="browser")

def plot_attn_pattern(pattern: TT["batch", "head_idx", "seqQ", "seqK"]):
    fig = px.imshow(
        pattern, 
        title="Estimate for avg attn probabilities when query is from '('",
        labels={"x": "Key tokens (avg of left & right parens)", "y": "Query tokens (all left parens)"},
        height=900, width=900,
        color_continuous_scale="RdBu_r", range_color=[0, pattern.max().item()]
    ).update_layout(
        xaxis = dict(
            tickmode = "array", ticktext = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals = list(range(42)), tickangle = 0,
        ),
        yaxis = dict(
            tickmode = "array", ticktext = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"],
            tickvals = list(range(42)), 
        ),
    )
    fig.show()
    

def hists_per_comp(out_by_component_in_unbalanced_dir: TT["component", "batch"], data: BracketsDataset, xaxis_range=(-1, 1)):
    '''
    Plots the contributions in the unbalanced direction, as supplied by the `out_by_component_in_unbalanced_dir` tensor.
    '''
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0", (2, 2): "head 0.1", (2, 3): "mlp 0",
        (3, 1): "head 1.0", (3, 2): "head 1.1", (3, 3): "mlp 1",
        (4, 1): "head 2.0", (4, 2): "head 2.1", (4, 3): "mlp 2"
    }
    n_layers = out_by_component_in_unbalanced_dir.shape[0] // 3
    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), in_dir in zip(titles.items(), out_by_component_in_unbalanced_dir):
        fig.add_trace(go.Histogram(x=utils.to_numpy(in_dir[data.isbal]), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=utils.to_numpy(in_dir[~data.isbal]), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()