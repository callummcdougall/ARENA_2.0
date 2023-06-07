import torch as t
from torch import Tensor
from typing import List, Union, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from typing import Dict
import pandas as pd
from jaxtyping import Float
import einops


# GENERIC PLOTTING FUNCTIONS

update_layout_set = {"xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat", "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor", "showlegend", "xaxis_tickmode", "yaxis_tickmode", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap", "coloraxis_showscale"}

def imshow(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "color_continuous_midpoint" not in kwargs_pre:
        kwargs_pre["color_continuous_midpoint"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post)
    if facet_labels:
        # Weird thing where facet col wrap means labels are in wrong order
        if "facet_col_wrap" in kwargs_pre:
            facet_labels = reorder_list_in_plotly_way(facet_labels, kwargs_pre["facet_col_wrap"])
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.show(renderer=renderer)


def reorder_list_in_plotly_way(L: list, col_wrap: int):
    '''
    Helper function, because Plotly orders figures in an annoying way when there's column wrap.
    '''
    L_new = []
    while len(L) > 0:
        L_new.extend(L[-col_wrap:])
        L = L[:-col_wrap]

    return L_new


def line(y: Union[t.Tensor, List[t.Tensor]], renderer=None, **kwargs):
    '''
    Edit to this helper function, allowing it to take args in update_layout (e.g. yaxis_range).
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    if "xaxis_tickvals" in kwargs_pre:
        tickvals = kwargs_pre.pop("xaxis_tickvals")
        kwargs_post["xaxis"] = dict(
            tickmode = "array",
            tickvals = kwargs_pre.get("x", np.arange(len(tickvals))),
            ticktext = tickvals
        )
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "use_secondary_yaxis" in kwargs_pre and kwargs_pre["use_secondary_yaxis"]:
        del kwargs_pre["use_secondary_yaxis"]
        if "labels" in kwargs_pre:
            labels: dict = kwargs_pre.pop("labels")
            kwargs_post["yaxis_title_text"] = labels.get("y1", None)
            kwargs_post["yaxis2_title_text"] = labels.get("y2", None)
            kwargs_post["xaxis_title_text"] = labels.get("x", None)
        for k in ["title", "template", "width", "height"]:
            if k in kwargs_pre:
                kwargs_post[k] = kwargs_pre.pop(k)
        fig = make_subplots(specs=[[{"secondary_y": True}]]).update_layout(**kwargs_post)
        y0 = to_numpy(y[0])
        y1 = to_numpy(y[1])
        x0, x1 = kwargs_pre.pop("x", [np.arange(len(y0)), np.arange(len(y1))])
        name0, name1 = kwargs_pre.pop("names", ["yaxis1", "yaxis2"])
        fig.add_trace(go.Scatter(y=y0, x=x0, name=name0), secondary_y=False)
        fig.add_trace(go.Scatter(y=y1, x=x1, name=name1), secondary_y=True)
        fig.show(renderer)
    else:
        y = list(map(to_numpy, y)) if isinstance(y, list) and not (isinstance(y[0], int) or isinstance(y[0], float)) else to_numpy(y)
        fig = px.line(y=y, **kwargs_pre).update_layout(**kwargs_post)
        if names is not None:
            fig.for_each_trace(lambda trace: trace.update(name=names.pop(0)))
        fig.show(renderer)
        

def scatter(x, y, renderer=None, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    add_line = None
    if "add_line" in kwargs:
        add_line = kwargs.pop("add_line")
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "facet_labels" in kwargs_pre:
        facet_labels = kwargs_pre.pop("facet_labels")
    else:
        facet_labels = None
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.scatter(y=y, x=x, **kwargs_pre).update_layout(**kwargs_post)
    if add_line is not None:
        xrange = fig.layout.xaxis.range or [x.min(), x.max()]
        yrange = fig.layout.yaxis.range or [y.min(), y.max()]
        add_line = add_line.replace(" ", "")
        if add_line in ["x=y", "y=x"]:
            fig.add_trace(go.Scatter(mode='lines', x=xrange, y=xrange, showlegend=False))
        elif re.match("(x|y)=", add_line):
            try: c = float(add_line.split("=")[1])
            except: raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
            x, y = ([c, c], yrange) if add_line[0] == "x" else (xrange, [c, c])
            fig.add_trace(go.Scatter(mode='lines', x=x, y=y, showlegend=False))
        else:
            raise ValueError(f"Unrecognized add_line: {add_line}. Please use either 'x=y' or 'x=c' or 'y=c' for some float c.")
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    fig.show(renderer)

def bar(tensor, renderer=None, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "hovermode" not in kwargs_post:
        kwargs_post["hovermode"] = "x unified"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    px.bar(y=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post).show(renderer)

def hist(tensor, renderer=None, **kwargs):
    '''
    '''
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.1
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    px.histogram(x=to_numpy(tensor), **kwargs_pre).update_layout(**kwargs_post).show(renderer)






# PLOTTING FUNCTIONS FOR PART 2: INTRO TO MECH INTERP

def plot_comp_scores(model, comp_scores, title: str = "", baseline: Optional[t.Tensor] = None) -> go.Figure:
    return px.imshow(
        to_numpy(comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title=title,
        color_continuous_scale="RdBu" if baseline is not None else "Blues",
        color_continuous_midpoint=baseline if baseline is not None else None,
        zmin=None if baseline is not None else 0.0,
    )

def convert_tokens_to_string(model: HookedTransformer, tokens, batch_index=0):
    '''
    Helper function to convert tokens into a list of strings, for printing.
    '''
    if len(tokens.shape) == 2:
        tokens = tokens[batch_index]
    return [f"|{model.tokenizer.decode(tok)}|_{c}" for (c, tok) in enumerate(tokens)]


def plot_logit_attribution(model: HookedTransformer, logit_attr: t.Tensor, tokens: t.Tensor, title: str = ""):
    tokens = tokens.squeeze()
    y_labels = convert_tokens_to_string(model, tokens[:-1])
    x_labels = ["Direct"] + [f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    imshow(
        to_numpy(logit_attr), 
        x=x_labels, y=y_labels, 
        labels={"x": "Term", "y": "Position", "color": "logit"}, title=title if title else None, 
        height=18*len(y_labels), width=24*len(x_labels)
    )







# PLOTTING FUNCTIONS FOR PART 4: INTERP ON ALGORITHMIC MODEL

color_discrete_map = dict(zip(['both failures', 'just neg failure', 'balanced', 'just total elevation failure'], px.colors.qualitative.D3))
# names = ["balanced", "just total elevation failure", "just neg failure", "both failures"]
# colors = ['#2CA02C', '#1c96eb', '#b300ff', '#ff4800']
# color_discrete_map = dict(zip(names, colors))

def plot_failure_types_scatter(
    unbalanced_component_1: Float[Tensor, "batch"],
    unbalanced_component_2: Float[Tensor, "batch"],
    failure_types_dict: Dict[str, Float[Tensor, "batch"]],
    data
):
    failure_types = np.full(len(unbalanced_component_1), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    failures_df = pd.DataFrame({
        "Head 2.0 contribution": to_numpy(unbalanced_component_1),
        "Head 2.1 contribution": to_numpy(unbalanced_component_2),
        "Failure type": to_numpy(failure_types),
    })[data.starts_open.tolist()]
    fig = px.scatter(
        failures_df, color_discrete_map=color_discrete_map,
        x="Head 2.0 contribution", y="Head 2.1 contribution", color="Failure type", 
        title="h20 vs h21 for different failure types", template="simple_white", height=600, width=800,
        # category_orders={"color": failure_types_dict.keys()},
    ).update_traces(marker_size=4)
    fig.show()

def plot_contribution_vs_open_proportion(unbalanced_component: Float[Tensor, "batch"], title: str, failure_types_dict: Dict, data):
    failure_types = np.full(len(unbalanced_component), "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    fig = px.scatter(
        x=to_numpy(data.open_proportion), y=to_numpy(unbalanced_component), color=failure_types, color_discrete_map=color_discrete_map,
        title=title, template="simple_white", height=500, width=800,
        labels={"x": "Open-proportion", "y": f"Head {title} contribution"}
    ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
    fig.show()

def mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir: Float[Tensor, "comp batch"], 
    data, failure_types_dict: Dict
) -> None:
    failure_types = np.full(out_by_component_in_pre_20_unbalanced_dir.shape[-1], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask), name, failure_types)
    for layer in range(2):
        mlp_output = out_by_component_in_pre_20_unbalanced_dir[3+layer*3]
        fig = px.scatter(
            x=to_numpy(data.open_proportion[data.starts_open]), 
            y=to_numpy(mlp_output[data.starts_open]), 
            color_discrete_map=color_discrete_map,
            color=to_numpy(failure_types)[to_numpy(data.starts_open)], 
            title=f"Amount MLP {layer} writes in unbalanced direction for Head 2.0", 
            template="simple_white", height=500, width=800,
            labels={"x": "Open-proportion", "y": "Head 2.0 contribution"}
        ).update_traces(marker_size=4, opacity=0.5).update_layout(legend_title_text='Failure type')
        fig.show()

def plot_neurons(neurons_in_unbalanced_dir: Float[Tensor, "batch neurons"], model: HookedTransformer, data, failure_types_dict: Dict, layer: int, renderer=None):
    
    failure_types = np.full(neurons_in_unbalanced_dir.shape[0], "", dtype=np.dtype("U32"))
    for name, mask in failure_types_dict.items():
        failure_types = np.where(to_numpy(mask[to_numpy(data.starts_open)]), name, failure_types)

    # Get data that can be turned into a dataframe (plotly express is sometimes easier to use with a dataframe)
    # Plot a scatter plot of all the neuron contributions, color-coded according to failure type, with slider to view neurons
    neuron_numbers = einops.repeat(t.arange(model.cfg.d_model), "n -> (s n)", s=data.starts_open.sum())
    failure_types = einops.repeat(failure_types, "s -> (s n)", n=model.cfg.d_model)
    data_open_proportion = einops.repeat(data.open_proportion[data.starts_open], "s -> (s n)", n=model.cfg.d_model)
    df = pd.DataFrame({
        "Output in 2.0 direction": to_numpy(neurons_in_unbalanced_dir.flatten()),
        "Neuron number": to_numpy(neuron_numbers),
        "Open-proportion": to_numpy(data_open_proportion),
        "Failure type": failure_types
    })
    fig = px.scatter(
        df, 
        x="Open-proportion", y="Output in 2.0 direction", color="Failure type", animation_frame="Neuron number",
        title=f"Neuron contributions from layer {layer}", 
        template="simple_white", height=800, width=1100
    ).update_traces(marker_size=3).update_layout(xaxis_range=[0, 1], yaxis_range=[-5, 5])
    fig.show(renderer=renderer)

def plot_attn_pattern(pattern: Float[Tensor, "batch head_idx seqQ seqK"]):
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
    

def hists_per_comp(out_by_component_in_unbalanced_dir: Float[Tensor, "component batch"], data, xaxis_range=(-1, 1)):
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
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[data.isbal]), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[~data.isbal]), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()


def plot_loss_difference(log_probs, rep_str, seq_len):
    fig = px.line(
        to_numpy(log_probs), hover_name=rep_str[1:],
        title=f"Per token log-prob on correct token, for sequence of length {seq_len}*2 (repeated twice)",
        labels={"index": "Sequence position", "value": "Loss"}
    ).update_layout(showlegend=False, hovermode="x unified")
    fig.add_vrect(x0=0, x1=seq_len-.5, fillcolor="red", opacity=0.2, line_width=0)
    fig.add_vrect(x0=seq_len-.5, x1=2*seq_len-1, fillcolor="green", opacity=0.2, line_width=0)
    fig.show()