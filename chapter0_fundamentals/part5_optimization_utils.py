import torch as t
from torch import nn
from einops import repeat
from typing import Callable
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from torchvision import datasets

class Net(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.classifier(self.base(x))

def construct_param_config_from_description(description, model):
    param_config = []
    for param_group in description:
        param_group_ = param_group.copy()
        param_group_["params"] = getattr(model, param_group_["params"]).parameters()
        param_config.append(param_group_)
    return param_config

def format_name(name):
    return name.replace("(", "<br>   ").replace(")", "").replace(", ", "<br>   ")

def format_config(config, line_breaks=False):
    if isinstance(config, dict):
        if line_breaks:
            s = "<br>   " + "<br>   ".join([f"{key}={value}" for key, value in config.items()])
        else:
            s = ", ".join([f"{key}={value}" for key, value in config.items()])
    else:
        param_config, args_config = config
        s = "[" + ", ".join(["{" + format_config(param_group_config) + "}" for param_group_config in param_config]) + "], " + format_config(args_config)
    return s

def plot_fn(fn: Callable, x_range=[-2, 2], y_range=[-1, 3], n_points=100, log_scale=True, show_min=False):
    '''Plot the specified function over the specified domain.

    If log_scale is True, take the logarithm of the output before plotting.
    '''
    x = t.linspace(*x_range, n_points)
    xx = repeat(x, "w -> h w", h=n_points)
    y = t.linspace(*y_range, n_points)
    yy = repeat(y, "h -> h w", w=n_points)

    z = fn(xx, yy)

    fig = make_subplots(
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        rows=1, cols=2,
        subplot_titles=["3D plot", "2D log plot"]
    ).update_layout(height=700, width=1600, title_font_size=40).update_annotations(font_size=20)

    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            colorscale="greys",
            showscale=False,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{z:.2f}</b>',
            contours = dict(
                x = dict(show=True, color="grey", start=x_range[0], end=x_range[1], size=0.2),
                y = dict(show=True, color="grey", start=y_range[0], end=y_range[1], size=0.2),
                # z = dict(show=True, color="red", size=0.001)
            )
        ), row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=t.log(z) if log_scale else z,
            customdata=z,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{customdata:.2f}</b>',
            colorscale="greys",
            # colorbar=dict(tickmode="array", tickvals=contour_range, ticktext=[f"{math.exp(i):.0f}" for i in contour_range])
        ),
        row=1, col=2
    )
    fig.update_traces(showscale=False, col=2)
    if show_min:
        fig.add_trace(
            go.Scatter(
                mode="markers", x=[1.0], y=[1.0], marker_symbol="x", marker_line_color="midnightblue", marker_color="lightskyblue",
                marker_line_width=2, marker_size=12, name="Global minimum"
            ),
            row=1, col=2
        )

    return fig

def plot_optimization_sgd(opt_fn_with_sgd: Callable, fn: Callable, xy: t.Tensor, x_range=[-2, 2], y_range=[-1, 3], lr=0.001, momentum=0.98, n_iters=100, log_scale=True, n_points=100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    xys = opt_fn_with_sgd(fn, xy, lr, momentum, n_iters)
    x, y = xys.T
    z = fn(x, y)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color="red"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color="red"), line=dict(width=1, color="red")), row=1, col=2)

    fig.update_layout(showlegend=False)
    fig.data = fig.data[::-1]

    return fig

def plot_optimization(opt_fn: Callable, fn: Callable, xy: t.Tensor, optimizers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer) in enumerate(zip(px.colors.qualitative.Set1, optimizers)):
        xys = opt_fn(fn, xy.clone().detach().requires_grad_(True), *optimizer, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name = format_name(str(optimizer_active))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig

def plot_optimization_with_schedulers(opt_fn_with_scheduler: Callable, fn: Callable, xy: t.Tensor, optimizers: list, schedulers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer, scheduler) in enumerate(zip(px.colors.qualitative.Set1, optimizers, schedulers)):
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name_opt = format_name(str(optimizer_active))
        if len(scheduler) == 0:
            scheduler = (None, dict())
            name = name_opt + "<br>(no scheduler)"
        else:
            scheduler_active = scheduler[0](optimizer_active, **scheduler[1])
            name_sch = format_name(str(scheduler_active))
            name = name_opt + "<br>" + name_sch
        xys = opt_fn_with_scheduler(fn, xy.clone().detach().requires_grad_(True), *optimizer, *scheduler, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig

def get_sgd_optimizer(model, opt_config, SGD):
    if isinstance(opt_config, dict):
        return SGD(model.parameters(), **opt_config)
    else:
        opt_params = [d.copy() for d in opt_config[0]]
        _opt_config = opt_config[1]
        weight_params = [param for name, param in model.named_parameters() if "weight" in name]
        bias_params = [param for name, param in model.named_parameters() if "bias" in name]
        for param_group in opt_params:
            param_group["params"] = weight_params if param_group["params"] == "weights" else bias_params
        return SGD(opt_params, **_opt_config)

def plot_results(loss_list, accuracy_list):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=loss_list, name="Training loss"))
    fig.update_xaxes(title_text="Num batches observed")
    fig.update_yaxes(title_text="Training loss", secondary_y=False)
    # This next bit of code plots vertical lines corresponding to the epochs
    if len(accuracy_list) > 1:
        for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), len(accuracy_list), endpoint=False)):
            fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.add_trace(
            go.Scatter(y=accuracy_list, x=np.linspace(0, len(loss_list), len(accuracy_list)), mode="lines", name="Accuracy"),
            secondary_y=True
        )
    fig.update_layout(template="simple_white", title_text="Training loss & accuracy on CIFAR10")
    fig.show()

def show_cifar_images(trainset: datasets.VisionDataset, rows=3, cols=5):
    
    img = trainset.data[:rows*cols]
    fig = px.imshow(img, facet_col=0, facet_col_wrap=cols, height=150*(rows+1), width=150*(cols+1))
    for i, j in enumerate(np.arange(rows*cols).reshape(rows, cols)[::-1].flatten()):
            fig.layout.annotations[i].text = trainset.classes[trainset.targets[j]]
    fig.show()