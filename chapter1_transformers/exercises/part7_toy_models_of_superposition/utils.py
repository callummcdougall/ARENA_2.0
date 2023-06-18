import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import colors  as mcolors
from matplotlib import collections  as mc
import einops
import plotly.express as px

def custom_colors(n):
    assert n <= 5, "Only supports up to 5 colors."
    single_colors = np.array([
        [1.0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0.55, 0, 1],
        [1, 0.5, 0, 1],
        [0.5, 0, 1, 1],
    ])[:n]
    repeated_colors = einops.repeat(single_colors, "colors rgba -> (colors pair) rgba", pair=2)
    return repeated_colors

def plot_Ws_from_model(model, config):
    cfg = model.config
    WA = model.W.detach().transpose(-1, -2)
    n_features = config.n_features
    sel = range(config.n_instances) # can be used to highlight specific sparsity levels
    n_uncorrelated = n_features - 2 * (config.n_correlated_pairs + config.n_anticorrelated_pairs)
    uncorrelated_colors = plt.cm.viridis(model.importance[0][:n_uncorrelated].cpu())
    correlated_colors = custom_colors(config.n_correlated_pairs + config.n_anticorrelated_pairs)
    main_colors = np.concatenate([correlated_colors, uncorrelated_colors])
    main_cycler = plt.cycler("color", main_colors)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", main_cycler)
    plt.rcParams['figure.dpi'] = 200
    fig, axs = plt.subplots(1,len(sel), figsize=(2*len(sel),2))
    for i, ax in zip(sel, axs):
        W = WA[i].cpu().detach().numpy()
        colors = [mcolors.to_rgba(c)
            for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
        ax.scatter(W[:,0], W[:,1], c=colors[0:len(W[:,0])])
        ax.set_aspect('equal')
        ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W),W), axis=1), colors=colors))
        
        z = 1.5
        ax.set_facecolor('#FCFBF8')
        ax.set_xlim((-z,z))
        ax.set_ylim((-z,z))
        ax.tick_params(left = True, right = False , labelleft = False ,
                    labelbottom = False, bottom = True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom','left']:
            ax.spines[spine].set_position('center')
    plt.show()


def plot_W(W):
    N = 1
    sel = range(N)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis([1.0, 0.85, 0.7, 0.55, 0.4]))
    plt.rcParams['figure.dpi'] = 200
    fig, ax = plt.subplots(figsize=(2*len(sel),2))
    W_numpy = W.T.cpu().detach().numpy()
    colors = [mcolors.to_rgba(c)
        for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    ax.scatter(W[0], W[1], c=colors[0:len(W[0])])
    ax.set_aspect('equal')
    ax.add_collection(mc.LineCollection(np.stack((np.zeros_like(W_numpy), W_numpy), axis=1), colors=colors))
    
    z = 1.5
    ax.set_facecolor('#FCFBF8')
    ax.set_xlim((-z,z))
    ax.set_ylim((-z,z))
    ax.tick_params(left = True, right = False , labelleft = False ,
                labelbottom = False, bottom = True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom','left']:
        ax.spines[spine].set_position('center')
    plt.show()


def plot_feature_geometry(model, dim_fracs = None):
    fig = px.line(
        x=1/model.feature_probability[:, 0].cpu(),
        y=(model.config.n_hidden/(torch.linalg.matrix_norm(model.W.detach(), 'fro')**2)).cpu(),
        log_x=True,
        markers=True,
        template="ggplot2",
        height=600,
        width=1000,
        title=""
    )
    fig.update_xaxes(title="1/(1-S), <-- dense | sparse -->")
    fig.update_yaxes(title=f"m/||W||_F^2")
    if dim_fracs is not None:
        dim_fracs = dim_fracs.detach().cpu().numpy()
        density = model.feature_probability[:, 0].cpu()

        for a,b in [(1,2), (2,3), (2,5), (2,6), (2,7)]:
            val = a/b
            fig.add_hline(val, line_color="purple", opacity=0.2, annotation=dict(text=f"{a}/{b}"))

        for a,b in [(5,6), (4,5), (3,4), (3,8), (3,12), (3,20)]:
            val = a/b
            fig.add_hline(val, line_color="blue", opacity=0.2, annotation=dict(text=f"{a}/{b}", x=0.05))

        for i in range(len(dim_fracs)):
            fracs_ = dim_fracs[i]
            N = fracs_.shape[0]
            xs = 1/density
            if i!= len(dim_fracs)-1:
                dx = xs[i+1]-xs[i]
            fig.add_trace(
                go.Scatter(
                    x=1/density[i]*np.ones(N)+dx*np.random.uniform(-0.1,0.1,N),
                    y=fracs_,
                    marker=dict(
                        color='black',
                        size=1,
                        opacity=0.5,
                    ),
                    mode='markers',
                )
            )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(showlegend=False)
    fig.show()



def render_features(model, which=np.s_[:]):
    cfg = model.config
    W = model.W.detach().transpose(-1, -2)
    W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

    interference = torch.einsum('ifh,igh->ifg', W_norm, W)
    interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

    polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
    norms = torch.linalg.norm(W, 2, dim=-1).cpu()

    WtW = torch.einsum('sih,soh->sio', W, W).cpu()

    x = torch.arange(cfg.n_features)
    width = 0.9

    which_instances = np.arange(cfg.n_instances)[which]
    fig = make_subplots(rows=len(which_instances),
                        cols=2,
                        shared_xaxes=True,
                        vertical_spacing=0.02,
                        horizontal_spacing=0.1)
    for (row, inst) in enumerate(which_instances):
        fig.add_trace(
            go.Bar(x=x, 
                  y=norms[inst],
                  marker=dict(
                      color=polysemanticity[inst],
                      cmin=0,
                      cmax=1
                  ),
                  width=width,
            ),
            row=1+row, col=1
        )
        data = WtW[inst].numpy()
        fig.add_trace(
            go.Image(
                z=plt.cm.coolwarm((1 + data)/2, bytes=True),
                colormodel='rgba256',
                customdata=data,
                hovertemplate='''\
    In: %{x}<br>
    Out: %{y}<br>
    Weight: %{customdata:0.2f}
    '''            
            ),
            row=1+row, col=2
        )

    fig.add_vline(
      x=cfg.n_hidden-0.5, 
      line=dict(width=0.5),
      col=1,
    )
      
    # fig.update_traces(marker_size=1)
    fig.update_layout(
        showlegend=False, 
        width=600,
        height=100*len(which_instances),
        margin=dict(t=40, b=40)
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# Neuron stack plots (work in progress)
# Also, the plots aren't very good because the training is hard, see comments in blog post

# def render_neurons(model, which=np.s_[:]):
#     cfg: Config = model.config
#     W = model.W.detach().transpose(-1, -2) # shape (instances, features, neurons)
#     W_norm = W / (1e-5 + t.linalg.norm(W, 2, dim=-1, keepdim=True))

#     interference = einops.einsum(W_norm, W, "inst feat1 hidden, inst feat2 hidden -> inst feat1 feat2")
#     interference[:, t.arange(cfg.n_features), t.arange(cfg.n_features)] = 0
#     # interference[:, t.arange(cfg.n_features), t.arange(cfg.n_features)] = 0
#     # print(interference.shape)

#     polysemanticity = interference.norm(dim=-1).cpu()

#     x = t.arange(cfg.n_hidden)

#     which_instances = np.arange(cfg.n_instances)[which]
#     fig = make_subplots(
#         rows=len(which_instances),
#         cols=2, # 3,
#         shared_xaxes=True,
#         vertical_spacing=0.02,
#         horizontal_spacing=0.0,
#         # specs=[[{}, {"colspan": 2}, None] for _ in range(len(which_instances))]
#     )
#     for (row, inst) in enumerate(which_instances):
#         for feature_idx in range(cfg.n_features):
#             fig.add_trace(
#                 go.Bar(
#                     x=x.cpu().numpy(), 
#                     y=W[inst, feature_idx].cpu().numpy(),
#                     marker=dict(
#                         color=np.clip(polysemanticity[inst].cpu().numpy(), 0.0, 0.85),
#                         cmin=0,
#                         cmax=1,
#                     ),
#                 ),
#                 row=1+row, col=2
#             )
#         fig.add_hline(y=0.0, line_width=1.0, col=2, row=1+row)
#         data = W.cpu().numpy()[inst]
#         fig.add_trace(
#             go.Image(
#                 z=plt.cm.coolwarm((1 + data)/2, bytes=True),
#                 colormodel='rgba256',
#                 customdata=data,
#             ),
#             row=1+row, col=1
#         )

#     # fig.update_traces(marker_size=1)
#     fig.update_layout(
#         showlegend=False, 
#         width=320,
#         height=180*len(which_instances),
#         margin=dict(t=40, b=40, r=0, l=0),
#         barmode='overlay',
#     )
#     fig.update_xaxes(visible=False)
#     fig.update_yaxes(visible=False)
#     # print(fig["layout"])
#     for i in range(1, 1+2*len(which_instances), 2):
#         fig["layout"][f"xaxis{i}"]["domain"] = [0.2, 0.5]
#     for i in range(2, 1+2*len(which_instances), 2):
#         fig["layout"][f"xaxis{i}"]["domain"] = [0.55, 0.8]
#         fig["layout"][f"yaxis{i}"]["range"] = [-2.5, 2.5]
#     return fig


# n_instances = 6
# n_features = 10

# importance = 0.75 ** t.arange(n_features)[None, :]
# feature_probability = t.tensor([0.6, 0.3, 0.12, 0.08, 0.05, 0.02])[:, None]

# config = Config(
#     n_instances = n_instances,
#     n_features = n_features,
#     n_hidden = 5,
# )

# model = NeuronModel(
#     config=config,
#     device=device,
#     feature_probability=feature_probability,
# )

# optimize(model, steps=10_000)

# fig = render_neurons(model)

# fig.show()