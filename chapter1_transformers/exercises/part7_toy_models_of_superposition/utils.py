import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import colors  as mcolors
from matplotlib import collections  as mc
import einops

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


def render_features(model, which=np.s_[:]):
    cfg = model.config
    W = model.W.detach().transpose(-1, -2)
    W_norm = W / (1e-5 + torch.linalg.norm(W, 2, dim=-1, keepdim=True))

    interference = torch.einsum('ifh,igh->ifg', W_norm, W)
    interference[:, torch.arange(cfg.n_features), torch.arange(cfg.n_features)] = 0

    polysemanticity = torch.linalg.norm(interference, dim=-1).cpu()
    net_interference = (interference**2 * model.feature_probability[:, None, :]).sum(-1).cpu()
    norms = torch.linalg.norm(W, 2, dim=-1).cpu()

    WtW = torch.einsum('sih,soh->sio', W, W).cpu()

    # width = weights[0].cpu()
    # x = torch.cumsum(width+0.1, 0) - width[0]
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
      x=(x[cfg.n_hidden-1]+x[cfg.n_hidden])/2, 
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