import plotly.express as px
import plotly.graph_objects as go
from torchtyping import TensorType as TT
from typing import Optional
from transformer_lens import utils, HookedTransformer
import torch as t

WIP = r"../../images/written_images"
def save_fig(fig, filename):
    with open(f"{WIP}/{filename}.html", "w") as f:
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))

def plot_comp_scores(model: HookedTransformer, comp_scores: TT["heads", "heads"], title: str = "", baseline: Optional[t.Tensor] = None) -> go.Figure:
    return px.imshow(
        utils.to_numpy(comp_scores),
        y=[f"L0H{h}" for h in range(model.cfg.n_heads)],
        x=[f"L1H{h}" for h in range(model.cfg.n_heads)],
        labels={"x": "Layer 1", "y": "Layer 0"},
        title=title,
        color_continuous_scale="RdBu" if baseline is not None else "Blues",
        color_continuous_midpoint=baseline if baseline is not None else None,
        zmin=None if baseline is not None else 0.0,
    )
