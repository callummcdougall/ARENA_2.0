import torch as t
import einops

# %%
import plotly.express as ex
import ipywidgets as widgets


def graph_cosines(dims=3, batch_size=100_000, normalize=True):
    """
    Computes the cosine similarity between batch_size random vectors of dimension `dims`.
    """
    if normalize:
        hist = t.cosine_similarity(
            t.randn(batch_size, dims), t.randn(batch_size, dims), dim=-1
        )
    else:
        hist = einops.einsum(
            "bd,bd->b", t.randn(batch_size, dims), t.randn(batch_size, dims)
        )
    fig = ex.histogram(
        hist, nbins=100, title=f"cosine similarity of random {dims}-dim vectors"
    )
    fig.show()


graph_cosines(dims=100, batch_size=100_000)

# %%
for i in range(1, 20):
    graph_cosines(dims=i, batch_size=100_000)
# %%
graph_cosines(dims=720, batch_size=100_000)
# %%
graph_cosines(dims=2000, batch_size=100_000)
# %%
