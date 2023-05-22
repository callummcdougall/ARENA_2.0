
#%%
import os
import sys
import torch as t
from torch import Tensor
import einops
from ipywidgets import interact
import plotly.express as px
from ipywidgets import interact
from pathlib import Path
from IPython.display import display
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"




#%%
x = t.randn(2, 3)
x_repeated = einops.repeat(x, 'a b ->  a b c', c=3)

print(x, '\n', x_repeated)

assert x_repeated.shape == (2, 3, 4)
for c in range(4):
    t.testing.assert_close(x, x_repeated[:, :, c])



#%%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

rays1d = t.tensor([[[  0.0000,   0.0000,   0.0000], [  1.0000, -10.0000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,  -7.5000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,  -5.0000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,  -2.5000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,   0.0000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,   2.5000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,   5.0000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,   7.5000,   0.0000]],
        [[  0.0000,   0.0000,   0.0000],
         [  1.0000,  10.0000,   0.0000]]])



# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    n_rays = rays.shape[0]
    n_segments = segments.shape[0]
    global A

    rays_repeated = einops.repeat(rays, "nrays pos dim -> nrays nsegments pos dim", nsegments=n_segments)
    segments_repeated = einops.repeat(segments, "nsegments pos dim -> nrays nsegments pos dim", nrays=n_rays)

    A = t.zeros(n_rays, n_segments, 2, 2)
    print(A.shape)
    A[:, :, :, 0] = rays_repeated[:, :, 1, :2]
    A[:, :, :, 1] = segments_repeated[:, :, 0, :2] - segments_repeated[:, :, 1, :2]

    # check singularity singularity
    sing_mask = (t.linalg.det(A).abs() < 1e-6)
    print(sing_mask)

    # replace with singular values
    A[sing_mask] = t.eye(2)

    b = segments_repeated[:, :, 0, :2]
    print("b shape ", b.shape)
    x = t.linalg.solve(A, b) # u,v of shape nrays nsegments
    u, v = einops.rearrange(x, "nrays nsegments uv -> uv nrays nsegments")


    
    
    # print(A)

intersect_rays_1d(rays1d, segments)




#%%
if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)