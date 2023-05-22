
# %%
import torch as t
t.cuda.get_device_name()

# %%
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

import utils

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part1_ray_tracing"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import render_lines_with_plotly, setup_widget_fig_ray, setup_widget_fig_triangle
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


# %%

DEBUG = True

def debug(l) -> None:
    if DEBUG:
        print(f"{l}")
    

# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    '''
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    '''
    r = t.zeros((num_pixels, 2, 3))
    r[:, 1, 0] = 1
    t.linspace(-y_limit, y_limit, steps=num_pixels, out=r[:, 1, 1])
    debug(r)
    return r



rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)

# %%

if MAIN:
    fig = setup_widget_fig_ray()
    display(fig)

@interact
def response(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01)):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

# %%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

utils.render_lines_with_plotly(rays1d, segments)

# %%
def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    O = ray[0, :2]
    D = ray[1, :2]
    L1 = segment[0, :2]
    L2 = segment[1, :2]

    right = L1 - O
    left = t.stack([D, L1 - L2], dim=1)

    try:
        X = t.linalg.solve(left, right)
    except RuntimeError:
        return False

    u, v = X
    return u >= 0 and 0 <= v <= 1


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%

# @jaxtyped
# @typeguard.typechecked

def intersect_ray_1d_typed(ray: Float[Tensor, "p s"],
                           segment: Float[Tensor, "p s"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''

    planar_rays = ray[..., :2]
    planar_segments = segments[..., :2]

    O = planar_rays[0]
    D = ray[1, :2]
    L1 = segment[0, :2]
    L2 = segment[1, :2]

    right = L1 - O
    left = t.stack([D, L1 - L2], dim=1)

    try:
        X = t.linalg.solve(left, right)
    except RuntimeError:
        return False

    u, v = X
    return u >= 0 and 0 <= v <= 1


intersect_ray_1d_typed(t.rand(3), t.rand(2))

# %%


# O = ray[0, :2]
# D = ray[1, :2]
# L1 = segment[0, :2]
# L2 = segment[1, :2]

# right = L1 - O
# left = t.stack([D, L1 - L2], dim=1)

# try:
#     X = t.linalg.solve(left, right)
# except RuntimeError:
#     return False

# u, v = X
# return u >= 0 and 0 <= v <= 1

def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    spacial_dims = 2
    nrays = rays.shape[0]
    nsegments = segments.shape[0]
    O = rays[:, 0, :spacial_dims]
    D = rays[:, 1, :spacial_dims]
    assert O.shape == (nrays, spacial_dims)
    assert D.shape == (nrays, spacial_dims)
    L1 = segments[:, 0, :spacial_dims]
    L2 = segments[:, 1, :spacial_dims]
    assert L1.shape == (nsegments, spacial_dims), f"{L1.shape=} {nsegments=} {spacial_dims=}"
    assert L2.shape == (nsegments, spacial_dims)
    L1s = einops.repeat(L1, "s p -> s r p", s=nsegments, p=spacial_dims, r=nrays)
    L2s = einops.repeat(L2, "s p -> s r p", s=nsegments, p=spacial_dims, r=nrays)
    Os = einops.repeat(O, "r p -> s r p" , s=nsegments, p=spacial_dims, r=nrays)
    Ds = einops.repeat(D, "r p -> s r p", s=nsegments, p=spacial_dims, r=nrays)
    
    targets = L1s - Os
    stacked = t.stack((Ds, L1s - L2s), dim=-1)
    #matrixes = einops.rearrange(stacked, "n d=2 s=2 -> ", )
    assert stacked.shape == (nsegments, nrays, spacial_dims, 2)
    singular = t.linalg.matrix_rank(stacked) != min(stacked.shape[-2], stacked.shape[-1])
    stacked[singular] = t.eye(spacial_dims, 2)
    X = t.linalg.solve(stacked, targets)
    assert X.shape == (nsegments, nrays, 2), f"{X.shape=}"
    us = X[..., 0]
    vs = X[..., 1]
    segment_ray_good = (us >= 0) & (0 <= vs) & ( vs <= 1) & ~singular
    ray_good = segment_ray_good.any(dim=0)
    return ray_good



if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    rays[..., 1, 0] = 1
    t.linspace(-y_limit, y_limit, steps=num_pixels_y, out=rays[..., 1, 1])
    t.linspace(-z_limit, z_limit, steps=num_pixels_z, out=rays[..., 1, 2])
    print(rays)
    return rays


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)


# %%
if MAIN:
    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = setup_widget_fig_triangle(x, y, z)

@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})


if MAIN:
    display(fig)

# %%
Point = Float[Tensor, "points=3"]

def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    left = t.stack([-D, B-A, C-A], dim=-1)
    right = O - A

    try:
        s, u, v = t.linalg.solve(left, right)
    except RuntimeError:
        return False

    return 0 <= u and 0 <= v and (u + v) <= 1

if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)