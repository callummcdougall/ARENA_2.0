# %%
import os
import sys
import torch as t
import torch
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
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import imshow
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
import part1_ray_tracing.tests as tests

MAIN = __name__ == "__main__"


# %%
def make_rays_1d(num_pixels: int, y_limit: float) -> t.Tensor:
    """
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
    """

    points = torch.zeros((num_pixels, 2, 3))
    points[:, 1, 0] = 1
    torch.linspace(-y_limit, y_limit, num_pixels, out=points[:, 1, 1])

    return points


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
segments = t.tensor(
    [
        [[1.0, -12.0, 0.0], [1, -6.0, 0.0]],
        [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]],
        [[2, 12.0, 0.0], [2, 21.0, 0.0]],
    ]
)

render_lines_with_plotly(segments, rays1d)

# %%


# @jaxtyped
# @typeguard.typechecked
def intersect_rays_1d(
    rays: Float[Tensor, "... points dim"], segments: Float[Tensor, "... points dim"]
) -> Bool[Tensor, '...']:
    """
    ray: shape (...batches, n_points=2, n_dim=3)  # O, D points
    segment: shape (...batches, n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    rays_repeated = einops.repeat(rays, "ray point dim -> ray segment point dim",
                                  segment=segments.shape[0])
    segments_repeated = einops.repeat(segments, "segment point dim -> ray segment point dim", 
                                   ray=rays.shape[0])

    D = rays_repeated[..., 1, :2]
    O = rays_repeated[..., 0, :2]
    L_1 = segments_repeated[..., 0, :2]
    L_2 = segments_repeated[..., 1, :2]

    directions = torch.stack((D, L_1 - L_2), dim=-1) 

    # check if is singular
    det = torch.det(directions)
    singluar = torch.isclose(det, torch.zeros_like(det))

    directions_masked = torch.where(singluar[..., None, None], torch.eye(2), directions)
    uv: Tensor = torch.linalg.solve(directions_masked, L_1 - O)
    u, v = uv.unbind(dim=-1)

    is_intersection = (u > 0) & (v >= 0) & (v <= 1)
    
    return (is_intersection & ~singluar).any(dim=1)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
# %%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''

    points = torch.zeros((num_pixels_y, num_pixels_z, 2, 3))
    points[:, :, 1, 0] = 1
    points[:, :, 1, 1] = torch.linspace(-y_limit, y_limit, num_pixels_y)[:, None]
    points[:, :, 1, 2] = torch.linspace(-z_limit, z_limit, num_pixels_z)[None]
    # torch.linspace(-y_limit, y_limit, num_pixels_y, out=points[:, :, 1, 1])
    # torch.linspace(-z_limit, z_limit, num_pixels_z, out=points[:, :, 1, 2])

    points = einops.rearrange(points, 'pix_y pix_z point dim -> (pix_y pix_z) point dim')
    return points 


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

@jaxtyped
@typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    directions = torch.stack((-D, B-A, C-A), dim=1)
    dist_from_O = O-A
    try:
        s, u, v = torch.linalg.solve(directions, dist_from_O)
    except ValueError:
        return False
    return ((s > 0) & (u >= 0) & (v >= 0) & (u+v < 1)).item()


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %%

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    A, B, C = einops.repeat(triangle, "point dim -> point nrays dim", nrays=rays.shape[0])
    O, D = rays.unbind(dim=1)

    directions = torch.stack((-D, B-A, C-A), dim=2)
    dist_from_O = O-A
    s, u, v = torch.linalg.solve(directions, dist_from_O).unbind(-1)
    return ((s >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1))
 


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

#%%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    A, B, C = einops.repeat(triangles, "triangle point dim -> point ray triangle dim",
                            ray=rays.shape[0])
    O, D = einops.repeat(rays, "ray point dim -> point ray triangle dim",
                         triangle=triangles.shape[0])
    directions = torch.stack((-D, B-A, C-A), dim=-1)
    dist_from_O = O-A
    s, u, v = torch.linalg.solve(directions, dist_from_O).unbind(-1)
    intersects = ((s >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1))
    return torch.where(intersects, s, float('inf')).min(dim=1).values
    # D = rays_repeated[..., 1, :2]
    # O = rays_repeated[..., 0, :2]
    # L_1 = segments_repeated[..., 0, :2]
    # L_2 = segments_repeated[..., 1, :2]

    # directions = torch.stack((D, L_1 - L_2), dim=-1) 

    # # check if is singular
    # det = torch.det(directions)
    # singluar = torch.isclose(det, torch.zeros_like(det))

    # directions_masked = torch.where(singluar[..., None, None], torch.eye(2), directions)
    # uv: Tensor = torch.linalg.solve(directions_masked, L_1 - O)
    # u, v = uv.unbind(dim=-1)

    # is_intersection = (u > 0) & (v >= 0) & (v <= 1)
    
    return (is_intersection & ~singluar).any(dim=1) 


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()

#%%
#%%
LIGHT = Tensor([0, 0, 0])

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Float[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    A, B, C = einops.repeat(triangles, "triangle point dim -> point ray triangle dim",
                            ray=rays.shape[0])
    O, D = einops.repeat(rays, "ray point dim -> point ray triangle dim",
                         triangle=triangles.shape[0])
    directions = torch.stack((-D, B-A, C-A), dim=-1)
    dist_from_O = O-A
    s, u, v = torch.linalg.solve(directions, dist_from_O).unbind(-1)
    intersects = ((s >= 0) & (u >= 0) & (v >= 0) & (u+v <= 1))
    hit_index = torch.where(intersects, s, float('inf')).min(dim=1).indices

    triangle_hit = triangles[hit_index]
    print(triangle_hit.shape)
    hit_a, hit_b, hit_c = triangle_hit.unbind(1)
    d = torch.stack([LIGHT - hit_a, hit_b - hit_a, hit_c - hit_a], dim=-1) 
    print(d.shape)
    d /= d.norm(dim=1)[:, None]
    angle = -torch.det(d)
    return torch.where(intersects.any(dim=1), angle, -1)

    # D = rays_repeated[..., 1, :2]
    # O = rays_repeated[..., 0, :2]
    # L_1 = segments_repeated[..., 0, :2]
    # L_2 = segments_repeated[..., 1, :2]

    # directions = torch.stack((D, L_1 - L_2), dim=-1) 

    # # check if is singular
    # det = torch.det(directions)
    # singluar = torch.isclose(det, torch.zeros_like(det))

    # directions_masked = torch.where(singluar[..., None, None], torch.eye(2), directions)
    # uv: Tensor = torch.linalg.solve(directions_masked, L_1 - O)
    # u, v = uv.unbind(dim=-1)

    # is_intersection = (u > 0) & (v >= 0) & (v <= 1)
    
    return (is_intersection & ~singluar).any(dim=1) 


if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()
# %%

