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
    # rays = []

    # for _ in range(num_pixels):
    #     rays.append(t.stack((t.zeros(3), t.FloatTensor(3).uniform_(-y_limit, y_limit))))

    # return t.stack(rays)

    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

rays1d = make_rays_1d(9, 10.0)
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

if MAIN:
    print(rays1d)
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
@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray: Float[t.Tensor, "points=2 dim=3"], segment: Float[t.Tensor, "points=2 dim=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    def find_intersection_point(lhs_matrix: Float[t.Tensor, "points=2 dim=2"], rhs_matrix: Float[t.Tensor, "points=2 dim=1"]):
        if t.linalg.det(lhs_matrix).item():
            return t.linalg.solve(lhs_matrix, rhs_matrix)
        return None
    
    segment_direction = segment[0] - segment[1]

    lhs_matrix = t.vstack((
        t.hstack((ray[1][0], segment_direction[0])),
        t.hstack((ray[1][1], segment_direction[1]))
    ))
    rhs_matrix = t.vstack((
        (segment[0] - ray[0])[0],
        (segment[0] - ray[0])[1]
    ))
    
    intersection_point = find_intersection_point(lhs_matrix, rhs_matrix)
    if intersection_point is None:
        return False
    return intersection_point[0] >= 0 and 0 <= intersection_point[1] <= 1 


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
  
# %%
from typing import Tuple

rays = einops.repeat(rays1d, "nr a b -> nr ns a b", nr=len(rays1d), ns=len(segments), a=2, b=3)
esegments = einops.repeat(segments, "ns a b -> nr ns a b", nr=len(rays1d), ns=len(segments), a=2, b=3)
print(f"rays: {rays.shape}")
print(f"segments: {esegments.shape}")

dx: Float[Tensor, "nrays 1"] = rays[:,:,1,0]
dy: Float[Tensor, "nrays 1"] = rays[:,:,1,1]

print(f"D_x of every ray: {dx.shape}")
print(f"D_y of every ray: {dy.shape}")

print(f"L1 of every segment: {esegments[:,:,0].shape}")
print(f"L2 of every segment: {esegments[:,:,1].shape}")

print(f"segment directions: {(esegments[:,:,0] - esegments[:,:,1]).shape}")

segments_directions: Float[Tensor, "nr ns  3"] = esegments[:,:,0] - esegments[:,:,1]

segments_direction_x: Float[Tensor, "nr ns 1"] = segments_directions[:,:,0]
segments_direction_y: Float[Tensor, "nr ns 1"] = segments_directions[:,:,1]
print(f"x of segments directions {segments_direction_x.shape}")
print(f"y of segments directions {segments_direction_y.shape}")

print(f"lhs entry:  {t.stack((dx, segments_direction_x), dim=-1).shape}")

lhs: Float[Tensor, "nr ns 2 2"] = t.stack((
        t.stack((dx, segments_direction_x), dim=-1),
        t.stack((dy, segments_direction_y), dim=-1)
), dim=-1)

print(f"lhs: {lhs.shape}")

rhs: Float[Tensor, "nrays 2"] = t.stack((
    (esegments[:,:,0] - rays[:,:,0])[:,:,0], 
    (esegments[:,:,0] - rays[:,:,0])[:,:,1]
    ), dim=-1)
print(f"rhs: {rhs.shape}")

singular: Float[Tensor, "nrays"] = t.linalg.det(lhs) < 1e-6

print(f"singular: {singular.shape}")

# replace non-invertible matrices with identity
lhs[singular] = t.eye(2)

# solve all the systems
intersection_points = t.linalg.solve(lhs, rhs)

print(f"intersection points: {intersection_points.shape}")

# create boolean array with entries corresponding to if solution satisfies constraints 
on_segment = (intersection_points[:,:,0] >= 0) & (0 <= intersection_points[:,:,1]) & (intersection_points[:,:,1] <= 1)
print(f"on segment: {on_segment.shape}")

# do an and with that array and the array for if a matrix is invertible
valid_intersections = on_segment & singular

print(t.any(valid_intersections, dim=1).shape)

# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    rays: Float[Tensor, "nr ns 2 3"] = einops.repeat(rays, "nr a b -> nr ns a b", nr=len(rays1d), ns=len(segments), a=2, b=3)
    segments: Float[Tensor, "nr ns 2 3"] = einops.repeat(segments, "ns a b -> nr ns a b", nr=len(rays1d), ns=len(segments), a=2, b=3)

    dx: Float[Tensor, "nrays 1"] = rays[:,:,1,0]
    dy: Float[Tensor, "nrays 1"] = rays[:,:,1,1]

    # print(f"D_x of every ray: {dx.shape}")
    # print(f"D_y of every ray: {dy.shape}")

    # print(f"L1 of every segment: {esegments[:,:,0].shape}")
    # print(f"L2 of every segment: {esegments[:,:,1].shape}")

    # print(f"segment directions: {(esegments[:,:,0] - esegments[:,:,1]).shape}")

    segments_directions: Float[Tensor, "nr ns  3"] = segments[:,:,0] - segments[:,:,1]

    segments_direction_x: Float[Tensor, "nr ns 1"] = segments_directions[:,:,0]
    segments_direction_y: Float[Tensor, "nr ns 1"] = segments_directions[:,:,1]
    # print(f"x of segments directions {segments_direction_x.shape}")
    # print(f"y of segments directions {segments_direction_y.shape}")

    # print(f"lhs entry:  {t.stack((dx, segments_direction_x), dim=-1).shape}")

    lhs: Float[Tensor, "nr ns 2 2"] = t.stack((
            t.stack((dx, segments_direction_x), dim=-1),
            t.stack((dy, segments_direction_y), dim=-1)
    ), dim=-1)

    # print(f"lhs: {lhs.shape}")

    rhs: Float[Tensor, "nrays 2"] = t.stack((
        (segments[:,:,0] - rays[:,:,0])[:,:,0], 
        (segments[:,:,0] - rays[:,:,0])[:,:,1]
    ), dim=-1)
    # print(f"rhs: {rhs.shape}")

    singular: Float[Tensor, "nrays"] = t.linalg.det(lhs) < 1e-6

    # replace non-invertible matrices with identity
    lhs[singular] = t.eye(2)

    # solve all the systems
    intersection_points = t.linalg.solve(lhs, rhs)

    # create boolean array with entries corresponding to if solution satisfies constraints 
    on_segment = (intersection_points[:,:,0] >= 0) & (0 <= intersection_points[:,:,1]) & (intersection_points[:,:,1] <= 1)

    # do an and with that array and the array for if a matrix is invertible
    valid_intersections = on_segment & (~singular)

    return t.any(valid_intersections, dim=1)

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
    rays_y = t.zeros((num_pixels_y, 2), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels_y, out=rays_y[:, 1])

    rays_z = t.zeros((num_pixels_z, 2), dtype=t.float32)
    t.linspace(-z_limit, z_limit, num_pixels_z, out=rays_z[:, 1])

    rays_x = t.zeros(num_pixels_y * num_pixels_z, 2)
    rays_x[:, 1] = 1

    rays_z_extended = einops.repeat(rays_z, "nz p -> (nz ny) p", ny=num_pixels_y)
    rays_y_extended = einops.repeat(rays_y, "ny p -> (nz ny) p", nz=num_pixels_z)
    return t.stack((rays_x, rays_y_extended, rays_z_extended), dim=-1)



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

# @jaxtyped
# @typeguard.typechecked
def triangle_ray_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the ray and the triangle intersect.
    '''
    def find_intersection_point(lhs: Float[t.Tensor, "3 3"], rhs: Float[t.Tensor, "3"]):
        if t.linalg.det(lhs).item():
            return t.linalg.solve(lhs, rhs)
        return None

    lhs = t.stack([
        -D, B - A, C - A 
    ], dim=-1)

    rhs = O - A

 
    intersection_point = find_intersection_point(lhs, rhs)
    if intersection_point is None:
        return False
    
    u, v = intersection_point[1:]
    return u >= 0 and v >= 0 and u + v <= 1

    return True


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    NR = len(rays)
    NT = len(triangles)
    rays: Float[Tensor, "nrays ntriangles rayPoints=2 dims=3"] = einops.repeat(rays, "nr rp d -> nr nt rp d", nt=NT)
    triangles: Float[Tensor, "nrays ntriangles rayPoints=2 dims=3"] = einops.repeat(triangles, "nt rp d -> nr nt rp d", nr=NR)

    A = triangles[:,:,0]
    B = triangles[:,:,]
    
    D = rays[:,:,1]



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