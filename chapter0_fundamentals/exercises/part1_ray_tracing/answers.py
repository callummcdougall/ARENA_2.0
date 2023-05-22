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
    rays = t.zeros(num_pixels, 2, 3)
    rays[:, 1, 0] = 1
    # rays[:, 1, 1] = t.linspace(-y_limit, y_limit, num_pixels)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    return rays

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

# %%
if MAIN:
    fig = render_lines_with_plotly(rays1d, segments)

# %%


def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    x = segment[0] - segment[1] # [3]
    D = ray[1]

    A = t.stack((D, x), dim=0)[:, :2]
    B = (segment[0] - ray[0])[:2]

    try:
        u_v = t.linalg.solve(A.T, B)
    except RuntimeError:
        return False

    return  u_v[0] >= 0 and 0 <= u_v[1] <= 1



if MAIN:
    # intersect_ray_1d(rays1d[0], segments[0])
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%


def intersect_rays_1d_not_finished(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    rays = rays[...,:2]
    segments = segments[...,:2]

    O = rays[:, 0, :]
    D = rays[:, 1, :]
    L1 = segments[:, 0, :]
    L2 = segments[:, 1, :]

    r_O = einops.repeat(O, 'nrays coords -> nrays nsegments coords', nsegments=segments.shape[0])
    r_D = einops.repeat(D, 'nrays coords -> nrays nsegments coords', nsegments=segments.shape[0])
    
    r_L1 = einops.repeat(L1, 'nsegments coords -> nrays nsegments coords', nrays=rays.shape[0])
    r_L2 = einops.repeat(L2, 'nsegments coords -> nrays nsegments coords', nrays=rays.shape[0])
    
    L1_L2 = r_L1 - r_L2  # nrays nsegments coords
    A = t.stack((r_D, L1_L2), dim=2)  # nrays nsegments 2 coords
    B = r_L1 - r_O
    assert B.shape == (rays.shape[0], segments.shape[0], 2)
    
    # nrays, nsegments, 2
    u_v = t.linalg.solve(einops.rearrange(A, 'nrays nsegments x coords -> nrays nsegments coords x'), B)
    print(u_v)
    u, v = u_v[...,0], u_v[...,1]
    u, v = u_v.unbind(-1)

    intersections = (v >= 0) & (v <= 1) & (u >= 0)

    return einops.reduce(intersections, "rays segments -> rays", "max")

from solutions import intersect_rays_1d, make_rays_2d



if MAIN:
    print(intersect_rays_1d(rays1d[:3], segments[:2]))
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
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
    mat = t.stack((
        -1 * D,
        B - A,
        C - A,
    ), dim=-1)
    vec = O - A

    try:
        sol = t.linalg.solve(mat, vec)
    except RuntimeError:
        return False

    s, u, v = sol
    return (u >= 0 and v >= 0 and u + v <= 1).item()


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
    r_triangle = einops.repeat(triangle, 'points dims -> nrays points dims', nrays=rays.shape[0])
    A, B, C = r_triangle.unbind(-2)  # nrays dims
    O, D = rays.unbind(-2)  # nrays dims

    mat = t.stack((
        -1 * D,
        B - A,
        C - A,
    ), dim=-1)  # nrays dims 3
    
    print(mat.device)
    # assert mat.device == "cuda:0"
    vec = O - A
    
    sol = t.linalg.solve(mat, vec)  # nrays
    
    s, u, v = sol.unbind(-1)
    result = (u >= 0) & (v >= 0) & (u + v <= 1)

    return result


if MAIN:
    A = t.tensor([1, 0.0, -0.5]).cuda()
    B = t.tensor([1, -0.5, 0.0]).cuda()
    C = t.tensor([1, 0.5, 0.5]).cuda()

    num_pixels_y = num_pixels_z = 15
    # num_pixels_y = num_pixels_z = 2
    y_limit = z_limit = 0.5

    # Plot triangle & rays
    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit).cuda()
    triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d.cpu(), triangle_lines.cpu())

    # Calculate and display intersections
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z).int()
    imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")

# %%
if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

# %%
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    n_triangles = triangles.shape[0]
    n_rays = rays.shape[0]
    r_rays = einops.repeat(rays, "nrays points dims -> nrays ntriangles points dims", ntriangles=n_triangles)
    r_triangles = einops.repeat(triangles, "ntriangles points dims -> nrays ntriangles points dims", nrays=n_rays)

    A, B, C = r_triangles.unbind(2)
    O, D = r_rays.unbind(2)

    mat = t.stack(( -1 * D, B - A, C - A,), dim=-1)  # nrays ntriangles 3
    vec = O - A

    sol = t.linalg.solve(mat, vec) # nrays ntriangles 3
    s, u, v = sol.unbind(-1) # nrays ntriangles

    u[u < 0] = float("inf")
    u[v < 0] = float("inf")
    u[u + v > 1] = float("inf")

    return einops.reduce(u, "nrays ntraingles -> nrays", "min")



if MAIN:
    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
	# Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    rot = t.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=t.float32)
    rays[:, 0] = t.tensor([-2.5, 0.0, 0.0])
    rays = rays @ rot
    # rays = einops.einsum(rays, rot, "nrays npoints ndims, x y -> ")
    dists = raytrace_mesh(rays, triangles)
    intersects = t.isfinite(dists).view(num_pixels_y, num_pixels_z)
    dists_square = dists.view(num_pixels_y, num_pixels_z)
    img = t.stack([intersects, dists_square], dim=0)

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma", width=1000)
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()