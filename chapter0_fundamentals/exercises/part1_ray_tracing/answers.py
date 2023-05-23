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
sys.path

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
    t.linspace(
        start = -y_limit,
        end = y_limit,
        steps = num_pixels,
        out = rays[:, 1, 1])
    
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
def check_intersection(rays, line):
    A = t.zeros(rays.shape[0],2,2)
    A[:,:,0] = rays[:,1,:2]
    A[:,:,1] = line[0,:2] - line[1,:2]
    b = line[0,:2] 
    try:
        x = t.linalg.solve(A, b)
    except:
        return False
    
    return (x[:,0]>=0) & (x[:,1]>=0) & (x[:,1]<=1)

render_lines_with_plotly(rays1d, segments[2].unsqueeze(0))
check_intersection(rays1d, segments[2])

# %%



def intersect_ray_1d(ray: Float[t.Tensor, "points=2 dims=3"], segment: Float[t.Tensor, "points=2 dims=3"]) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    # SOLUTION
    # Get the x and y coordinates (ignore z)
    ray = ray[..., :2]
    segment = segment[..., :2]

    # Ray is [[Ox, Oy], [Dx, Dy]]
    O, D = ray
    # Segment is [[L1x, L1y], [L2x, L2y]]
    L_1, L_2 = segment

    # Create matrix and vector, and solve equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    vec = L_1 - O

    # Solve equation (return False if no solution)
    try:
        sol = t.linalg.solve(mat, vec)
    except:
        return False

    # If there is a solution, check the soln is in the correct range for there to be an intersection
    u = sol[0].item()
    v = sol[1].item()
    return (u >= 0.0) and (v >= 0.0) and (v <= 1.0)

intersect_ray_1d(rays1d[0] , segments[0])

# %%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    n_rays = rays.shape[0]
    n_segments = segments.shape[0]

    rays_repeated = einops.repeat(rays, "nrays pos dim -> nrays nsegments pos dim", nsegments=n_segments)
    segments_repeated = einops.repeat(segments, "nsegments pos dim -> nrays nsegments pos dim", nrays=n_rays)

    A = t.zeros(n_rays, n_segments, 2, 2)
    A[:, :, :, 0] = rays_repeated[:, :, 1, :2]
    A[:, :, :, 1] = segments_repeated[:, :, 0, :2] - segments_repeated[:, :, 1, :2]
    b = segments_repeated[:, :, 0, :2]
    invertible = (t.linalg.det(A).abs() < 1e-6)
    A[invertible] = t.eye(2)
    x = t.linalg.solve(A,b) # u,v of shape nrays nsegments
    u, v = einops.rearrange(x, "nrays nsegments uv -> uv nrays nsegments")

    bool_tensor = (u >= 0.0) & (v >= 0.0) & (v <= 1.0) & (invertible==False)
    return t.any(bool_tensor, dim=1)

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
    # SOLUTION
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays

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
    matrix = t.stack((D, B-A, C-A)).T
    print(matrix)
    x = t.linalg.solve(matrix, O-A)
    print(x)

    s, u, v = x.unbind(0)

    return ((u >= 0) & (v >= 0) & (u+v <= 1)).item()

#%%

D = t.tensor([1, 0, 0], dtype=t.float32)
O = t.tensor([0, 0, 0], dtype=t.float32)
A = t.tensor([1, 0, 1], dtype=t.float32)
B = t.tensor([1, 1, -1], dtype=t.float32)
C = t.tensor([1, -1, -1], dtype=t.float32)

triangle_ray_intersects(A, B, C, O, D)


#%%
if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)
# %% Test raytrace_triangle

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''

    nrays = rays.shape[0]
    # Repeat triangle along nrays dimension
    A, B, C = einops.repeat(triangle, "trianglePoints dims -> trianglePoints nrays dims", nrays=nrays)
    assert A.shape == (nrays, 3)

    # build matrix of linear equations
    mat: Float[t.Tensor, "nrays 3 3"] = t.stack((rays[:, 1], B-A, C-A), dim =-1)
    assert mat.shape == (nrays, 3, 3)

    # check for singularity
    is_singular: Bool[t.Tensor, "nrays"] = t.linalg.det(mat).abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # solve set of linear equations
    sol = t.linalg.solve(mat, rays[:, 0] - A)
    s, u, v = sol.unbind(dim=-1)

    # check of requirements to u and v are fulfilled and singularity mask is false
    return ((u >= 0) & (v >= 0) & (u+v <= 1) & ~is_singular)




#%%
ray = t.stack((O, D))
rays = t.stack((ray, ray))
triangle = t.stack((A, B, C))
triangles = t.stack((triangle, triangle))


raytrace_triangle(rays, triangle)

#%%



if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 30
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

# %% Debugging

def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(1)

    mat = t.stack([- D, B - A, C - A], dim=-1)

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)

intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")



#%% raytrace mesh
def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    # retrieve nrays ntriangles
    NR = rays.shape[0]
    NT = triangles.shape[0]

    # repeat along axes
    rays_repeated = einops.repeat(rays, "nrays rayPoints dims -> nrays ntriangles rayPoints dims", ntriangles = NT)
    triangles_repeated = einops.repeat(triangles, "ntriangles trianglePoints dims -> nrays ntriangles trianglePoints dims", nrays = NR)

    # retrieve single vectors

    O, D = rays_repeated.unbind(dim=2)
    A, B, C = triangles_repeated.unbind(dim=2)

    # build mat
    mat: Float[Tensor, "nrays rayPoints 3 3"] = t.stack((-D, B-A, C-A), dim=-1)
    assert mat.shape == (NR, NT, 3, 3)

    # check for singularities
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # do solver
    sol = t.linalg.solve(mat, O-A)
    assert sol.shape == (NR, NT, 3)

    s, u, v = sol.unbind(dim=-1)

    # where u, v not fulfilled, plug inf to s
    is_intersect = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~is_intersect] = t.inf
    assert s.shape == (NR,NT)

    # min s along all axes
    s = t.min(s, dim=-1, out=(s, idxs))

    return s

    


#%% raytrace mesh simple test

D = t.tensor([1, 0, 0], dtype=t.float32)
O = t.tensor([0, 0, 0], dtype=t.float32) # setting O back
A = t.tensor([1, 0, 1], dtype=t.float32)
B = t.tensor([1, 1, -1], dtype=t.float32)
C = t.tensor([1, -1, -1], dtype=t.float32)

ray = t.stack((O, D))
rays = t.stack((ray, ray))
triangle = t.stack((A, B, C))
triangles = t.stack((triangle, triangle))

raytrace_mesh(rays, triangles)





#%% raytrace mesh test

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
