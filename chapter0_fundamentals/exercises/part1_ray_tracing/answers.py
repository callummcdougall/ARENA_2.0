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
#%%

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
    interval = 2*y_limit/(num_pixels-1)
    rays = t.zeros((num_pixels,2,3))
    #y_values = t.arange(-y_limit,y_limit+interval/2, interval)
    t.arange(-y_limit,y_limit+interval/2, interval, out=rays[:,1,1])
    #rays[:,1,1] = y_values
    rays[:,1,0] = t.ones(num_pixels)
    return rays


rays1d = make_rays_1d(9, 10.0)

if MAIN:
    fig = render_lines_with_plotly(rays1d)

#%%
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
#%%

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    O = ray[0,:2]
    D = ray[1,:2]
    L_1 = segment[0,:2]
    L_2 = segment[1,:2]
    A = t.stack((D, L_1-L_2), dim=1)
    if t.det(A) == 0:
        return False
    B = L_1 - O
    solution = t.linalg.solve(A,B)
    u = solution[0]
    v = solution[1]
    return True if u >= 0 and v >= 0 and v <= 1 else False

    # didn't check whether the line segment lies on any of the rays but cba

if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

#%%
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard
from torch import Tensor

@jaxtyped
@typeguard.typechecked
def my_concat(x: Float[Tensor, "a1 b"], y: Float[Tensor, "a2 b"]) -> Float[Tensor, "a1+a2 b"]:
    return t.concat([x, y], dim=0)

x = t.ones(3, 2)
y = t.randn(4, 2)
z = my_concat(x, y)
#%%
def intersect_rays_1d(rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    r = rays.shape[0]
    s = segments.shape[0]
    O = rays[:,0,:2] #r, 2
    D = rays[:,1,:2] #r, 2
    D = einops.repeat(D, 'a b -> c a b', c=s)
    O = einops.repeat(O, 'a b -> c a b', c=s)
    L_1 = segments[:,0,:2] #s, 2
    L_1 = einops.repeat(L_1, 'a b -> a c b', c=r)
    L_2 = segments[:,1,:2] #s, 2
    L_2 = einops.repeat(L_2, 'a b -> a c b', c=r)
    A = t.stack((D, L_1-L_2), dim=-1) #s,r,2,2
    B = L_1 - O #s,r,2
    singular = (t.linalg.det(A).abs()  < 1e-6) #s,r
    A[singular] = t.eye(2)
    solutions = t.linalg.solve(A,B) #s,r,2
    intersections = (solutions[...,0] >=0) & (solutions[...,1]>=0) & (solutions[...,1]<=1)
    intersections[singular] = False
    return intersections.any(dim=0)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)
#%%
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    num_rays = num_pixels_y * num_pixels_z
    rays = t.zeros(num_rays, 2, 3)
    rays[:,1,0] = 1
    y_interval = 2*y_limit/(num_pixels_y-1)
    z_interval = 2*z_limit/(num_pixels_z-1)
    y_values = t.arange(-y_limit,y_limit+y_interval/2, y_interval)
    z_values = t.arange(-z_limit,z_limit+z_interval/2, z_interval)
    y = einops.repeat(y_values, 'a -> (b a)', b=num_pixels_z)
    z = einops.repeat(z_values, 'a -> (a b)', b=num_pixels_y)
    rays[:,1,1] = y
    rays[:,1,2] = z
    return rays

if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)
#%%

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

    left = t.stack((-D, B-A, C-A),dim=-1)
    right = O-A
    if t.linalg.det(left).abs() < 1e-6:
        return False
    s, u, v = t.linalg.solve(left, right)
    return True if u >=0 and v >=0 and u+v <= 1 else False


if MAIN:
    tests.test_triangle_ray_intersects(triangle_ray_intersects)

#%%

def raytrace_triangle(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    nrays = rays.shape[0]
    O, D = rays.unbind(1) #nrays, 3
    A, B, C = triangle #3
    C_1 = -D
    C_2 = einops.repeat(B-A, 'a -> b a', b=nrays)
    C_3 = einops.repeat(C-A, 'a -> b a', b=nrays)
    R = O - A #should broadcast to nrays, 3
    L = t.stack((C_1,C_2,C_3), dim=-1)
    _, u, v = t.linalg.solve(L,R).unbind(dim=-1) #nrays
    intersections = (u>=0) & (v>=0) & (u+v<=1)
    return intersections


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

#%%

if MAIN:
    with open(section_dir / "pikachu.pt", "rb") as f:
        triangles = t.load(f)

#%%

def raytrace_mesh(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangles: Float[Tensor, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[Tensor, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''
    r = rays.shape[0]
    print(f'{r=}')
    nt = triangles.shape[0]
    print(f'{nt=}')
    rays = einops.repeat(rays, 'a b c -> d a b c', d=nt) #t r 2 3
    triangles = einops.repeat(triangles, 'a b c -> a d b c', d=r) #t r 3 3
    O, D = rays.unbind(-2) #t, r, 3
    A, B, C = triangles.unbind(-2) #t, r, 3
    C_1 = -D
    C_2 = B-A
    C_3 = C-A
    left = t.stack((C_1,C_2,C_3),dim=-1) #t, r, 3, 3
    right = O-A #t, r, 3
    s, u, v = t.linalg.solve(left,right).unbind(-1) #t, r
    intersections = (u>=0) & (v>=0) & (u+v<=1) #t, r
    check = t.any(intersections,dim=0) #check which rays have any intersection
    distances = s*intersections #elementwise bool mult, keep the distances to triangles that intersect
    distances[:,~check] = float('inf')
    min_distance = t.min(distances,dim=0).values
    return min_distance


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
