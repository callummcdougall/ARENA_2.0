# %%

import os
import torch as t
import einops
import matplotlib.pyplot as plt
from ipywidgets import interact
import plotly.express as px
import ipywidgets as wg
import plotly.graph_objs as go
from IPython.display import display
from torchtyping import TensorType as TT

import part1_raytracing_tests as tests

MAIN = __name__ == "__main__"

if "SKIP":
    """Teacher only - generate a nicely transformed Pikachu from the STL.
    This reduces messing around with 3D transformations, which isn't the learning objective today.
    """
    from stl import mesh

    model = mesh.Mesh.from_file("pikachu.stl")
    triangles = t.tensor(model.vectors.copy())
    mesh_center = triangles.mean(dim=(0, 1))
    triangles -= mesh_center  # Shift to origin
    # Rotate standing up (isn't the cutest pose but good enough)
    R = t.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    triangles = t.einsum("ij,...i->...j", R, triangles)
    # Scale down so they can use limits of 1
    triangles /= 20.0
    with open("pikachu.pt", "wb") as f:
        t.save(triangles, f)

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
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays


def render_lines_with_plotly(lines: t.Tensor, bold_lines: t.Tensor = t.Tensor()):
    '''
    Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).

    bold_lines: same shape as lines. If supplied, these lines will be rendered in black on top of the other lines.
    '''
    fig = go.Figure(layout=dict(showlegend=False, title="3D rays"))
    for line in lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines")
    for line in bold_lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines", line_width=5, line_color="black")
    fig.show()


rays1d = make_rays_1d(9, 10.0)
if MAIN:
    fig = render_lines_with_plotly(rays1d)


# %%


if MAIN:
    v = wg.FloatSlider(min=-2, max=2, step=0.01, value=0.5, description="v")
    seed = wg.IntSlider(min=0, max=10, step=1, value=0, description="random seed")

    fig = go.FigureWidget(go.Scatter(x=[], y=[]))
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12)
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
    fig.update_layout(showlegend=False, xaxis_range=[-1.5, 2.5], yaxis_range=[-1.5, 2.5])

    def response(change):
        t.manual_seed(seed.value)
        L_1, L_2 = t.rand(2, 2)
        P = lambda v: L_1 + v * (L_2 - L_1)
        x, y = zip(P(-2), P(2))
        with fig.batch_update(): 
            fig.data[0].update({"x": x, "y": y}) 
            fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
            fig.data[2].update({"x": [P(v.value)[0]], "y": [P(v.value)[1]]}) 
        
    v.observe(response)
    seed.observe(response)
    response("")

    box = wg.VBox([v, seed, fig])
    display(box)



# %%

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])
if MAIN:
    for i, ((L1x, L1y, _), (L2x, L2y, _)) in enumerate(segments):
        intersects = []
        for j, ((Ox, Oy, _), (Dx, Dy, _)) in enumerate(rays1d):
            mat = t.tensor(
                [[Dx, L1x - L2x], 
                [Dy, L1y - L2y]
            ])
            vec = t.tensor([L1x - Ox, L1y - Oy])
            try:
                u, v = t.linalg.solve(mat, vec)
                if u >= 0 and 0 <= v <= 1:
                    intersects.append(j)
            except:
                pass
        print(f"Segment {i+1} intersects {tuple(intersects) if intersects else 'none'}")


if MAIN:
    render_lines_with_plotly(rays1d, segments)

# %%

def intersect_ray_1d(ray: t.Tensor, segment: t.Tensor) -> bool:
    '''
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
    # Get the x and y coordinates (ignore z)
    ray = ray[..., :2]
    segment = segment[..., :2]

    # Ray is [[Ox, Oy], [Dx, Dy]]
    O, D = ray
    # Segment is [[L1x, L1y], [L2x, L2y]
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


if MAIN:
    tests.test_intersect_ray_1d(intersect_ray_1d)
    tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

# %%

def intersect_rays_1d(rays: TT["nrays", 2, 3], segments: TT["nsegments", 2, 3]) -> TT["nrays", bool]:
    '''
    For each ray, return True if it intersects any segment.
    '''
    NR = rays.size(0)
    NS = segments.size(0)

    # Get just the x and y coordinates
    rays = rays[..., :2]
    segments = segments[..., :2]

    # Repeat rays and segments so that we can compuate the intersection of every (ray, segment) pair
    rays = einops.repeat(rays, "nrays p d -> nrays nsegments p d", nsegments=NS)
    segments = einops.repeat(segments, "nsegments p d -> nrays nsegments p d", nrays=NR)

    # Each element of `rays` is [[Ox, Oy], [Dx, Dy]]
    O = rays[:, :, 0]
    D = rays[:, :, 1]
    assert O.shape == (NR, NS, 2)

    # Each element of `segments` is [[L1x, L1y], [L2x, L2y]]
    L_1 = segments[:, :, 0]
    L_2 = segments[:, :, 1]
    assert L_1.shape == (NR, NS, 2)

    # Define matrix on left hand side of equation
    mat = t.stack([D, L_1 - L_2], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    assert is_singular.shape == (NR, NS)
    mat[is_singular] = t.eye(2)

    # Define vector on the right hand side of equation
    vec = L_1 - O

    # Solve equation, get results
    sol = t.linalg.solve(mat, vec)
    u = sol[..., 0]
    v = sol[..., 1]

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (v <= 1) & ~is_singular).any(dim=-1)


if MAIN:
    tests.test_intersect_rays_1d(intersect_rays_1d)
    tests.test_intersect_rays_1d_special_case(intersect_rays_1d)


# %%

def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension

    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.

    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''
    n_pixels = num_pixels_y * num_pixels_z
    ygrid = t.linspace(-y_limit, y_limit, num_pixels_y)
    zgrid = t.linspace(-z_limit, z_limit, num_pixels_z)
    rays = t.zeros((n_pixels, 2, 3), dtype=t.float32)
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = einops.repeat(ygrid, "y -> (y z)", z=num_pixels_z)
    rays[:, 1, 2] = einops.repeat(zgrid, "z -> (y z)", y=num_pixels_y)
    return rays


if MAIN:
    rays_2d = make_rays_2d(10, 10, 0.3, 0.3)
    render_lines_with_plotly(rays_2d)

# %%


if MAIN:
    u = wg.FloatSlider(min=-0.5, max=1.5, step=0.01, value=0, description="u")
    v = wg.FloatSlider(min=-0.5, max=1.5, step=0.01, value=0, description="v")

    one_triangle = t.tensor([[0, 0, 0], [3, 0.5, 0], [2, 3, 0]])
    A, B, C = one_triangle
    x, y, z = one_triangle.T

    fig = go.FigureWidget(
        data=[
            go.Scatter(x=x, y=y, mode="markers+text", text=["A", "B", "C"], textposition="middle left", textfont_size=18, marker_size=12),
            go.Scatter(x=[*x, x[0]], y=[*y, y[0]], mode="lines"),
            go.Scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
        ],
        layout=dict(
            title="Barycentric coordinates illustration", showlegend=False,
            xaxis_range=[-3, 8], yaxis_range=[-2, 5.5],
        )
    )

    def response(change):
        P = A + u.value * (B - A) + v.value * (C - A)
        fig.data[2].update({"x": [P[0]], "y": [P[1]]})
        
    u.observe(response)
    v.observe(response)
    response("")

    box = wg.VBox([u, v, fig])
    display(box)


# %%


def triangle_line_intersects(A: t.Tensor, B: t.Tensor, C: t.Tensor, O: t.Tensor, D: t.Tensor) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    '''
    mat = t.stack([D, B - A, C - A], dim=-1)
    det = t.linalg.det(mat)
    if det.abs() < 1e-8:
        return False
    s, u, v = t.linalg.solve(mat, O - A)
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()


if MAIN:
    tests.test_triangle_line_intersects(triangle_line_intersects)

# %%


def raytrace_triangle(
    rays: t.Tensor, #TT["nrays", "points": 2, "ndims": 3], 
    triangle: t.Tensor, #TT["npoints": 3, "ndims": 3]
) -> t.Tensor: #TT["nrays", bool]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    assert isinstance(rays, t.Tensor)
    assert isinstance(triangle, t.Tensor)

    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "p d -> p nrays d", nrays=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O = rays[:, 0]
    D = rays[:, 1]
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.T

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


if MAIN:
    A = t.tensor([1, 0.0, -0.5])
    B = t.tensor([1, -0.5, 0.0])
    C = t.tensor([1, 0.5, 0.5])
    num_pixels_y = num_pixels_z = 15
    y_limit = z_limit = 0.5

    test_triangle = t.stack([A, B, C], dim=0)
    rays2d = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    triangle_lines = t.stack([A, B, B, C, C, A], dim=0).reshape(-1, 2, 3)
    render_lines_with_plotly(rays2d, triangle_lines)
    intersects = raytrace_triangle(rays2d, test_triangle)
    img = intersects.reshape(num_pixels_y, num_pixels_z)
    px.imshow(img, origin="lower", labels={"x": "X", "y": "Y"}).update_layout(coloraxis_showscale=False).show()

# %%

if MAIN:
    '''
    Teacher only - generate a nicely transformed Pikachu from the STL.

    This reduces messing around with 3D transformations, which isn't the learning objective today.
    '''
    from stl import mesh

    model = mesh.Mesh.from_file("pikachu.stl")
    triangles = t.tensor(model.vectors.copy())
    mesh_center = triangles.mean(dim=(0, 1))
    triangles -= mesh_center  # Shift to origin
    # Rotate standing up (isn't the cutest pose but good enough)
    R = t.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    triangles = t.einsum("ij,...i->...j", R, triangles)
    # Scale down so they can use limits of 1
    triangles /= 20.0
    with open("pikachu.pt", "wb") as f:
        t.save(triangles, f)

# %%

if MAIN:
    with open("pikachu.pt", "rb") as f:
        triangles = t.load(f)

# %%

def raytrace_mesh(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangles: TT["ntriangles", "npoints": 3, "ndims": 3]
) -> TT["nrays", bool]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT p d -> p NR NT d", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR p d -> p NR NT d", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns (note, s is the distance along ray)
    sol = t.linalg.solve(mat, vec)
    s, u, v = einops.rearrange(sol, "NR NT d -> d NR NT")

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values


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

    fig = px.imshow(img, facet_col=0, origin="lower", color_continuous_scale="magma")
    fig.update_layout(coloraxis_showscale=False)
    for i, text in enumerate(["Intersects", "Distance"]): 
        fig.layout.annotations[i]['text'] = text
    fig.show()

# %%

from typing import Callable
from tqdm import tqdm

def raytrace_mesh_video(
    rays: TT["nrays", "points": 2, "ndims": 3], 
    triangles: TT["ntriangles", "npoints": 3, "ndims": 3],
    rotation_matrix: Callable[[float], TT[3, 3]],
    num_frames: int,
) -> TT["nframes", "nrays", bool]:
    result = []
    theta = t.tensor(2 * t.pi) / num_frames
    R = rotation_matrix(theta)
    for theta in tqdm(range(num_frames)):
        triangles = triangles @ R
        result.append(raytrace_mesh(rays, triangles))
    return t.stack(result, dim=0)

if MAIN:
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1
    num_frames = 50

    rotation_matrix = lambda theta: t.tensor([
        [t.cos(theta), 0.0, t.sin(theta)],
        [0.0, 1.0, 0.0],
        [-t.sin(theta), 0.0, t.cos(theta)],
    ])

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0] = t.tensor([-2, 0.0, 0.0])
    dists = raytrace_mesh_video(rays, triangles, rotation_matrix, num_frames)
    dists_square = dists.view(num_frames, num_pixels_y, num_pixels_z)

    fig = px.imshow(dists_square, animation_frame=0, origin="lower", color_continuous_scale="viridis_r")
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

# %%

if MAIN:
    fig = px.imshow(dists_square, animation_frame=0, origin="lower", zmin=0, zmax=2, color_continuous_scale="Brwnyl")
    fig.update_layout(coloraxis_showscale=False)
    fig.show()

# %%


if MAIN:
    '''
    Bonus Solution: rendering on GPU
    Thanks to Edgar Lin and Sam Eisenstat
    '''
    from einops import repeat, reduce, rearrange

    def make_rays_2d_origin(
        num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float, origin: t.Tensor
    ) -> t.Tensor:
        '''
        num_pixels_y: The number of pixels in the y dimension
        num_pixels_z: The number of pixels in the z dimension
        y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
        z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.
        Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
        '''
        rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
        rays[:, :, 1, 0] = 1
        rays[:, :, 1, 1] = repeat(
            t.arange(num_pixels_y) * 2.0 * y_limit / (num_pixels_y - 1) - y_limit, "y -> y z", z=num_pixels_z
        )
        rays[:, :, 1, 2] = repeat(
            t.arange(num_pixels_z) * 2.0 * z_limit / (num_pixels_z - 1) - z_limit, "z -> y z", y=num_pixels_y
        )
        rays[:, :, 0, :] = origin
        return rearrange(rays, "y z n d -> (y z) n d", n=2, d=3)

    def raytrace_mesh_gpu(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
        '''For each ray, return the distance to the closest intersecting triangle, or infinity.
        triangles: shape (n_triangles, n_points=3, n_dims=3)
        rays: shape (n_pixels, n_points=2, n_dims=3)
        return: shape (n_pixels, )
        '''
        n_triangles = triangles.size(0)
        n_pixels = rays.size(0)
        device = "cuda"
        matrices = t.zeros((n_pixels, n_triangles, 3, 3)).to(device)
        rays_gpu = rays.to(device)
        matrices[:, :, :, 0] = repeat(rays_gpu[:, 0] - rays_gpu[:, 1], "r d -> r t d", t=n_triangles)
        triangles_gpu = triangles.to(device)
        matrices[:, :, :, 1] = repeat(triangles_gpu[:, 1] - triangles_gpu[:, 0], "t d -> r t d", r=n_pixels)
        matrices[:, :, :, 2] = repeat(triangles_gpu[:, 2] - triangles_gpu[:, 0], "t d -> r t d", r=n_pixels)
        bs = repeat(rays_gpu[:, 0], "r d -> r t d", t=n_triangles) - repeat(
            triangles_gpu[:, 0], "t d -> r t d", r=n_pixels
        )
        mask = t.linalg.det(matrices) != 0
        distances = t.full((n_pixels, n_triangles), float("inf")).to(device)
        solns = t.linalg.solve(matrices[mask], bs[mask])
        distances[mask] = t.where(
            (solns[:, 0] >= 0) & (solns[:, 1] >= 0) & (solns[:, 2] >= 0) & (solns[:, 1] + solns[:, 2] <= 1),
            solns[:, 0],
            t.tensor(float("inf")).to(device),
        )
        return reduce(distances, "r t -> r", "min").to("cpu")

    num_pixels_y = 120
    num_pixels_z = 120
    y_limit = z_limit = 3
    rays = make_rays_2d_origin(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-3.0, 0, 0]))
    intersections = raytrace_mesh_gpu(triangles, rays)
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    
    fig = px.imshow(picture, origin="lower").update_layout(coloraxis_showscale=False)
    fig.show()

# %%
'''
Bonus Solution: Lighting
Thanks to Edgar Lin and Sam Eisenstat
'''
import math

def raytrace_mesh_lighting(
    triangles: t.Tensor, rays: t.Tensor, light: t.Tensor, ambient_intensity: float, device: str = "cpu"
) -> t.Tensor:
    '''For each ray, return the shade of the nearest triangle.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    light: shape (n_dims=3, )
    device: The device to place tensors on.
    return: shape (n_pixels, )
    '''
    n_triangles = triangles.size(0)
    n_pixels = rays.size(0)
    triangles = triangles.to(device)
    rays = rays.to(device)
    light = light.to(device)

    matrices = t.zeros((n_pixels, n_triangles, 3, 3)).to(device)
    directions = rays[:, 1] - rays[:, 0]
    matrices[:, :, :, 0] = repeat(-directions, "r d -> r t d", t=n_triangles)
    matrices[:, :, :, 1] = repeat(triangles[:, 1] - triangles[:, 0], "t d -> r t d", r=n_pixels)
    matrices[:, :, :, 2] = repeat(triangles[:, 2] - triangles[:, 0], "t d -> r t d", r=n_pixels)
    bs = repeat(rays[:, 0], "r d -> r t d", t=n_triangles) - repeat(triangles[:, 0], "t d -> r t d", r=n_pixels)
    mask = t.linalg.det(matrices) != 0
    distances = t.full((n_pixels, n_triangles), float("inf")).to(device)
    solns = t.linalg.solve(matrices[mask], bs[mask])
    distances[mask] = t.where(
        (solns[:, 0] >= 0) & (solns[:, 1] >= 0) & (solns[:, 2] >= 0) & (solns[:, 1] + solns[:, 2] <= 1),
        solns[:, 0],
        t.tensor(float("inf")).to(device),
    )
    closest_triangle = distances.argmin(1)

    normals = t.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=1)
    normals = t.nn.functional.normalize(normals, p=2.0, dim=1)
    intensity = t.einsum("td,d->t", normals, light).gather(0, closest_triangle)
    side = t.einsum("rd,rd->r", normals.gather(0, repeat(closest_triangle, "r -> r d", d=3)), directions)
    intensity = t.maximum(t.sign(side) * intensity, t.zeros(())) + ambient_intensity
    intensity = t.where(
        distances.gather(1, closest_triangle.unsqueeze(1)).squeeze(1) == float("inf"),
        t.tensor(0.0).to(device),
        intensity,
    )

    return intensity.to("cpu")

def make_rays_camera(
    num_pixels_v: int,
    num_pixels_w: int,
    v_limit: float,
    w_limit: float,
    origin: t.Tensor,
    screen_distance: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> t.Tensor:
    '''
    num_pixels_y: The number of pixels in the y dimension
    num_pixels_z: The number of pixels in the z dimension
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both.
    z_limit: At x=1, the rays should extend from -z_limit to +z_limit, inclusive of both.
    Returns: shape (num_rays=num_pixels_y * num_pixels_z, num_points=2, num_dims=3).
    '''

    normal = t.tensor([math.cos(pitch) * math.cos(yaw), math.sin(pitch), math.cos(pitch) * math.sin(yaw)])
    w_vec = t.nn.functional.normalize(t.tensor([normal[2], 0, -normal[0]]), p=2.0, dim=0)
    v_vec = t.cross(normal, w_vec)
    w_vec_r = math.cos(roll) * w_vec + math.sin(roll) * v_vec
    v_vec_r = math.cos(roll) * v_vec - math.sin(roll) * w_vec

    rays = t.zeros((num_pixels_y, num_pixels_z, 2, 3))
    rays[:, :, 1, :] += repeat(origin + normal * screen_distance, "d -> w v d", w=num_pixels_w, v=num_pixels_v)
    rays[:, :, 1, :] += repeat(
        t.einsum("w, d -> w d", (t.arange(num_pixels_w) * 2.0 * w_limit / (num_pixels_w - 1) - w_limit), w_vec_r),
        "w d -> w v d",
        v=num_pixels_v,
    )
    rays[:, :, 1, :] += repeat(
        t.einsum("v, d -> v d", t.arange(num_pixels_v) * 2.0 * v_limit / (num_pixels_v - 1) - v_limit, v_vec_r),
        "v d -> w v d",
        w=num_pixels_w,
    )

    rays[:, :, 0, :] = origin
    return rearrange(rays, "y z n d -> (y z) n d", n=2, d=3)

# %%

if MAIN:
    num_pixels_y = 150
    num_pixels_z = 150
    y_limit = z_limit = 3
    rays = make_rays_2d_origin(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-3.0, 0, 0]))
    light = t.tensor([0.0, -1.0, 1.0])
    ambient_intensity = 0.5
    intersections = raytrace_mesh_lighting(triangles, rays, light, ambient_intensity, "cuda")
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    
    fig = px.imshow(picture, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()

# %%

def get_random_rotation_matrix(N, theta_max=t.pi):
    mat = t.eye(N)
    for i in range(N):
        rot_mat = t.eye(N)
        theta = (t.rand(1) - 0.5) * theta_max
        rot_mat_2d = t.tensor([
            [t.cos(theta), -t.sin(theta)], 
            [t.sin(theta), t.cos(theta)]
        ])
        if i == N - 1:
            rot_mat[[-1, -1, 0, 0], [-1, 0, -1, 0]] = rot_mat_2d.flatten()
        else:
            rot_mat[i :i+2, i :i+2] = rot_mat_2d
        mat = mat @ rot_mat
    return mat

if MAIN:
    num_pixels_y = 150
    num_pixels_z = 150
    y_limit = z_limit = 3

    rays = make_rays_camera(num_pixels_y, num_pixels_z, y_limit, z_limit, t.tensor([-1.0, 3.0, 0.0]), 3.0, 0.0, -1.0, 0.0)
    light = t.tensor([0.0, -1.0, 1.0])
    ambient_intensity = 0.5
    intersections = raytrace_mesh_lighting(triangles, rays, light, ambient_intensity, "cuda")
    picture = rearrange(intersections, "(y z) -> y z", y=num_pixels_y, z=num_pixels_z)
    fig = px.imshow(picture, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()

# %%
'''
Bonus solution: Lighting using Lambert shading
Thanks to Jordan Taylor and Alexander Mont
'''

def raytrace_mesh_lambert(triangles: t.Tensor, rays: t.Tensor) -> t.Tensor:
    '''For each ray, return the distance to the closest intersecting triangle, or infinity.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    return: shape (n_pixels, )
    '''
    # triangles = [triangle, point, coord]
    # rays = [pixel, orig_dir, coord]

    n_triangles = len(triangles)
    n_pixels = len(rays)

    rep_triangles = einops.repeat(triangles, "triangle point coord -> pixel triangle point coord", pixel=n_pixels)
    rep_rays = einops.repeat(rays, "pixel orig_dir coord -> pixel triangle orig_dir coord", triangle=n_triangles)

    O = rep_rays[:, :, 0, :]  # [pixel, triangle, coord]
    D = rep_rays[:, :, 1, :]  # [pixel, triangle, coord]
    A = rep_triangles[:, :, 0, :]  # [pixel, triangle, coord]
    B = rep_triangles[:, :, 1, :]  # [pixel, triangle, coord]
    C = rep_triangles[:, :, 2, :]  # [pixel, triangle, coord]
    rhs = O - A  # [pixel, triangle, coord]
    lhs = t.stack([-D, B - A, C - A], dim=3)  # [pixel, triangle, coord, suv]
    dets = t.linalg.det(lhs)  # [pixel, triangle]
    dets = dets < 1e-5
    eyes = t.einsum("i j , k l -> i j k l", [dets, t.eye(3)])
    lhs += eyes
    results = t.linalg.solve(lhs, rhs)  # [pixel, triangle, suv]
    intersects = (
        ((results[:, :, 1] + results[:, :, 2]) <= 1)
        & (results[:, :, 0] >= 0)
        & (results[:, :, 1] >= 0)
        & (results[:, :, 2] >= 0)
        & (dets == False)
    )  # [pixel, triangle]
    distances = t.where(intersects, results[:, :, 0].double(), t.inf)  # [pixel, triangle]

    # Lambert shading (dot product of triangle's normal vector with light direction)
    indices = t.argmin(distances, dim=1)
    tri_vecs1 = triangles[:, 0, :] - triangles[:, 1, :]
    tri_vecs2 = triangles[:, 1, :] - triangles[:, 2, :]
    normvecs = t.cross(tri_vecs1, tri_vecs2, dim=1)  # [triangle coord]
    normvecs -= normvecs.min(1, keepdim=True)[0]
    normvecs /= normvecs.max(1, keepdim=True)[0]
    lightvec = t.tensor([[0.0, 1.0, 1.0]] * n_triangles)
    tri_lights = abs(t.einsum("t c , t c -> t", [normvecs, lightvec]))  # triangle
    pixel_lights = 1.0 / (einops.reduce(distances, "pixel triangle -> pixel", "min")) ** 2
    pixel_lights *= tri_lights[indices]
    return pixel_lights



if MAIN:
    rot_mat = get_random_rotation_matrix(N=3, theta_max=t.pi/3)
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1
    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0, 0] = -2
    rays[0, :, 0]
    result = raytrace_mesh_lambert(t.einsum("i j k, k l -> i j l", [triangles, rot_mat]), rays)
    result = result.reshape(num_pixels_y, num_pixels_z)
    fig = px.imshow(result, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()

# %%


# Solution with hollow triangles


def raytrace_mesh_lambert_wireframe(triangles: t.Tensor, rays: t.Tensor, triangle_perim: float = 0) -> t.Tensor:
    '''For each ray, return the distance to the closest intersecting triangle, or infinity.
    triangles: shape (n_triangles, n_points=3, n_dims=3)
    rays: shape (n_pixels, n_points=2, n_dims=3)
    return: shape (n_pixels, )
    '''
    # triangles = [triangle, point, coord]
    # rays = [pixel, orig_dir, coord]

    n_triangles = len(triangles)
    n_pixels = len(rays)

    rep_triangles = einops.repeat(triangles, "triangle point coord -> pixel triangle point coord", pixel=n_pixels)
    rep_rays = einops.repeat(rays, "pixel orig_dir coord -> pixel triangle orig_dir coord", triangle=n_triangles)

    O = rep_rays[:, :, 0, :]  # [pixel, triangle, coord]
    D = rep_rays[:, :, 1, :]  # [pixel, triangle, coord]
    A = rep_triangles[:, :, 0, :]  # [pixel, triangle, coord]
    B = rep_triangles[:, :, 1, :]  # [pixel, triangle, coord]
    C = rep_triangles[:, :, 2, :]  # [pixel, triangle, coord]
    rhs = O - A  # [pixel, triangle, coord]
    lhs = t.stack([-D, B - A, C - A], dim=3)  # [pixel, triangle, coord, suv]
    dets = t.linalg.det(lhs)  # [pixel, triangle]
    dets = dets < 1e-5
    eyes = t.einsum("i j , k l -> i j k l", [dets, t.eye(3)])
    lhs += eyes
    results = t.linalg.solve(lhs, rhs)  # [pixel, triangle, suv]
    intersects = (
        ((results[:, :, 1] + results[:, :, 2]) <= 1)
        & (results[:, :, 0] >= 0.0)
        & (results[:, :, 1] >= 0.0)
        & (results[:, :, 2] >= 0.0)
        & (dets == False)
    )  # [pixel, triangle]
    intersects_perim = (
        ((results[:, :, 1] + results[:, :, 2]) >= 1 - triangle_perim)
        | (results[:, :, 1] <= triangle_perim)
        | (results[:, :, 2] <= triangle_perim)
    )
    intersects = intersects & intersects_perim
    distances = t.where(intersects, results[:, :, 0].double(), t.inf)  # [pixel, triangle]

    # Lambert shading (dot product of triangle's normal vector with light direction)
    indices = t.argmin(distances, dim=1)
    tri_vecs1 = triangles[:, 0, :] - triangles[:, 1, :]
    tri_vecs2 = triangles[:, 1, :] - triangles[:, 2, :]
    normvecs = t.cross(tri_vecs1, tri_vecs2, dim=1)  # [triangle coord]
    normvecs -= normvecs.min(1, keepdim=True)[0]
    normvecs /= normvecs.max(1, keepdim=True)[0]
    lightvec = t.tensor([[0.0, 1.0, 1.0]] * n_triangles)
    tri_lights = abs(t.einsum("t c , t c -> t", [normvecs, lightvec]))  # triangle
    pixel_lights = 1.0 / (einops.reduce(distances, "pixel triangle -> pixel", "min")) ** 2
    pixel_lights *= tri_lights[indices]
    return pixel_lights

if MAIN:
    rot_mat = get_random_rotation_matrix(N=3, theta_max=t.pi/4)
    num_pixels_y = 200
    num_pixels_z = 200
    y_limit = z_limit = 1

    triangle_perim = 0.1

    rays = make_rays_2d(num_pixels_y, num_pixels_z, y_limit, z_limit)
    rays[:, 0, 0] = -2
    rays[0, :, 0]
    result = raytrace_mesh_lambert_wireframe(t.einsum("i j k, k l -> i j l", [triangles, rot_mat]), rays, triangle_perim)
    result = result.reshape(num_pixels_y, num_pixels_z)
    fig = px.imshow(result, origin="lower", labels={"x": "X", "y": "Y"}, color_continuous_scale="magma").update_layout(coloraxis_showscale=False)
    fig.show()

# %%
