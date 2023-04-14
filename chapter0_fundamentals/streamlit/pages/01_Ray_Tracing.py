
import streamlit as st
import st_dependencies
st_dependencies.styling()

st.sidebar.markdown(r"""

## Table of Contents

<ul class="contents">
    <li class='margtop'><a class='contents-el' href='#setup'>Setup</a></li>
    <li class='margtop'><a class='contents-el' href='#1d-image-rendering'>1D Image Rendering</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#tip-the-<code>out</code>-keyword-argument'>Tip - the <code>out</code> keyword argument</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#ray-object-intersection'>Ray-Object Intersection</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-<code>intersect-ray-1d</code>'>Exercise - implement <code>intersect_ray_1d</code></a></li>
        <li><a class='contents-el' href='#aside-typechecking'>Aside - typechecking</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#batched-ray-segment-intersection'>Batched Ray-Segment Intersection</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#tip-ellipsis'>Tip - Ellipsis</a></li>
        <li><a class='contents-el' href='#tip-elementwise-logical-operations-on-tensors'>Tip - Elementwise Logical Operations on Tensors</a></li>
        <li><a class='contents-el' href='#tip-<code>einops</code>'>Tip - <code>einops</code></a></li>
        <li><a class='contents-el' href='#tip-logical-reductions'>Tip - Logical Reductions</a></li>
        <li><a class='contents-el' href='#tip-broadcasting'>Tip - Broadcasting</a></li>
        <li><a class='contents-el' href='#summary-of-all-these-tips'>Summary of all these tips</a></li>
        <li><a class='contents-el' href='#exercise-implement-<code>intersect-rays-1d</code>'>Exercise - implement <code>intersect_rays_1d</code></a></li>
        <li><a class='contents-el' href='#using-gpt-to-understand-code'>Using GPT to understand code</a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#2d-rays'>2D Rays</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-<code>make-rays-2d</code>'>Exercise - implement <code>make_rays_2d</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#triangle-coordinates'>Triangle Coordinates</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#triangle-ray-intersection'>Triangle-Ray Intersection</a></li>
        <li><a class='contents-el' href='#exercise-implement-<code>triangle-line-intersects</code>'>Exercise - implement <code>triangle_line_intersects</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#single-triangle-rendering'>Single-Triangle Rendering</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#views-and-copies'>Views and Copies</a></li>
        <li><a class='contents-el' href='#storage-objects'>Storage Objects</a></li>
        <li><a class='contents-el' href='#<code>tensor.-base</code>'><code>Tensor._base</code></a></li>
        <li><a class='contents-el' href='#exercise-implement-<code>raytrace-triangle</code>'>Exercise - implement <code>raytrace_triangle</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#mesh-loading'>Mesh Loading</a></li>
    <li class='margtop'><a class='contents-el' href='#mesh-rendering'>Mesh Rendering</a></li>
    <li><ul class="contents">
        <li><a class='contents-el' href='#exercise-implement-<code>raytrace-mesh</code>'>Exercise - implement <code>raytrace_mesh</code></a></li>
    </ul></li>
    <li class='margtop'><a class='contents-el' href='#bonus-content'>Bonus Content</a></li>
</ul></li>""", unsafe_allow_html=True)

st.markdown(r"""

<img src="https://raw.githubusercontent.com/callummcdougall/Fundamentals/main/images/raytracing.png" width="350">


If you have any feedback on this course (e.g. bugs, confusing explanations, parts that you feel could be structured better), please let me know using [this Google Form](https://forms.gle/2ZhdHa87wWsrATjh9).


# Day 1 - Ray Tracing

Today we'll be practicing batched matrix operations in PyTorch by writing a basic graphics renderer. We'll start with an extremely simplified case and work up to rendering your very own 3D Pikachu! Note that if you're viewing this file on GitHub, some of the equations may not render properly. Viewing it locally in VS Code should fix this.


## Setup

Run these cells below (don't worry about reading through them).


```python
import torch as t
import os
import einops
from pathlib import Path
import matplotlib.pyplot as plt
from ipywidgets import interact
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as wg
from ipywidgets import interact, interactive, fixed
from IPython.display import display
from jaxtyping import Float, Int, jaxtyped
from typeguard import typechecked as typechecker

import part1_raytracing_tests as tests
# import part1_raytracing_solutions as solutions

```

## 1D Image Rendering

In our initial setup, the **camera** will be a single point at the origin, and the **screen** will be the plane at x=1.

**Objects** in the world consist of triangles, where triangles are represented as 3 points in 3D space (so 9 floating point values per triangle). You can build any shape out of sufficiently many triangles and your Pikachu will be made from 412 triangles.

The camera will emit one or more **rays**, where a ray is represented by an **origin** point and a **direction** point. Conceptually, the ray is emitted from the origin and continues in the given direction until it intersects an object.

We have no concept of lighting or color yet, so for now we'll say that a pixel on our screen should show a bright color if a ray from the origin through it intersects an object, otherwise our screen should be dark.


<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/ray_tracing.png" width="400">


To start, we'll let the z dimension in our `(x, y, z)` space be zero and work in the remaining two dimensions. 

Implement the following `make_rays_1d` function so it generates some rays coming out of the origin, which we'll take to be `(0, 0, 0)`.

Calling `render_lines_with_pyplot` on your rays should look like this (note the orientation of the axes):


```python
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
    fig = go.Figure(layout=dict(showlegend=False, title="3D rays", height=600, width=600))
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

```

<details>
<summary>Solution</summary>


```python
def first_func(num_pixels, y_limit) -> t.Tensor:
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)
    t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    rays[:, 1, 0] = 1
    return rays

segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])

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
    fig = go.Figure(layout=dict(showlegend=False, title="3D rays", height=600, width=600))
    for line in lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines")
    for line in bold_lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines", line_width=5, line_color="black")
    fig.show()


rays1d = make_rays_1d(9, 10.0)
```
</details>


### Tip - the `out` keyword argument

Many PyTorch functions take an optional keyword argument `out`. If provided, instead of allocating a new tensor and returning that, the output is written directly to the `out` tensor.

If you used `torch.arange` or `torch.linspace` above, try using the `out` argument. Note that a basic indexing expression like `rays[:, 1, 1]` returns a view that shares storage with `rays`, so writing to the view will modify `rays`. You'll learn more about views later today.

## Ray-Object Intersection

Suppose we have a line segment defined by points $L_1$ and $L_2$. Then for a given ray, we can test if the ray intersects the line segment like so:

- Supposing both the ray and line segment were infinitely long, solve for their intersection point.
- If the point exists, check whether that point is inside the line segment and the ray. 

Our camera ray is defined by the origin $O$ and direction $D$ and our object line is defined by points $L_1$ and $L_2$.

We can write the equations for all points on the camera ray as $R(u)=O +u D$ for $u \in [0, \infty)$ and on the object line as $O(v)=L_1+v(L_2 - L_1)$ for $v \in [0, 1]$.

The following interactive widget lets you play with this parameterization of the problem. Run the cells one after another:


```python
if MAIN:
	fig = go.FigureWidget(go.Scatter(x=[], y=[]), layout={"width": 600, "height": 500})
	fig.add_scatter(x=[], y=[], mode="markers", marker_size=12)
	fig.add_scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
	fig.update_layout(showlegend=False, xaxis_range=[-1.5, 2.5], yaxis_range=[-1.5, 2.5], template="simple_white")
	
@interact(seed=(0, 10, 1), v=(-2.0, 2.0, 0.01))
def response(seed=0, v=0.5):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

if MAIN:
	display(fig)

```

Setting the line equations from above equal gives the solution:

$$
\begin{aligned}O + u D &= L_1 + v(L_2 - L_1) \\ u D - v(L_2 - L_1) &= L_1 - O  \\ \begin{pmatrix} D_x & (L_1 - L_2)_x \\ D_y & (L_1 - L_2)_y \\ \end{pmatrix} \begin{pmatrix} u \\ v \\ \end{pmatrix} &=  \begin{pmatrix} (L_1 - O)_x \\ (L_1 - O)_y \\ \end{pmatrix} \end{aligned}
$$

Once we've found values of $u$ and $v$ which satisfy this equation, if any (the lines could be parallel) we just need to check that $u \geq 0$ and $v \in [0, 1]$.


#### Exercise - which segments intersect with the rays?
 
For each of the following segments, which camera rays from earlier intersect? You can do this by inspection or using `render_lines_with_pyplot`.


```python
segments = t.tensor([
    [[1.0, -12.0, 0.0], [1, -6.0, 0.0]], 
    [[0.5, 0.1, 0.0], [0.5, 1.15, 0.0]], 
    [[2, 12.0, 0.0], [2, 21.0, 0.0]]
])
if MAIN:
	render_lines_with_plotly(rays1d, segments)

```

<details>
<summary>Solution - intersecting rays</summary>

```python
render_lines_with_plotly(rays1d, segments)
```

- Segment 0 intersects the first two rays.
- Segment 1 doesn't intersect any rays.
- Segment 2 intersects the last two rays. Computing `rays * 2` projects the rays out to `x=1.5`. Remember that while the plot shows rays as line segments, rays conceptually extend indefinitely.
</details>


### Exercise - implement `intersect_ray_1d`

Using [`torch.lingalg.solve`](https://pytorch.org/docs/stable/generated/torch.linalg.solve.html) and [`torch.stack`](https://pytorch.org/docs/stable/generated/torch.stack.html), implement the `intersect_ray_1d` function to solve the above matrix equation.

<details>
<summary>Aside - difference between stack and concatenate</summary>

`torch.stack` will combine tensors along a new dimension.

```python
>>> t.stack([t.ones(2, 2), t.zeros(2, 2)], dim=0)
tensor([[[1., 1.],
         [1., 1.]],

        [[0., 0.],
         [0., 0.]]])
```

`torch.concat` (alias `torch.cat`) will combine tensors along an existing dimension.

```python
>>> t.cat([t.ones(2, 2), t.zeros(2, 2)], dim=0)
tensor([[1., 1.], 
        [1., 1.],
        [0., 0.],
        [0., 0.]])
```

Here, you should use `torch.stack` to construct e.g. the matrix on the left hand side, because you want to combine the vectors $D$ and $L_1 - L_2$ to make a matrix.
</details>

Is it possible for the solve method to fail? Give a sample input where this would happen.

<details>
<summary>Answer - Failing Solve</summary>

If the ray and segment are exactly parallel, then the solve will fail because there is no solution to the system of equations. For this function, handle this by catching the exception and returning False.
</details>


```python
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


if MAIN:
	tests.test_intersect_ray_1d(intersect_ray_1d)
	tests.test_intersect_ray_1d_special_case(intersect_ray_1d)

```

<details>
<summary>Help! My code is failing with a 'must be batches of square matrices' exception.</summary>

Our formula only uses the x and y coordinates - remember to discard the z coordinate for now. 

It's good practice to write asserts on the shape of things so that your asserts will fail with a helpful error message. In this case, you could assert that the `mat` argument is of shape (2, 2) and the `vec` argument is of shape (2,). Also, see the aside below on typechecking.
</details>

<details>
<summary>Solution</summary>

Note the use of `.item()` at the end. This converts a scalar PyTorch tensor into a Python scalar.
```python
def response(seed=0, v=0.5):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(-2), P(2))
    with fig.batch_update(): 
        fig.data[0].update({"x": x, "y": y}) 
        fig.data[1].update({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}) 
        fig.data[2].update({"x": [P(v)[0]], "y": [P(v)[1]]})

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
```
</details>


### Aside - typechecking

Typechecking is a useful habit to get into. It's not strictly necessary, but it can be a great help when you're debugging.

One good way to typecheck in PyTorch is with the `jaxtyping` library. In this library, we can use objects like `Float`, `Int`, `Bool`, etc object to specify the shape and data type of a tensor (or `Shaped` if we don't care about the data type).

In its simplest form, this just behaves like a fancier version of a docstring or comment (signalling to you, as well as any readers, what the size of objects should be). But you can also use the `typeguard` library to strictly enforce the type signatures of your inputs and outputs. For instance, consider the following typechecked function:


```python
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard
from torch import Tensor as T

@jaxtyped
@typeguard.typechecked
def my_concat(x: Float[T, "a1 b"], y: Float[T, "a2 b"]) -> Float[T, "a1+a2 b"]:
    return t.concat([x, y], dim=0)

if MAIN:
	x = t.ones(3, 2)
	y = t.randn(4, 2)
	z = my_concat(x, y)

```

This cell will run without error, because the tensor `t.concat([x, y], dim=0)` has shape `(3+4, 2) = (7, 2)`, which agrees with the symbolic representation of `(a1 b), (a2 b) -> (a1+a2, b)` in the type signature. But this code will fail if the type signatures are violated in any way, for instance:

* `x` or `y` are not 2D tensors
* The last dimension of `x` and `y` doesn't match
* The output doesn't have shape `(x.shape[0] + y.shape[0], x.shape[1])`
* `x`, `y` or the output are of type int rather than type float

You can test these out for yourself, by changing the cell above and then re-running it.

Jaxtyping has many other useful features, for example:

* Generic tensors can be represented with `Float[T, "..."]`
* Tensors with a single scalar value can be represented with `Float[T, ""]`
* Fixed dimensions can be represented with numbers, e.g. `Float[T, "a b 4"]`
* Dimensions can be named *and* fixed, e.g. `x: Float[T, "b=3"], y: Float[T, "b"]` will raise an error if `x` and `y` don't *both* have shape `(3,)`
* You can even use these objects for inline assert statements, e.g. `assert isinstance(x, Float[T, "3 b"])` asserts that `x` is a 2D tensor of type float, with first dimension `3`.

You can find more features of jaxtyping [here](https://github.com/google/jaxtyping/blob/main/API.md).

Overall, type-checking is a really useful tool to have at your disposal, because it can help you quickly catch bugs in your code (and it helps your code be explicit and readable, both to possible collaborators / pair programming partners, and to your future self!). 

We will give you strictly typechecked functions for today's exercises, but after that point we'll leave the inclusion of strict typechecking up to you (though we strongly recommend it).


Exercise - can you rewrite the function `intersect_ray_1d` to use typechecking?

<details>
<summary>Answer</summary>

Your typechecked function might look like:

```python
@jaxtyped
@typeguard.typechecked
def intersect_ray_1d(ray: Float[T, "points=2 dim=3"], segment: Float[T, "points=2 dim=3"]) -> bool:
    '''
    ray: O, D points
    segment: L_1, L_2 points

    Return True if the ray intersects the segment.
    '''
```

This is if your solution just returned a boolean. If you left your boolean as a single-element PyTorch tensor, then a return type of `Bool[T, ""]` would be appropriate.
</details>


## Batched Ray-Segment Intersection

Next, implement a batched version that takes multiple rays, multiple line segments, and returns a boolean for each ray indicating whether **any** segment intersects with that ray.

Note - in the batched version, we don't want the solver to throw an exception just because some of the equations don't have a solution - these should just return False. 


### Tip - Ellipsis

You can use an ellipsis `...` in an indexing expression to avoid repeated `:` and to write indexing expressions that work on varying numbers of input dimensions. 

For example, `x[..., 0]` is equivalent to `x[:, :, 0]` if `x` is 3D, and equivalent to `x[:, :, :, 0]` if `x` is 4D.

### Tip - Elementwise Logical Operations on Tensors

For regular booleans, the keywords `and`, `or`, and `not` are used to do logical operations and the operators `&`, `|`, and `~` do and, or and not on each bit of the input numbers. For example `0b10001 | 0b11000` is `0b11001` or 25 in base 10.

Tragically, Python doesn't allow classes to overload keywords, so if `x` and `y` are of type `torch.Tensor`, then `x and y` does **not** do the natural thing that you probably expect, which is compute `x[i] and y[i]` elementwise. It actually tries to coerce `x` to a regular boolean, which throws an exception.

As a workaround, PyTorch (and NumPy) have chosen to overload the bitwise operators but have them actually mean logical operations, since you usually don't care to do bitwise operations on tensors. So the correct expression would be `x & y` to compute `x[i] and y[i]` elementwise.

Another gotcha when it comes to logical operations - **operator precedence**. For instance, `v >= 0 & v <= 1` is actually evaluated as `(v >= (0 & v)) <= 1` (because `&` has high precedence). When in doubt, use parentheses to force the correct parsing: `(v >= 0) & (v <= 1)`.

### Tip - `einops`

Einops is a useful library which we'll dive deeper with tomorrow. For now, the only important function you'll need to know is `einops.repeat`. This takes as arguments a tensor and a string, and returns a new tensor which has been repeated along the specified dimensions. For example, the following code shows how we can repeat a 2D tensor along the last dimension

```python
x = t.randn(2, 3)
x_repeated = einops.repeat(x, 'a b -> a b c', c=4)

assert x_repeated.shape == (2, 3, 4)
for c in range(4):
    t.testing.assert_close(x, x_repeated[:, :, c])
```

*(The function `t.testing.assert_close` checks that two tensors are the same shape, and all elements are the same value, within some very small numerical error.)*

### Tip - Logical Reductions

In plain Python, if you have a list of lists and want to know if any element in a row is `True`, you could use a list comprehension like `[any(row) for row in rows]`. The efficient way to do this in PyTorch is with `torch.any()` or equivalently the `.any()` method of a tensor, which accept the dimension to reduce over. Similarly, `torch.all()` or `.all()` method. Both of these methods accept a `dim` argument, which is the dimension to reduce over.

<details>
<summary>Aside - tensor methods</summary>

Most functions like `torch.any(tensor, ...)` (which take a tensor as first argument) have an equivalent tensor method `tensor.any(...)`. We'll see many more examples of functions like this as we go through the course.
</details>

### Tip - Broadcasting

Broadcasting is what happens when you perform an operation on two tensors, and one is a smaller size, but is copied along the dimensions of the larger one in order to apply to it. Below is an example (where `B` is copied along the first dimension of `A`):

```python
A = t.randn(2, 3)
B = t.randn(3)
AplusB = A + B

assert AplusB.shape == (2, 3)
for i in range(2):
    t.testing.assert_close(AplusB[i], A[i] + B)
```

Broadcasting sematics are a bit messy, and we'll go into it in more detail later in the course. If you want to get a full picture of it then click on the dropdown below, but for now here's the important thing to know - *dimensions get appended to the **start** of the smaller tensor `B`, and it is copied along those dimensions until its shape matches the larger tensor `A`*.

<details>
<summary>Aside - details of broadcasting</summary>

If you try to broadcast tensors `A` and `B`, then the following happens:

* The tensor with fewer dimensions is padded with dimensions of size one on the left.
* Once the tensors have the same number of dimensions, they are checked for compatibility.
    * Two dimensions are compatible if they are equal, or if one of them is one (in the latter case, we repeat the size-1 tensor along that dimension until it's the same size as the larger one).
    * If they are not compatible, then broadcasting is not allowed.

For instance, in the example above, first `B` is left-padded to shape `(1, 3)`, then it is copied along the first dimension to shape `(2, 3)`, when it can be added to `A`.

On the other hand, if `B` had shape `(2,)` then broadcasting would fail, because dimensions can't be added to the right of a tensor.
</details>


### Summary of all these tips

> * Use `...` to avoid repeated `:` in indexing expressions.
> * Use `&`, `|`, and `~` for elementwise logical operations on tensors.
> * Use parentheses to force the correct operator precedence.
> * Use `torch.any()` or `.any()` to do logical reductions (you can do this over a single dimension, with the `dim` argument).
> * If you're trying to broadcast tensors `A` and `B` (where `B` has fewer dimensions), this will work if the shape of `B` matches the ***last*** dimensions of `A` (in this case, `B` will get copied along the earlier dimensions of `A`).


### Exercise - implement `intersect_rays_1d`


```python
def intersect_rays_1d(rays: Float[T, "nrays 2 3"], segments: Float[T, "nsegments 2 3"]) -> Bool[T, "nrays"]:
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

```

<details>
<summary>Help - I'm not sure how to implment this function without a for loop.</summary>

Initially, `rays.shape == (NR, 2, 3)` and `segments.shape == (NS, 2, 3)`. Try performing `einops.repeat` on them, so both their shapes are `(NR, NS, 2, 3)`. Then you can formulate and solve the batched system of matrix equations.
</details>

<details>
<summary>Help - I'm not sure how to deal with the cases of zero determinant.</summary>

You can use `t.linalg.det` to compute the determinant of a matrix, or batch of matrices *(gotcha: the determinant won't be exactly zero, but you can check that it's very close to zero, e.g. `det.abs() < 1e-6`)*. This will give you a boolean mask for which matrices are singular.

You can set all singular matrices to the identity (this avoids errors), and then at the very end you can use your boolean mask again to set the intersection to `False` for the singular matrices.
</details>

<details>
<summary>Help - I'm still stuck on the zero determinant cases.</summary>

After formulating the matrix equation, you should have a batch of matrices of shape `(NR, NS, 2, 2)`, i.e. `mat[i, j, :, :]` is a matrix which looks like:

$$
\begin{pmatrix} D_x & (L_1 - L_2)_x \\ D_y & (L_1 - L_2)_y \\ \end{pmatrix}
$$

Calling `t.linalg.det(mat)` will return an array of shape `(NR, NS)` containing the determinants of each matrix. You can use this to construct a mask for the singular matrices *(gotcha: the determinant won't be exactly zero, but you can check that it's very close to zero, e.g. `det.abs() < 1e-6`)*.

Indexing `mat` by this mask will return an array of shape `(x, 2, 2)`, where the zeroth axis indexes the singular matrices. As we discussed in the broadcasting section earlier, this means we can use broadcasting to set all these singular matrices to the identity:

```python
mat[is_singular] = t.eye(2)
```
</details>

<details>
<summary>Solution</summary>


```python
def my_concat(x: Float[T, "a1 b"], y: Float[T, "a2 b"]) -> Float[T, "a1+a2 b"]:
    return t.concat([x, y], dim=0)

def intersect_rays_1d(rays: Float[T, "nrays 2 3"], segments: Float[T, "nsegments 2 3"]) -> Bool[T, "nrays"]:
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
```
</details>


### Using GPT to understand code


Next week we'll start learning about transformers and how to build them, but it's not too early to start using them to accelerate your own learning!

We'll be discussing more advanced ways to use GPT 3 and 4 as coding partners / research assistants in the coming weeks, but for now we'll look at a simple example: **using GPT to understand code**. You're recommended to read the recent [LessWrong post](https://www.lesswrong.com/posts/ptY6X3BdW4kgqpZFo/using-gpt-4-to-understand-code) by Siddharth Hiregowdara in which he explains his process. This works best on GPT-4, but I've found GPT-3.5 works equally well for reasonably straightforward problems (see the section below).

Firstly, you should get an account to use GPT with if you haven't already. Next, try asking GPT-3.5 / 4 for an explanation of the function above. You can do this e.g. via the following prompt:

```
Explain this Python function, line by line. You should break up your explanation by inserting sections of the code.

def intersect_rays_1d(rays: Float[T, "nrays 2 3"], segments: Float[T, "nsegments 2 3"]) -> Bool[T, "nrays"]:
    NR = rays.size(0)
    NS = segments.size(0)
    rays = rays[..., :2]
    ...
```

I've found removing comments is often more helpful, because then GPT will answer in its own words rather than just repeating the comments (and the comments can sometimes confuse it). 

Once you've got a response, here are a few more things you might want to consider asking:

* Can you suggest ways to improve the code?
    * GPT-4 recommended using a longer docstring and more descriptive variable names, among other things.
* Can you explain why the line `mat[is_singular] = t.eye(2)` works?
    * GPT-4 gave me a correct and very detailed explanation involving broadcasting and tensor shapes.

Is using GPT in this way cheating? It can be, if your first instinct is to jump to GPT rather than trying to understand the code yourself. But it's important here to bring up the distinction of [playing in easy mode](https://www.lesswrong.com/posts/nJPtHHq6L7MAMBvRK/play-in-easy-mode) vs [playing in hard mode](https://www.lesswrong.com/posts/7hLWZf6kFkduecH2g/play-in-hard-mode). There are situations where it's valuable for you to think about a problem for a while before moving forward because that deliberation will directly lead to you becoming a better researcher or engineer (e.g. when you're thinking of a hypothesis for how a circuit works while doing mechanistic interpretability on a transformer, or you're pondering which datastructure best fits your use case while implementing some RL algorithm). But there are also situations (like this one) where you'll get more value from speedrunning towards an understanding of certain code or concepts, and apply your understanding in subsequent exercises. It's important to find a balance!

#### When to use GPT-3.5 and GPT-4

GPT-3.5 and 4 both have advantages and disadvantages in different situations. GPT-3.5 has a large advantage in speed over GPT-4, and works equally well on simple problems or functions. If it's anything that Copilot is capable of writing, then you're likely better off using it instead of GPT-4.

On the other hand, GPT-4 has an advantage at generating coherent code (although we don't expect you to be using it for code generation much at this stage in the program), and is generally better at responding to complex tasks with less prompt engineering.

#### Additional notes on using GPT (from Joseph Bloom)

* ChatGPT is overly friendly. If you give it bad code it won't tell you it's shit so you need to encourage it to give feedback and/or show you examples of great code. Especially for beginner coders using it, it's important to realise how *under* critical it is. 
* GPT is great at writing tests (asking it to write a test for a function is often better than asking it if a function is correct), refactoring code (identifying repeated tasks and extracting them) and naming variables well. These are specific things worth doing a few times to see how useful they can be. 
* GPT-4 does well with whole modules/scripts so don't hesitate to add those. When you start managing repos on GitHub, use tracked files so that when you copy-paste edited code back, all the changes are highlighted for you as if you'd made them.

Here are some things you can play around with:

* Ask GPT it to write tests for the function. You can give more specific instructions (e.g. asking it to use / not to use the `unittests` library, or to print more informative error messages).
* Ask GPT how to refactor the function above. (When I did this, it suggested splitting the function up into subfunctions which performed the discrete tasks of "compute intersection points")


## 2D Rays

Now we're going to make use of the z dimension and have rays emitted from the origin in both y and z dimensions.


### Exercise - implement `make_rays_2d`

Implement `make_rays_2d` analogously to `make_rays_1d`. The result should look like a pyramid with the tip at the origin.


```python
@jaxtyped
@typeguard.typechecked
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
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

```

<details>
<summary>Help - I'm not sure how to implement this function.</summary>

Don't write it as a function right away. The most efficient way is to write and test each line individually in the REPL to verify it does what you expect before proceeding.

You can either build up the output tensor using `torch.stack`, or you can initialize the output tensor to its final size and then assign to slices like `rays[:, 1, 1] = ...`. It's good practice to be able to do it both ways.

Each y coordinate needs a ray with each corresponding z coordinate - in other words this is an outer product. The most elegant way to do this is with two calls to `einops.repeat`. You can also accomplish this with `unsqueeze`, `expand`, and `reshape` combined.
</details>

<details>
<summary>Solution</summary>


```python
def make_rays_2d(num_pixels_y: int, num_pixels_z: int, y_limit: float, z_limit: float) -> Float[t.Tensor, "nrays 2 3"]:
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
```
</details>


## Triangle Coordinates

The area inside a triangle can be defined by three (non-collinear) points $A$, $B$ and $C$, and can be written algebraically as a **convex combination** of those three points:

$$
\begin{align*}
P(w, u, v) &= wA + uB + vC \quad\quad \\
    \\
s.t. \quad 0 &\leq w,u,v \\
1 &= w + u + v
\end{align*}
$$

Or equivalently:

$$
\begin{align*}
\quad\quad\quad\quad P(u, v) &= (1 - u - v)A + uB + vC \\
&= A + u(B - A) + v(C - A) \\
\\
s.t. \quad 0 &\leq u,v \\
u + v &\leq 1
\end{align*}
$$

These $u, v$ are called "barycentric coordinates".

If we remove the bounds on $u$ and $v$, we get an equation for the plane containing the triangle. Play with the widget to understand the behavior of $u, v$.


```python
if MAIN:
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
	        xaxis_range=[-3, 8], yaxis_range=[-2, 5.5], height=600, width=800, template="simple_white"
	    )
	)
	
@interact(u=(-0.5, 1.5, 0.01), v=(-0.5, 1.5, 0.01))
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})

if MAIN:
	display(fig)

```

### Triangle-Ray Intersection

Given a ray with origin $O$ and direction $D$, our intersection algorithm will consist of two steps:

- Finding the intersection between the line and the plane containing the triangle, by solving the equation $P(u, v) = P(s)$;
- Checking if $u$ and $v$ are within the bounds of the triangle.

Expanding the equation $P(u, v) = P(s)$, we have:

$$
\begin{align*}
A + u(B - A) + v(C - A) &= O + sD \\ 
\Rightarrow
\begin{pmatrix}
    -D & (B - A) & (C - A) \\
\end{pmatrix}
\begin{pmatrix} 
    s \\ 
    u \\ 
    v  
\end{pmatrix}
&= \begin{pmatrix} O - A \end{pmatrix} \\
\Rightarrow \begin{pmatrix} 
    -D_x & (B - A)_x & (C - A)_x \\
    -D_y & (B - A)_y & (C - A)_y \\ 
    -D_z & (B - A)_z & (C - A)_z \\
\end{pmatrix}
\begin{pmatrix}
    s \\ 
    u \\ 
    v  
\end{pmatrix} &= \begin{pmatrix}
    (O - A)_x \\ 
    (O - A)_y \\ 
    (O - A)_z \\ 
\end{pmatrix}
\end{align*}
$$

$$
$$

We can therefore find the coordinates `s`, `u`, `v` of the intersection point by solving the linear system above.


### Exercise - implement `triangle_line_intersects`

Using `torch.linalg.solve` and `torch.stack`, implement `triangle_line_intersects(A, B, C, O, D)`.

A few tips:

* If you have a 0-dimensional tensor with shape `()` containing a single value, use the `item()` method to convert it to a plain Python value.
* If you have a tensor of shape `tensor.shape = (3, ...)`, then you can unpack it along the first dimension into three separate tensors the same way that you'd unpack a normal python list: `s, u, v = tensor`.
    * Note - if the dimension you want to unpack isn't at the start, a nice alternative is `s, u, v = tensor.unbind(dim)`, which does the same thing but along the dimension given by `dim` rather than the first dimension.
* If your function isn't working, try making a simple ray and triangle with nice round numbers where you can work out manually if it should intersect or not, then debug from there.


```python
Point = Float[T, "points=3"]

@jaxtyped
@typeguard.typechecked
def triangle_line_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    '''
    s, u, v = t.linalg.solve(
        t.stack([-D, B - A, C - A], dim=1), 
        O - A
    )
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()


if MAIN:
	tests.test_triangle_line_intersects(triangle_line_intersects)

```

<details>
<summary>Solution</summary>


```python
def response(u=0.0, v=0.0):
    P = A + u * (B - A) + v * (C - A)
    fig.data[2].update({"x": [P[0]], "y": [P[1]]})

def triangle_line_intersects(A: Point, B: Point, C: Point, O: Point, D: Point) -> bool:
    '''
    A: shape (3,), one vertex of the triangle
    B: shape (3,), second vertex of the triangle
    C: shape (3,), third vertex of the triangle
    O: shape (3,), origin point
    D: shape (3,), direction point

    Return True if the line and the triangle intersect.
    '''
    s, u, v = t.linalg.solve(
        t.stack([-D, B - A, C - A], dim=1), 
        O - A
    )
    return ((u >= 0) & (v >= 0) & (u + v <= 1)).item()
```
</details>


## Single-Triangle Rendering

Implement `raytrace_triangle` using only one call to `torch.linalg.solve`. 

Reshape the output and visualize with `plt.imshow`. It's normal for the edges to look pixelated and jagged - using a small number of pixels is a good way to debug quickly. 

If you think it's working, increase the number of pixels and verify that it looks less pixelated at higher resolution.


### Views and Copies

It's critical to know when you are making a copy of a `Tensor`, versus making a view of it that shares the data with the original tensor. It's preferable to use a view whenever possible to avoid copying memory unnecessarily. On the other hand, modifying a view modifies the original tensor which can be unintended and surprising. Consult [the documentation](https://pytorch.org/docs/stable/tensor_view.html) if you're unsure if a function returns a view. A short reference of common functions:

- `torch.expand`: always returns a view
- `torch.view`: always returns a view
- `torch.detach`: always returns a view
- `torch.repeat`: always copies
- `torch.clone`: always copies
- `torch.flip`: always copies (different than numpy.flip which returns a view)
- `torch.tensor`: always copies, but PyTorch recommends using `.clone().detach()` instead.
- `torch.Tensor.contiguous`: returns self if possible, otherwise a copy
- `torch.transpose`: returns a view if possible, otherwise (sparse tensor) a copy
- `torch.reshape`: returns a view if possible, otherwise a copy
- `torch.flatten`: returns a view if possible, otherwise a copy (different than numpy.flatten which returns a copy)
- `einops.repeat`: returns a view if possible, otherwise a copy
- `einops.rearrange`: returns a view if possible, otherwise a copy
- Basic indexing returns a view, while advanced indexing returns a copy.


### Storage Objects

Calling `storage()` on a `Tensor` returns a Python object wrapping the underlying C++ array. This array is 1D regardless of the dimensionality of the `Tensor`. This allows you to look inside the `Tensor` abstraction and see how the actual data is laid out in RAM.

Note that a new Python wrapper object is generated each time you call `storage()`, and both `x.storage() == x.storage()` and `x.storage() is x.storage()` evaluates to False.

If you want to check if two `Tensor`s share an underlying C++ array, you can compare their `storage().data_ptr()` fields. This can be useful for debugging.


### `Tensor._base`

If `x` is a view, you can access the original `Tensor` with `x._base`. This is an undocumented internal feature that's useful to know. Consider the following code:

```python
x = t.zeros(1024*1024*1024)
y = x[0]
del x
```

Here, `y` was created through basic indexing, so `y` is a view and `y._base` refers to `x`. This means `del x` won't actually deallocate the 4GB of memory, and that memory will remain in use which can be quite surprising. `y = x[0].clone()` would be an alternative here that does allow reclaiming the memory.


### Exercise - implement `raytrace_triangle`


```python
@jaxtyped
@typeguard.typechecked
def raytrace_triangle(
    rays: Float[T, "nrays rayPoints=2 dims=3"],
    triangle: Float[T, "trianglePoints=3 dims=3"]
) -> Bool[T, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat: Float[T, "NR 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
    dets: Float[T, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol: Float[T, "NR 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

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
	triangle_lines = t.stack([A, B, C, A, B, C], dim=0).reshape(-1, 2, 3)
	render_lines_with_plotly(rays2d, triangle_lines)
	intersects = raytrace_triangle(rays2d, test_triangle)
	img = intersects.reshape(num_pixels_y, num_pixels_z)
	px.imshow(img, origin="lower", labels={"x": "X", "y": "Y"}).update_layout(coloraxis_showscale=False, width=600).show()

```

<details>
<summary>Solution</summary>


```python
def raytrace_triangle(
    rays: Float[T, "nrays rayPoints=2 dims=3"],
    triangle: Float[T, "trianglePoints=3 dims=3"]
) -> Bool[T, "nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size(0)

    # Triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)
    assert A.shape == (NR, 3)

    # Each element of `rays` is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    O, D = rays.unbind(dim=1)
    assert O.shape == (NR, 3)

    # Define matrix on left hand side of equation
    mat: Float[T, "NR 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    # Note - this works because mat[is_singular] has shape (NR_where_singular, 3, 3), so we
    # can broadcast the identity matrix to that shape.
    dets: Float[T, "NR"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec = O - A

    # Solve eqns
    sol: Float[T, "NR 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    # Return boolean of (matrix is nonsingular, and solution is in correct range implying intersection)
    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
```
</details>


## Mesh Loading

You can download the pikachu `.pt` file from [this link](https://drive.google.com/drive/folders/15k_u8ESO2gVzs8_HUBgCAk-EFcb01LZw?usp=sharing), and save it to your Colab's local storage.

Use the given code to load the triangles for your Pikachu. By convention, files written with `torch.save` end in the `.pt` extension, but these are actually just zip files.


```python
if MAIN:
	with open("pikachu.pt", "rb") as f:
	    triangles = t.load(f)

```

## Mesh Rendering

For our purposes, a mesh is just a group of triangles, so to render it we'll intersect all rays and all triangles at once. We previously just returned a boolean for whether a given ray intersects the triangle, but now it's possible that more than one triangle intersects a given ray. 

For each ray (pixel) we will return a float representing the minimum distance to a triangle if applicable, otherwise the special value `float('inf')` representing infinity. We won't return which triangle was intersected for now.


### Exercise - implement `raytrace_mesh`

Implement `raytrace_mesh` and as before, reshape and visualize the output. Your Pikachu is centered on (0, 0, 0), so you'll want to slide the ray origin back to at least `x=-2` to see it properly.

Reminder - `t.linalg.solve` (and most batched operations) can accept multiple dimensions as being batch dims. Previously, you've just used `NR` (the number of rays) as the batch dimension, but you can also use `(NR, NT)` (the number of rays and triangles) as your batch dimensions, so you can solve for all rays and triangles at once.


```python
def raytrace_mesh(
    rays: Float[T, "nrays rayPoints=2 dims=3"],
    triangles: Float[T, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[T, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[T, "NR NT 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[T, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[T, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[T, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float("inf") # t.inf

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

```

<details>
<summary>Solution</summary>


```python
def raytrace_mesh(
    rays: Float[T, "nrays rayPoints=2 dims=3"],
    triangles: Float[T, "ntriangles trianglePoints=3 dims=3"]
) -> Bool[T, "nrays"]:
    '''
    For each ray, return the distance to the closest intersecting triangle, or infinity.
    '''

    NR = rays.size(0)
    NT = triangles.size(0)

    # Each triangle is [[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]]
    triangles = einops.repeat(triangles, "NT pts dims -> pts NR NT dims", NR=NR)
    A, B, C = triangles
    assert A.shape == (NR, NT, 3)

    # Each ray is [[Ox, Oy, Oz], [Dx, Dy, Dz]]
    rays = einops.repeat(rays, "NR pts dims -> pts NR NT dims", NT=NT)
    O, D = rays
    assert O.shape == (NR, NT, 3)

    # Define matrix on left hand side of equation
    mat: Float[T, "NR NT 3 3"] = t.stack([- D, B - A, C - A], dim=-1)
    # Get boolean of where matrix is singular, and replace it with the identity in these positions
    dets: Float[T, "NR NT"] = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    # Define vector on the right hand side of equation
    vec: Float[T, "NR NT 3"] = O - A

    # Solve eqns (note, s is the distance along ray)
    sol: Float[T, "NR NT 3"] = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(-1)

    # Get boolean of intersects, and use it to set distance to infinity wherever there is no intersection
    intersects = ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)
    s[~intersects] = float("inf") # t.inf

    # Get the minimum distance (over all triangles) for each ray
    return s.min(dim=-1).values
```
</details>


## Bonus Content

Congratulations, you've finished the main content for today!

Some fun extensions to try:

- Vectorize further to make a video. 
    - Each frame will have its own rays coming from a slightly different position.
    - Pan the camera around for some dramatic footage. 
    - One way to do it is using the `mediapy` library to render the video.
- Try rendering on the GPU and see if you can make it faster.
- Allow each triangle to have a corresponding RGB color value and render a colored image.
- Use multiple rays per pixel and combine them somehow to have smoother edges.




""", unsafe_allow_html=True)
