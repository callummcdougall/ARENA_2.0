from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard
from torch import Tensor
import torch as t
@jaxtyped
@typeguard.typechecked
def my_concat(x: Float[Tensor, "a1, b"], y: Float[Tensor, "a2, b"]) -> Float[Tensor, "a1+a2, b"]:
    return t.concat([x, y], dim=0)

x = t.ones(3, 2)
y = t.randn(4, 2)
z = my_concat(x, y)
