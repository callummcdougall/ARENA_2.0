import torch as t
from torch import Tensor
from typing import Callable
from jaxtyping import Float

# FLAT SOLUTION
# YOUR CODE HERE - define `blank_probe` and `my_probe`

# FLAT SOLUTION END

import part6_othellogpt.tests as tests

def test_my_probes(blank_probe: t.Tensor, my_probe: t.Tensor, linear_probe: t.Tensor):

    blank_probe_expected = linear_probe[..., 0] - linear_probe[..., 1] * 0.5 - linear_probe[..., 2] * 0.5
    my_probe_expected = linear_probe[..., 2] - linear_probe[..., 1]

    t.testing.assert_close(blank_probe, blank_probe_expected)
    t.testing.assert_close(my_probe, my_probe_expected)
    print("All tests in `test_my_probes` passed!")


def test_apply_scale(apply_scale: Callable):

    pos = 20
    resid = t.randn(1, 60, 512)
    flip_dir = t.randn(512)
    flip_dir_normed = flip_dir / flip_dir.norm()
    alpha = resid[0, pos] @ flip_dir_normed

    for scale in [0, 2, 8]:
        resid_expected = resid.clone()
        resid_expected[0, pos] -= (scale+1) * alpha * flip_dir_normed
        resid_actual = apply_scale(resid.clone(), flip_dir, scale, pos)
        t.testing.assert_close(resid_expected, resid_actual)

    print("All tests in `test_apply_scale` passed!")
