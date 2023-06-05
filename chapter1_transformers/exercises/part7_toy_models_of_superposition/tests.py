import os
import sys
from pathlib import Path
import torch as t
import torch.nn.functional as F

def test_model(Model):
    import part7_toy_models_of_superposition.solutions as solutions
    config = solutions.Config(10, 5, 2)
    # get actual
    model = Model(config)
    t.manual_seed(0)
    model.W.data = t.randn_like(model.W.data)
    model.b_final.data = t.randn_like(model.b_final.data)
    batch = model.generate_batch(10)
    out_actual = model(batch)
    # get expected
    expected = solutions.Model(config)
    t.manual_seed(0)
    model.W.data = t.randn_like(model.W.data)
    model.b_final.data = t.randn_like(model.b_final.data)
    batch = model.generate_batch(10)
    out_expected = model(batch)
    assert out_actual.shape == out_expected.shape, f"Expected shape {out_expected.shape}, got {out_actual.shape}"
    assert t.allclose(out_actual, F.relu(out_actual)), "Did you forget to apply the ReLU (or do it in the wrong order)?"
    assert t.allclose(out_actual, out_expected), "Incorrect output when compared to solution."
    print("All tests in `test_model` passed!")