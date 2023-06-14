import os
import sys
from pathlib import Path
import torch as t
import torch.nn.functional as F
import einops

def test_model(Model):
    import part7_toy_models_of_superposition.solutions as solutions
    config = solutions.Config(10, 5, 2)
    # get actual
    model = Model(config)
    model_soln = solutions.Model(config)
    assert set(model.state_dict().keys()) == set(model_soln.state_dict().keys()), "Incorrect parameters."
    model.load_state_dict(model_soln.state_dict())
    batch = model_soln.generate_batch(10)
    out_actual = model(batch)
    out_expected = model_soln(batch)
    assert out_actual.shape == out_expected.shape, f"Expected shape {out_expected.shape}, got {out_actual.shape}"
    assert t.allclose(out_actual, F.relu(out_actual)), "Did you forget to apply the ReLU (or do it in the wrong order)?"
    assert t.allclose(out_actual, out_expected), "Incorrect output when compared to solution."
    print("All tests in `test_model` passed!")


def test_generate_batch(Model):
    import part7_toy_models_of_superposition.solutions as solutions
    n_features = 5
    n_instances = 10
    n_hidden = 2
    batch_size = 5000
    config = solutions.Config(n_instances, n_features, n_hidden)
    feature_probability=(t.arange(1, 11) / 11).unsqueeze(-1)
    model = Model(config, feature_probability=feature_probability)
    batch = model.generate_batch(batch_size)
    assert batch.shape == (batch_size, n_instances, n_features), f"Expected shape (500, 10, 5), got {batch.shape}"
    assert t.allclose(batch, batch.clamp(0, 1)), "Not all elements of batch are in the [0, 1] range."
    feature_probability = (batch.abs() > 1e-5).float().mean((0, -1))
    diff = (feature_probability - model.feature_probability.squeeze()).abs().sum()
    assert diff < 0.05, "Incorrect feature_probability implementation."
    print("All tests in `test_generate_batch` passed!")


def test_calculate_loss(calculate_loss):
    import part7_toy_models_of_superposition.solutions as solutions
    batch_size, instances, features = (10, 5, 5)
    out = t.rand((batch_size, instances, features))
    batch = t.rand((batch_size, instances, features))

    unif_importance = t.ones((instances, features))
    model_unif_importance=solutions.Model(solutions.Config(n_instances=instances, n_features=features), importance=unif_importance)
    model_unif_importance.importance = model_unif_importance.importance.cpu()
    loss_actual = calculate_loss(out, batch, model_unif_importance)
    loss_expected = einops.reduce(unif_importance * ((batch - out)**2), 'batch instances features -> instances', 'mean').sum()
    assert t.allclose(loss_actual, loss_expected)

    nonunif_importance = t.rand((instances, features))
    model_nonunif_importance=solutions.Model(solutions.Config(n_instances=instances, n_features=features), importance=nonunif_importance)
    model_nonunif_importance.importance = model_nonunif_importance.importance.cpu()
    loss_actual = calculate_loss(out, batch, model_nonunif_importance)
    loss_expected = einops.reduce(nonunif_importance * ((batch - out)**2), 'batch instances features -> instances', 'mean').sum()
    assert t.allclose(loss_actual, loss_expected)

    print("All tests in `test_calculate_loss` passed!")


def test_compute_dimensionality(compute_dimensionality):
    import part7_toy_models_of_superposition.solutions as solutions
    W = t.randn(5, 20, 40)
    result = compute_dimensionality(W)
    expected = solutions.compute_dimensionality(W)
    t.testing.assert_close(result, expected)
    print("All tests in `test_compute_dimensionality` passed!")