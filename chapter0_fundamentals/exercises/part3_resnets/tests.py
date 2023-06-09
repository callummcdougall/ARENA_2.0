import torch as t
from torch import nn

def test_batchnorm2d_module(BatchNorm2d):
    '''The public API of the module should be the same as the real PyTorch version.'''
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
    assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
    assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"
    print("All tests in `test_batchnorm2d_module` passed!")

def test_batchnorm2d_forward(BatchNorm2d):
    '''For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps).'''
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
    t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)
    print("All tests in `test_batchnorm2d_forward` passed!")

def test_batchnorm2d_running_mean(BatchNorm2d):
    '''Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean.'''
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 30
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        t.testing.assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    t.testing.assert_close(actual_eval_mean, t.zeros(3))
    print("All tests in `test_batchnorm2d_running_mean` passed!")

def test_get_resnet_for_feature_extraction(get_resnet_for_feature_extraction):

    resnet: nn.Module = get_resnet_for_feature_extraction(10)

    num_params = len(list(resnet.parameters()))

    error_msg = "\nNote - make sure you've defined your resnet modules in the correct order (with the final linear layer last), \
otherwise this can cause issues for the test function."

    # Check all gradients are correct
    for i, (name, param) in enumerate(resnet.named_parameters()):
        if i < num_params - 2:
            assert not param.requires_grad, f"Found param {name!r} before the final layer, which has requires_grad=True." + error_msg
        else:
            assert param.requires_grad, f"Found param {name!r} in the final layer, which has requires_grad=False." + error_msg
            if param.ndim == 2:
                assert tuple(param.shape) == (10, 512), f"Expected final linear layer weights to have shape (n_classes=10, 512), instead found {tuple(param.shape)}" + error_msg
            else:
                assert tuple(param.shape) == (10,), f"Expected final linear layer bias to have shape (n_classes=10,), instead found {tuple(param.shape)}" + error_msg
    
    print("All tests in `test_get_resnet_for_feature_extraction` passed!")