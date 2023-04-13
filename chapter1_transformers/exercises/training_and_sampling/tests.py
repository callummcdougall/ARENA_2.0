import torch as t

def test_tensor_dataset(TensorDataset):
    tensors = [t.rand((10, 20)), t.rand((10, 5)), t.arange(10)]
    dataset = TensorDataset(*tensors)
    assert len(dataset) == 10
    for index in [0, slice(0, 5, 1), slice(1, 5, 2)]:
        print("Testing with index:", index)
        expected = tuple(tensor[index] for tensor in tensors)
        actual = dataset[index]
        for e, a in zip(expected, actual):
            t.testing.assert_close(e, a)
    print("All tests in `test_tensor_dataset` passed!")