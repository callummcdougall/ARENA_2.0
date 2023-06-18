import torch
from solutions import  QTensor, quantize_tensor, dequantize_tensor, calcScaleZeroPoint, quantizeLayer

def test_daxpy_random_input(fn1, fn2):

    alpha = torch.rand(1, device='cuda')
    x = torch.randn(1823, 1823, device='cuda')
    y = torch.randn(1823, 1823, device='cuda')

    assert torch.allclose(fn1(alpha, x, y), fn2(alpha, x, y), 0, 1e-6), "Implementations are not analogous"
    print('Tests passed')

def test_quantize_tensor(fn):

    test_tensor = torch.randn(5,5, device = 'cuda')

    correct = quantize_tensor(test_tensor)
    test = fn(test_tensor)

    #assert type(correct) == type(test), 'output types are different'
    assert torch.all(torch.eq(correct[0],test[0])) , 'quantized tensors are incorrect'
    assert correct[1] == test[1], 'the scale in incorrect'
    assert correct[2] == test[2], 'the zero point is incorrect'
    print('Tests passed') 

def test_dequantize_tensor(fn):

    test_tensor = QTensor(tensor=torch.randn(5,5, device='cuda'), scale=8, zero_point=1)

    correct = dequantize_tensor(test_tensor)
    test = fn(test_tensor)

    assert torch.all(torch.eq(correct, test)), 'quantized tensors is incorrect'

    print('Tests passed')

def test_calc_scale_zero_point(fn):

    min_val = 0.0
    max_val = 2.**32 -1
    num_bits = 8

    correct = calcScaleZeroPoint(min_val, max_val, num_bits)
    test = fn(min_val, max_val, num_bits)

    assert correct[0] == test[0], 'The scale value is incorrect'
    assert correct[1] == test[1], 'The zero point in incorrect'

    print('Tests passed')

def test_quantize_layer(fn):

    x = torch.randn(1,28,28, device='cuda')
    
    conv1 = torch.nn.Conv2d(1,20,5,1, device='cuda')
    stats = {}
    stats['conv1'] = {}
    stats['conv1']['min'] = 0.0
    stats['conv1']['max'] = 2.**32 - 1


    stats['conv2'] = {}
    stats['conv2']['min'] = 0.0
    stats['conv2']['max'] = 2.**32 - 1

    x = quantize_tensor(x, min_val = stats['conv1']['min'], max_val = stats['conv1']['max'])


    correct_x, correct_scale_next, correct_zero_point_next = quantizeLayer(x.tensor, conv1, stats['conv2'], x.scale, x.zero_point)
    fn_x, fn_scale_next, fn_zero_point_next = fn(x.tensor, conv1, stats['conv2'], x.scale, x.zero_point)

    assert torch.all(torch.eq(correct_x,fn_x)), 'Incorrect output tensor values'
    assert correct_scale_next == fn_scale_next, 'Incorrect scale value'
    assert correct_zero_point_next == fn_zero_point_next, 'Incorrect zero point value'

    print('Tests passed')



