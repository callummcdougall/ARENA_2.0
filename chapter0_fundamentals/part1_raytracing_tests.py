import torch as t

def test_intersect_ray_1d(intersect_ray_1d):
    import part1_raytracing_solution as soln

    expected = [(0, 0), (0, 1), (2, 7), (2, 8)]
    actual = []
    for i, segment in enumerate(soln.segments):
        for j, ray in enumerate(soln.rays1d):
            if intersect_ray_1d(ray, segment):
                actual.append((i, j))
    if expected != actual:
        print("Expected segment-ray intersections: ", expected)
        print("Actual:", actual)
    assert expected == actual
    print("All tests in `test_intersect_ray_1d` passed!")


def test_intersect_ray_1d_special_case(intersect_ray_1d):
    ray = t.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    segment = t.tensor([[0.0, 2.0, 2.0], [0.0, 4.0, 4.0]])
    actual = intersect_ray_1d(ray, segment)
    assert actual == False
    print("All tests in `test_intersect_ray_1d_special_case` passed!")


def test_intersect_rays_1d(intersect_rays_1d):
    import part1_raytracing_solution as soln

    expected = t.tensor([True, True, False, False, False, False, False, True, True])
    actual = intersect_rays_1d(soln.rays1d, soln.segments)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_intersect_rays_1d` passed!")


def test_intersect_rays_1d_special_case(intersect_rays_1d):
    ray = t.tensor([[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, -10.0, 0.0]]])
    segment = t.tensor(
        [
            [[0.0, 2.0, 2.0], [0.0, 4.0, 4.0]],
            [[1.0, -12.0, 0.0], [1.0, -6.0, 0.0]],
        ]
    )
    actual = intersect_rays_1d(ray, segment)
    expected = t.tensor([False, True])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_intersect_rays_1d_special_case` passed!")


def test_triangle_line_intersects(triangle_line_intersects):
    A = t.tensor([2, 0.0, -1.0])
    B = t.tensor([2, -1.0, 0.0])
    C = t.tensor([2, 1.0, 1.0])
    O = t.tensor([0.0, 0.0, 0.0])
    D = t.tensor([1.0000, 0.3333, 0.3333])
    assert triangle_line_intersects(A, B, C, O, D)

    O2 = t.tensor([0.0, 0.0, 0.0])
    D2 = t.tensor([1.0, 1.0, -1.0])
    assert not triangle_line_intersects(A, B, C, O2, D2)
    print("All tests in `test_triangle_line_intersects` passed!")
