import torch as t

import solutions

p = solutions.p
device = solutions.device
fourier_basis = solutions.fourier_basis


# %% TESTS FOR SECTION 1

def test_make_fourier_basis(make_fourier_basis):

    p = 17
    fourier_basis_actual, fourier_names_actual = make_fourier_basis(p)
    fourier_basis_expected, fourier_names_expected = solutions.make_fourier_basis(p)

    t.testing.assert_close(fourier_basis_actual, fourier_basis_expected)
    assert fourier_names_actual == fourier_names_expected

    print('All tests in `test_make_fourier_basis` passed!')


def test_fft1d(fft1d): 

    x = t.randn(p).to(device)
    actual = fft1d(x)
    expected = solutions.fft1d(x)
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a vector input.")

    print("Tests passed for `fft1d` with a vector input!")

    x = t.randn(3, p).to(device)
    actual = fft1d(x)
    expected = solutions.fft1d(x)
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a batch of vectors.")

    print("Tests passed for `fft1d` with a batch of vectors!")


def test_fourier_2d_basis_term(fourier_2d_basis_term):

    (i, j) = (3, 5)
    actual = fourier_2d_basis_term(i, j)
    expected = solutions.fourier_2d_basis_term(i, j)

    t.testing.assert_close(actual, expected)

    print('All tests in `test_fourier_2d_basis_term` passed!')



def test_fft2d(fft2d): 

    x = t.randn(p, p).to(device)
    actual = fft2d(x)
    expected = solutions.fft2d(x)
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a single input.")

    print("Tests passed for `fft1d` with a single input!")

    x = t.randn(p, p, 3).to(device)
    actual = fft2d(x)
    expected = solutions.fft2d(x)
    t.testing.assert_close(actual, expected, msg="Tests failed for `fft1d` with a batch of inputs.")

    print("Tests passed for `fft1d` with a batch of inputs!")


# %% TESTS FOR SECTION 2

def test_project_onto_direction(project_onto_direction):

    v = t.randn(3).to(device)
    batch_vecs = t.randn(3, 4).to(device)
    actual = project_onto_direction(batch_vecs, v)
    expected = solutions.project_onto_direction(batch_vecs, v)

    t.testing.assert_close(actual, expected)

    print('All tests in `test_project_onto_direction` passed!')


def test_project_onto_frequency(project_onto_frequency):

    freq = 7
    batch_vecs = t.randn(p*p, 4).to(device)
    actual = project_onto_frequency(batch_vecs, freq)
    expected = solutions.project_onto_frequency(batch_vecs, freq)

    t.testing.assert_close(actual, expected)

    print('All tests in `test_project_onto_frequency` passed!')


def test_get_trig_sum_directions(get_trig_sum_directions):

    k = 3
    actual = get_trig_sum_directions(3)
    expected = solutions.get_trig_sum_directions(3)

    t.testing.assert_close(actual, expected)

    print('All tests in `test_get_trig_sum_directions` passed!')