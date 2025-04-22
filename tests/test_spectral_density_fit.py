from spectral_density_fit import __version__, spectral_density_fitter, Jmod, Jmod_naive

import numpy as np
import jax
import pytest

# run the tests in 64-bit precision
jax.config.update("jax_enable_x64", True)


def test_version():
    assert __version__ == "0.2.1"


def get_test_data(complex):
    Nm = 3
    Ne = 4
    np.random.seed(1234)
    ω = np.linspace(-3.5, 3.5, 101)
    H = np.random.normal(size=(Nm, Nm))
    Heff = H + H.T - 0.5j * np.diag(np.random.rand(Nm))

    g = np.random.normal(size=(Ne, Nm))
    if complex:
        g = g + 1j * np.random.normal(size=(Ne, Nm))

    return Nm, Ne, ω, Heff, g


def get_bench_data(complex):
    Nm = 20
    Ne = 12
    np.random.seed(1234)
    ω = np.linspace(-3.5, 3.5, 1001)
    H = np.random.normal(size=(Nm, Nm))
    Heff = H + H.T - 0.5j * np.diag(np.random.rand(Nm))

    g = np.random.normal(size=(Ne, Nm))
    if complex:
        g = g + 1j * np.random.normal(size=(Ne, Nm))

    return Nm, Ne, ω, Heff, g


def randomize_Hκg(Heff, g, Hfac=0.03, κfac=0.1, gfac=0.1):
    Ne, Nm = g.shape
    x = np.random.normal(size=(Nm, Nm))
    Hr = Heff.real + 0.5 * Hfac * (x + x.T)
    κr = abs(-2 * Heff.imag.diagonal() + κfac * np.random.normal(Nm))
    gr = g + gfac * np.random.normal(size=(Ne, Nm))
    if np.iscomplexobj(g):
        gr += 1j * gfac * np.random.normal(size=(Ne, Nm))
    return (Hr, κr, gr)


def _test_Jmod(complex):
    Nm, Ne, ω, Heff, g = get_test_data(complex)
    J1 = Jmod(ω, Heff, g)
    J2 = Jmod_naive(ω, Heff, g)
    istypeobj = np.iscomplexobj if complex else np.isrealobj
    assert istypeobj(J1)
    assert istypeobj(J2)
    assert np.allclose(J1, J2)


def test_Jmod_real():
    _test_Jmod(False)


def test_Jmod_complex():
    _test_Jmod(True)


def _test_gradient(complex):
    Nm, Ne, ω, Heff, g = get_test_data(complex)
    J = Jmod(ω, Heff, g)
    opt_1 = spectral_density_fitter(ω, J, Nm, diagonalize=False)
    opt_2 = spectral_density_fitter(ω, J, Nm, diagonalize=True)

    ps = opt_1.Hκg_to_ps(*randomize_Hκg(Heff, g))
    grad_1 = np.empty_like(ps)
    grad_2 = np.empty_like(grad_1)
    err_1 = opt_1.obj_fun(ps, grad_1)
    err_2 = opt_2.obj_fun(ps, grad_2)
    assert np.allclose(err_1, err_2)
    assert np.allclose(grad_1, grad_2)


def test_gradient():
    for complex in (False, True):
        _test_gradient(complex)


@pytest.mark.benchmark(warmup=True)
def _test_gradient_bench(benchmark, complex, diagonalize):
    Nm, Ne, ω, Heff, g = get_bench_data(complex)
    J = Jmod(ω, Heff, g)
    opt = spectral_density_fitter(ω, J, Nm, diagonalize=diagonalize)
    ps = opt.Hκg_to_ps(*randomize_Hκg(Heff, g))
    grad = np.empty_like(ps)
    benchmark(opt.obj_fun, ps, grad)


def test_bench_gradient_real(benchmark):
    _test_gradient_bench(benchmark, False, True)


def test_bench_gradient_complex(benchmark):
    _test_gradient_bench(benchmark, True, True)


def test_bench_gradient_real_nodiag(benchmark):
    _test_gradient_bench(benchmark, False, False)


def test_bench_gradient_complex_nodiag(benchmark):
    _test_gradient_bench(benchmark, True, False)


def _test_fitting(complex, thresh):
    Nm, Ne, ω, Heff, g = get_test_data(complex)
    J = Jmod(ω, Heff, g)
    opt = spectral_density_fitter(ω, J, Nm)
    ps = opt.Hκg_to_ps(*randomize_Hκg(Heff, g, Hfac=0.005, κfac=0.001, gfac=0.01))
    opt.set_ftol_rel(0)
    # since we are fitting to a spectral density given by the few-mode model, we
    # know that the global minimum is at zero
    opt.set_stopval(thresh)
    ps = opt.optimize(ps)
    if complex:
        assert np.iscomplexobj(opt.Jfun(ω, ps))
    else:
        assert np.isrealobj(opt.Jfun(ω, ps))
    assert opt.obj_fun(ps, np.empty(0)) < thresh


def test_fitting_real():
    _test_fitting(False, 0.005)


def test_fitting_complex():
    _test_fitting(True, 0.005)
