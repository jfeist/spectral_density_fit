from spectral_density_fit import __version__, spectral_density_fitter, Jmod, Jmod_naive

# run the tests in 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np

def test_version():
    assert __version__ == '0.2.0'

def get_test_data(complex):
    Nm = 3
    Ne = 4
    np.random.seed(1234)
    ω = np.linspace(-3.5,3.5,101)
    H = np.random.normal(size=(Nm,Nm))
    Heff = H + H.T - 0.5j*np.diag(np.random.rand(Nm))

    g = np.random.normal(size=(Ne,Nm))
    if complex:
        g = g + 1j*np.random.normal(size=(Ne,Nm))

    return Nm, Ne, ω, Heff, g

def test_Jmod_real():
    Nm, Ne, ω, Heff, g = get_test_data(False)
    J1 = Jmod(ω,Heff,g)
    J2 = Jmod_naive(ω,Heff,g)
    assert np.isrealobj(J1)
    assert np.isrealobj(J2)
    assert np.allclose(J1,J2)

def test_Jmod_complex():
    Nm, Ne, ω, Heff, g = get_test_data(True)
    J1 = Jmod(ω,Heff,g)
    J2 = Jmod_naive(ω,Heff,g)
    assert np.iscomplexobj(J1)
    assert np.iscomplexobj(J2)
    assert np.allclose(J1,J2)

def test_real():
    Nm, Ne, ω, Heff, g = get_test_data(False)
    J = Jmod(ω,Heff,g)
    opt = spectral_density_fitter(ω,J,Nm)
    ps = opt.Hκg_to_ps(Heff.real + 0.03*((x:=np.random.rand(Nm,Nm)) + x.T), -2*Heff.imag.diagonal() + 0.1*np.random.rand(Nm), g + 0.1*np.random.rand(Ne,Nm))
    ps = opt.optimize(ps)
    assert np.isrealobj(opt.Jfun(ω,ps))
    assert opt.obj_fun(ps,np.empty(0)) < 0.001

def test_complex():
    Nm, Ne, ω, Heff, g = get_test_data(True)
    J = Jmod(ω,Heff,g)
    opt = spectral_density_fitter(ω,J,Nm)
    ps = opt.Hκg_to_ps(Heff.real + 0.03*((x:=np.random.rand(Nm,Nm)) + x.T), -2*Heff.imag.diagonal() + 0.1*np.random.rand(Nm), g + 0.1*np.random.rand(Ne,Nm))
    ps = opt.optimize(ps)
    assert np.iscomplexobj(opt.Jfun(ω,ps))
    assert opt.obj_fun(ps,np.empty(0)) < 0.005
