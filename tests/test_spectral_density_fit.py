from spectral_density_fit import __version__, spectral_density_fitter, Jmod

import numpy as np

def test_version():
    assert __version__ == '0.1.0'

def test_real():
    Nm = 3
    Ne = 4
    np.random.seed(1234)
    ω = np.linspace(-3.5,3.5,101)
    H = np.random.normal(size=(Nm,Nm))
    Heff = H + H.T - 0.5j*np.diag(np.random.rand(Nm))
    g = np.random.normal(size=(Ne,Nm))
    J = Jmod(ω,Heff,g)
    opt = spectral_density_fitter(ω,J,Nm)
    ps = opt.Hκg_to_ps(Heff.real + 0.03*((x:=np.random.rand(Nm,Nm)) + x.T), -2*Heff.imag.diagonal() + 0.1*np.random.rand(Nm), g + 0.1*np.random.rand(Ne,Nm))
    ps = opt.optimize(ps)
    assert opt.obj_fun(ps,np.empty(0)) < 0.001

def test_complex():
    Nm = 3
    Ne = 4
    np.random.seed(1234)
    ω = np.linspace(-3.5,3.5,101)
    H = np.random.normal(size=(Nm,Nm))
    Heff = H + H.T - 0.5j*np.diag(np.random.rand(Nm))
    g = np.random.normal(size=(Ne,Nm)) + 1j*np.random.normal(size=(Ne,Nm))
    J = Jmod(ω,Heff,g)
    opt = spectral_density_fitter(ω,J,Nm,complex=True)
    ps = opt.Hκg_to_ps(Heff.real + 0.03*((x:=np.random.rand(Nm,Nm)) + x.T), -2*Heff.imag.diagonal() + 0.1*np.random.rand(Nm), g + 0.1*np.random.rand(Ne,Nm))
    ps = opt.optimize(ps)
    assert opt.obj_fun(ps,np.empty(0)) < 0.005