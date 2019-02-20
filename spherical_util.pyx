
cimport numpy as np

cdef extern from "spherical_util.h":
    double legendre(unsigned int l,double x)
    double associated_legendre(unsigned int l,int m,double x)
    double associated_legendre_nonorm(unsigned int l,int m,double x)
    void spherical_harmonics(double *thet,double *phi,int *l,int *ml,double *result,int length)
    void cart_to_spher(double *x,double *y,double *z,double * r,double *t,double *f,int length)
    void density_cont_val(double* reorthog,double *imorthog,double *x,double*y,double*z,double*kx,double*ky,double*kz,int nx,int ny,int nz,int nk,double* dens_val)

def pspher_harmo(np.ndarray[double, ndim=1,mode="c"] thet,np.ndarray[double, ndim=1,mode="c"] phi,np.ndarray[ int, ndim=1,mode="c"] l,np.ndarray[int, ndim=1,mode="c"] ml,np.ndarray[double, ndim=1,mode="c"] result):

    spherical_harmonics(&thet[0],&phi[0],&l[0],&ml[0],&result[0],len(thet))

    return 0

def plegendre(l,x):
    return legendre(l,x)

def passociated_legendre(l,m,x):
    return associated_legendre(l,m,x)

def passociated_legendre_nonorm(l,m,x):
    return associated_legendre_nonorm(l,m,x)

def pcart_to_spher(np.ndarray[double, ndim=1,mode="c"] x,np.ndarray[double, ndim=1,mode="c"] y,np.ndarray[double, ndim=1,mode="c"] z,np.ndarray[double, ndim=1,mode="c"] r,np.ndarray[double, ndim=1,mode="c"] t,np.ndarray[double, ndim=1,mode="c"] f):

    cart_to_spher(&x[0],&y[0],&z[0],&r[0],&t[0],&f[0],len(x))

def pdensity_cont_val(np.ndarray[double, ndim=1,mode="c"] reorthog,np.ndarray[double, ndim=1,mode="c"] imorthog,np.ndarray[double, ndim=1,mode="c"] x,np.ndarray[double, ndim=1,mode="c"] y,np.ndarray[double, ndim=1,mode="c"] z,np.ndarray[double, ndim=1,mode="c"] kx, np.ndarray[double, ndim=1,mode="c"] ky,np.ndarray[double, ndim=1,mode="c"] kz, nx, ny, nz, nk,np.ndarray[double, ndim=1,mode="c"] dens_val):

    print("Entering final density routine")
    density_cont_val(&reorthog[0],&imorthog[0],&x[0],&y[0],&z[0],&kx[0],&ky[0],&kz[0],nx,ny,nz,nk,&dens_val[0])
    print("Density routine done!")

    return 0

