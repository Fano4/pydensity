void spherical_harmonics(double *thet,double *phi, int *l,int *ml,double* result,int length);
double factorial(int n);
double legendre(int l,double x); 
double associated_legendre(int l,int m,double x); 
double associated_legendre_nonorm(int l,int m,double x); 
void cart_to_spher(double *x,double *y,double *z,double * r,double *t,double *f,int length);
void density_cont_val(double* reorthog,double* imorthog,double *x,double*y,double*z,double*kx,double*ky,double*kz,int nx,int ny,int nz,int nk,double* dens_val);

#include "spherical_utils.cpp"
