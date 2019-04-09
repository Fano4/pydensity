/*#ifndef MYBOOLEAN_H
#define MYBOOLEAN_H

#define false 0
#define true 1
typedef int bool; // or #define bool int

#endif
*/


#include <cstdlib>
#include <iostream>
#include <math.h>

void spherical_harmonics(double *thet,double *phi, int *l,int *ml,double* result,int length);
double factorial(int n);
double legendre(int l,double x); 
double associated_legendre(int l,int m,double x); 
double associated_legendre_nonorm(int l,int m,double x); 
void cart_to_spher(double *x,double *y,double *z,double * r,double *t,double *f,int length);
void density_cont_val(double* reorthog,double* imorthog,double *x,double*y,double*z,double*kx,double*ky,double*kz,int nx,int ny,int nz,int nk,double* dens_val);
double build_reduced_determinant( int ai,int aj,int n_elec,int n_closed,int n_occ,int* mo_vector_1,int* mo_vector_2,int *spin_vector_1,int *spin_vector_2);
void build_transition_density_matrix(int n_states_neut,int n_closed,int n_occ,int ci_size_neut,int n_elec_neut,double *ci_vector,int *mos_vector,int *spin_vector,double *tran_den_mat_mo);


//#include "spherical_utils.cpp"

