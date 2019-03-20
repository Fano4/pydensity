double factorial(int n)
{
   if(n <= 1 && n >= 0 )
      return 1;
   else if(n > 1)
      return n*factorial(n-1);
   else
   {
      exit(EXIT_FAILURE);
   }
}
double associated_legendre(int l,int m,double x)
{

   if(x==1 || l < m)
      return 0;
   else
   {
      if(m == 0)
      {
         return sqrt((2*l+1)/(4*acos(-1)))*legendre(l,x);
      }
      else if(m > 0)
      {
         return sqrt((2*l+1) * factorial(l-m) / (4 * acos(-1) * factorial(l+m)))
            * ((l-m+1) * x * associated_legendre_nonorm(l,m-1,x) - (l+m-1) * associated_legendre_nonorm(l-1,m-1,x)) / sqrt(1-x*x);
      }
      else 
      {
         return factorial(l+m)*associated_legendre(l,-m,x)/factorial(l-m);
      }
   }
   return 0;
}
double associated_legendre_nonorm(int l,int m,double x)
{

   if(x==1 || l < m)
      return 0;
   else
   {
      if(m == 0)
      {
         return legendre(l,x);
      }
      else if(m > 0)
      {
         return ((l-m+1)*x*associated_legendre_nonorm(l,m-1,x)-(l+m-1)*associated_legendre_nonorm(l-1,m-1,x))/sqrt(1-x*x);
      }
      else 
      {
         return associated_legendre_nonorm(l,-m,x);
      }
   }
   return 0;
}

double legendre( int l,double x)
{
   switch (l)
   {
      case 0:
         return 1;
      case 1:
         return x;
      default:
         return ((2*l-1)*x*legendre(l-1,x)-(l-1)*legendre(l-2,x))/l; 
   }
}

void spherical_harmonics(double* thet,double* phi, int* lp,int* ml,double* result,int length)
{
   int sign;
   int i;
   unsigned int l;


   for(i=0;i!=length;i++)
   {
      l=lp[i];
      sign=(-1) * ( ml[i] % 2 != 0 ) + ( ml[i] % 2 == 0 );

       if(ml[i] < 0)
          result[i] =  sign * sqrt(2.) * sin( -ml[i] * phi[i]) * associated_legendre( l , -ml[i] , cos(thet[i]));

       else if( ml[i] == 0 )
          result[i] = sign * associated_legendre( l , ml[i], cos(thet[i]));

       else
          result[i] =  sign * sqrt(2.) * cos( ml[i] * phi[i] ) * associated_legendre( l , ml[i] , cos(thet[i]));
   }
}
void cart_to_spher(double* x,double* y,double* z,double * r,double* t,double *f,int length)
{
   int i=0;
   for( i = 0 ; i != length ; i++)
   {
      r[i]=sqrt(x[i]*x[i]+y[i]*y[i]+z[i]*z[i]);

      if(r[i]==0)
      {
         t[i]=0;
         f[i]=0;
      }
      else
      {
         t[i]=acos(z[i]/r[i]);
         if(x[i] == 0 && y[i] > 0)
         {
            f[i]=acos(-1)/2.;
         }
         else if (x == 0 && y < 0 )
         {
            f[i]=3.*acos(-1)/2.;
         }
         else
         {
            f[i]=atan2(y[i],x[i]);
         }
      }
      if(f[i] < 0)
         f[i]+=2*acos(-1);
   }
}

void density_cont_val(double* reorthog,double* imorthog,double *x,double*y,double*z,double*kx,double*ky,double*kz,int nx,int ny,int nz,int nk,double* dens_val)
{
   int j=0;
   int i=0;

   for( i=0;i!=nx*ny*nz;i++)
   {
      for( j=0;j!=nk;j++)
      {
         dens_val[i*nk+j]=pow(cos(x[i]*kx[j]+y[i]*ky[j]+z[i]*kz[j])-reorthog[i*nk+j],2)+pow(sin(x[i]*kx[j]+y[i]*ky[j]+z[i]*kz[j])-imorthog[i*nk+j],2);
      }
   }
   return ;
}

void build_transition_density_matrix(int n_states_neut,int n_closed,int n_occ,int ci_size_neut,int n_elec_neut,double *ci_vector,int *mos_vector,int* spin_vector,double *tran_den_mat_mo)
{

   bool test(0);
   bool test2(0);
   bool test3(0);
   int q(0);
   int p(0);
   double sum(0);
   double det_val;

    for (int i=0; i!=n_states_neut; i++)//ELECTRONIC STATE N
    {
        for (int j=0; j!=n_states_neut; j++)//ELECTRONIC STATE K
        {
           std::cout<<" density between states "<<i<<" and "<<j<<std::endl;
           sum=0;
         for(int k=0;k!=(n_closed+n_occ);k++)
         {
            for(int kp=0;kp!=n_closed+n_occ;kp++)
            {
               //tran_den_mat_mo[n_states_neut*i+j][(n_occ+n_closed)*k+kp] = 0;
               tran_den_mat_mo[i*n_states_neut*(n_occ+n_closed)*(n_occ+n_closed)+j*(n_occ+n_closed)*(n_occ+n_closed)+k*(n_occ+n_closed)+kp] = 0;
               for(int m=0;m!=ci_size_neut;m++)
               {
                  for(int n=0;n!=ci_size_neut;n++)
                  {

                     det_val=build_reduced_determinant(k,kp,n_elec_neut,n_closed,n_occ,&mos_vector[m*n_elec_neut],&mos_vector[n_elec_neut*n],&spin_vector[n_elec_neut*m],&spin_vector[n_elec_neut*n]);

                     //tran_den_mat_mo[i*n_states_neut+j][k*(n_occ+n_closed)+kp]+=ci_vec_neut[0][(n_elec_neut+n_states_neut)*(m)+n_elec_neut+i]*ci_vec_neut[0][(n_elec_neut+n_states_neut)*(n)+n_elec_neut+j]*det_val;
                     tran_den_mat_mo[i*n_states_neut*(n_occ+n_closed)*(n_occ+n_closed)+j*(n_occ+n_closed)*(n_occ+n_closed)+k*(n_occ+n_closed)+kp]+=ci_vector[n_states_neut*m+i]*ci_vector[n_states_neut*n+j]*det_val;
                   }
                 }
//               if(k==kp)
//               {
//                 sum+=tran_den_mat_mo[i*n_states_neut*(n_occ+n_closed)*(n_occ+n_closed)+j*(n_occ+n_closed)*(n_occ+n_closed)+k*(n_occ+n_closed)+kp];
               //  std::cout<<" from orbital "<<k<<" and from orbital "<<kp<<":"<<tran_den_mat_mo[i*n_states_neut+j][k*(n_occ+n_closed)+kp]<<std::endl;
//               }
//               std::cout<<std::setprecision(8)<<"trdm val "<<tran_den_mat_mo[i*n_states_neut+j][k*(n_occ+n_closed)+kp]<<std::endl;
              }
           }
//         std::cout<<"SUM = "<<sum<<std::endl;
        }
    }
}

double build_reduced_determinant( int ai,int aj,int n_elec,int n_closed,int n_occ,double* mo_vector_1,double* mo_vector_2,double *spin_vector_1,double *spin_vector_2)
{
   /* Given the vectors containing the mo labels and the spin labels of the electrons, this routine builds a slater determinant from which one electron contained in the mo's i and j have been removed   !!!! ONLY FOR SINGLET AND SIMPLE EXCITATION
   */

   bool test2(0);
   bool test3(0);
   int spin(0);
   double temp(0);

   int new_vector_1[(n_occ+n_closed)];
   int new_vector_2[(n_occ+n_closed)];

   double prefactor(1);

   for(int k=0;k!=(n_occ+n_closed);k++)
   {
      new_vector_1[k]=0;
      new_vector_2[k]=0;
   }

   for(int e=0;e!=n_elec;e++)
   {
      new_vector_1[int(mo_vector_1[e])]+=1;
      new_vector_2[int(mo_vector_2[e])]+=1;
   }
   /*
   std::cout<<"Taking electron from orbitals "<<ai<<","<<aj<<std::endl;
   for(int k=0;k!=(n_occ+n_closed);k++)
   {
      std::cout<<new_vector_1[k]<<" ";
   }std::cout<<std::endl;

   for(int k=0;k!=(n_occ+n_closed);k++)
   {
      std::cout<<new_vector_2[k]<<" ";
   }std::cout<<std::endl;
   */
   prefactor=sqrt(double(new_vector_1[ai]))*sqrt(double(new_vector_2[aj]));
   new_vector_1[ai]--;
   new_vector_2[aj]--;

   for(int k=0;k!=(n_occ+n_closed);k++)
   {
//      std::cout<<new_vector_1[k]<<std::endl;
      if(new_vector_1[k]!=new_vector_2[k])
         return 0;
   }

  // if(prefactor != 0 )
  //    std::cout<<prefactor<<std::endl;
   return prefactor;
}
