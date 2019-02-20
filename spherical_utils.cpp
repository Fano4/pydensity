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

