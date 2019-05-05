'''
This utility reads an h5 file produced by the PWAPIC code and .wvpck files generated by the Wavepack code and generates a cube file containing the time-dependent electronic density corresponding to the molecule whose dynamics has been computed using Wavepack.
'''

import numpy as np
import math
import h5py as h5
import spherical_util as spher
import fortranformat as ff
import sys
import scipy as sp
from scipy import special

def intplushalf_gamma(n):
    return (np.arccos(-1)**0.5)*sp.special.factorial(2*n)/((4**n)*sp.special.factorial(n));

def mo_value(r,t,f,mo_index,nucl_index,nucl_coord,bas_fun_type,cont_num,cont_zeta,cont_coeff,lcao_num_array,lcao_coeff_array,angular):

    val=0

    lcao_num=lcao_num_array
    lcao_coeff=lcao_coeff_array[mo_index]
#    print(lcao_coeff_array.shape,lcao_coeff.shape)

    r2=np.outer(r,np.ones(max(cont_num)))

    coeff=np.sum(cont_coeff*np.exp(-cont_zeta*r2**2),axis=1)*angular*r**bas_fun_type.T[0]

    val=np.dot(lcao_coeff,coeff)

    return val

def cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,filename,array_val,nucl_coord):
    file=open(filename,"w")
    file.write("Cube file written using python density utility \n")
    file.write("LiH electronic density \n")
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nucl_coord.shape[0],xmin,ymin,zmin))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nx,dx,0.000000,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(ny,0.000000,dy,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nz,0.000000,0.000000,dz))

    for i in np.arange(0,nucl_coord.shape[0]):
        file.write('{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f} \n'.format(1,1.000000,nucl_coord[i][0],nucl_coord[i][1],nucl_coord[i][2]))

    lineformat=ff.FortranRecordWriter('(1E13.5)')
    for ix in np.arange(0,nx):
        for iy in np.arange(0,ny):
            for iz in np.arange(0,nz):
#                file.write('{:13.5E}'.format(array_val[ix][iy][iz]))
                file.write(lineformat.write([array_val[ix,iy,iz]]))
                if( (iz + 1) % 6 == 0 and iz != 0):
                    file.write('\n')
            file.write('\n')

    file.close()

'''
def main(MO_index, spin_state):

    #H5 LOCATION ON SASHA GREY
    fn = '/home/alessio/config/Stephan/zNorbornadiene_P005-000_P020-000_P124-190.rasscf.h5'


    import quantumpropagator as qp

    ci_coefficients = all_things['CI_VECTORS'].flatten() # I need this vector flattened

    # PARSE THINGS
    all_things = qp.readWholeH5toDict(fn)

    MO_OCCUPATIONS = all_things['MO_OCCUPATIONS']


    n_mo = MO_OCCUPATIONS[np.nonzero(MO_OCCUPATIONS)].size
    nucl_index = all_things['BASIS_FUNCTION_IDS'][:,0] # THIS NEEDS TO BE np.array([])
    nucl_coord = all_things['CENTER_COORDINATES'] # THIS NEEDS TO BE np.array([])
    bas_fun_type = all_things['BASIS_FUNCTION_IDS'][:,1:3] # THIS NEEDS TO BE np.array([])
    n_states_neut = all_things['ROOT_ENERGIES'].size # this is ok


    # cont_num =
    # cont_zeta =
    # cont_coeff =

    lcao_coeff_array = all_things['MO_VECTORS']
    lcao_num_array = all_things['MO_VECTORS'].size

    # we might be able to compute this
    # tran_den_mat =
'''
def string_active_space_transformer(fn, inactive):
    '''
    from a grepped molcas file to the right vector
    fn :: filepath <- file with a list of  in the right order 222ud000
    '''
    with open(fn,'r') as f:
        content = f.readlines()

    inactive_string = '2'*inactive
    strings = [ inactive_string + x.replace('\n','') for x in content if x != '\n' ]

    print('\n\nThis file contains {} cases\n\n'.format(len(strings)))
    a = [ from_string_to_vector(x) for x in strings ]
    MO_index = []
    spin_state = []
    for x in strings:
        MO, spin = from_string_to_vector(x)
        MO_index = MO_index + MO
        spin_state = spin_state + spin
    MO_index_out = np.array(MO_index,dtype=np.int32) 
    spin_state_out = np.array(spin_state,dtype=np.int32)
    return (MO_index_out,spin_state_out)

def from_string_to_vector(strin):
    '''
    from a string of occupation it get the two vectors MO_index and spin_state
    strin :: String <- '22222222222222222222222u000d002'
    '''
    MO_index = []
    spin_state = []
    for a,i in enumerate(strin): # for each character of this string
        if i == '2':
            MO_index.append(a)
            spin_state.append(0)
            MO_index.append(a)
            spin_state.append(1)
        elif i == 'u':
            MO_index.append(a)
            spin_state.append(0)
        elif i == 'd':
            MO_index.append(a)
            spin_state.append(1)
    return(MO_index,spin_state)



def Molcas_to_Wavepack(molcas_h5file, up_down_file, inactive):
    '''
    This is intended to convert Molcas generated hdf5 file data (rasscf) to the format used by pydensity.
    argument: h5 file object
    returns the set of data arrays
    '''
    print("Getting CI vectors data")
    _ , ci_length = molcas_h5file['CI_VECTORS'].shape
    ci_coefficients = np.asarray(molcas_h5file['CI_VECTORS']).swapaxes(0,1).flatten() # I need this vector flattened
    MO_index, spin_state = string_active_space_transformer(up_down_file, inactive)

    # PARSE THINGS
    print("Getting MO Occupation and electron data")

    MO_OCCUPATIONS = np.asarray(molcas_h5file['MO_OCCUPATIONS'])
    n_electrons = int(sum(MO_OCCUPATIONS))


    n_mo = MO_OCCUPATIONS[np.nonzero(MO_OCCUPATIONS)].size
    print("Getting nuclear data")
    nucl_index = np.asarray(molcas_h5file['BASIS_FUNCTION_IDS'][:,0]) # THIS NEEDS TO BE np.array([])
    nucl_coord = np.asarray(molcas_h5file['CENTER_COORDINATES']) # THIS NEEDS TO BE np.array([])
    print("Getting spherical harmonics data")
    bas_fun_type = np.asarray(molcas_h5file['BASIS_FUNCTION_IDS'][:,1:3],dtype=np.int32) # THIS NEEDS TO BE np.array([])
    n_states_neut = molcas_h5file['ROOT_ENERGIES'].size # this is ok

    '''
    Now we need to compute the transition density matrix. transition density matrix requires the ci vector,
    the array containing the occupied mos and the spin state vector.
    '''
    n_active = n_mo - inactive

    tran_den_mat = np.zeros(n_mo*n_mo*n_states_neut*n_states_neut)
    print("Entering TDM: building routine")

    spher.pbuild_transition_density_matrix(
            n_states_neut,
            inactive,
            n_active,
            ci_length,
            n_electrons,
            ci_coefficients,
            MO_index,
            spin_state,
            tran_den_mat
            )

    print("TDM Built!")

    np.save("testing_tdm",tran_den_mat)
    tran_den_mat=tran_den_mat.reshape((n_states_neut*n_states_neut,n_mo*n_mo))
    '''
    We want to evaluate the contraction numbers for every basis function.
    each basis function is described by CENTER - SHELL - L - ML
    for each of them, the number of contractions is given by the number
    of corresponding primitives with the same CENTER - SHELL - L

    we just need then to compute this number

    We also track the id of every primitve involved in the basis function array in cont_id

    '''
    cont_num = np.zeros((molcas_h5file['BASIS_FUNCTION_IDS']).shape[0],dtype=int)
    cont_id=[]

    for i in np.arange(0,np.asarray(molcas_h5file['BASIS_FUNCTION_IDS']).shape[0]):
            for j in np.arange(0,np.asarray(molcas_h5file['PRIMITIVE_IDS']).shape[0] ):
                if( molcas_h5file['BASIS_FUNCTION_IDS'][i][0] == molcas_h5file['PRIMITIVE_IDS'][j][0] and molcas_h5file['BASIS_FUNCTION_IDS'][i][1] == molcas_h5file['PRIMITIVE_IDS'][j][2] and molcas_h5file['BASIS_FUNCTION_IDS'][i][2] == molcas_h5file['PRIMITIVE_IDS'][j][1]):
                    cont_num[i]+=1
            cont_id.append(np.zeros(cont_num[i],dtype=int))
            tot=0
            for j in np.arange(0,np.asarray(molcas_h5file['PRIMITIVE_IDS']).shape[0] ):
                if( molcas_h5file['BASIS_FUNCTION_IDS'][i][0] == molcas_h5file['PRIMITIVE_IDS'][j][0] and molcas_h5file['BASIS_FUNCTION_IDS'][i][1] == molcas_h5file['PRIMITIVE_IDS'][j][2] and molcas_h5file['BASIS_FUNCTION_IDS'][i][2] == molcas_h5file['PRIMITIVE_IDS'][j][1]):
                    cont_id[i][tot]=j
                    tot=tot+1

    '''
    The maximum contraction number gives us the size of dimension 1 for the arrays cont_zeta and cont_coeff
    '''
    cont_zeta = np.zeros((molcas_h5file['BASIS_FUNCTION_IDS'].shape[0],cont_num.max()))
    cont_coeff = np.zeros((molcas_h5file['BASIS_FUNCTION_IDS'].shape[0],cont_num.max()))

    '''
    We need to fill cont_zeta and cont_coeff now. These numbers are given, for every contraction in the PRIMITIVES array.
    The difficulty is that we should fill the arrays with the values resptective to the corresponding primitive.
    We assume that the primitives are used in the same order as they are called in the BASIS_FUNCTION_IDS array
    '''
    for i in np.arange(0,(molcas_h5file['BASIS_FUNCTION_IDS']).shape[0]):
        for j in np.arange(0,cont_num[i]):
            cont_zeta[i,j]=np.asarray(molcas_h5file['PRIMITIVES'])[cont_id[i][j],0]
            cont_coeff[i,j]=np.asarray(molcas_h5file['PRIMITIVES'])[cont_id[i][j],1]
            cont_coeff[i,j]=cont_coeff[i,j]/((0.5*intplushalf_gamma(bas_fun_type[i,0]+1)/(2*cont_zeta[i,j])**(bas_fun_type[i,0]+1.5))**0.5);
#    cont_zeta=np.asarray(molcas_h5file['PRIMITIVES'])[:,0]
#    cont_coeff=np.asarray(molcas_h5file['PRIMITIVES'])[:,1]

    lcao_coeff_array = np.asarray(molcas_h5file['MO_VECTORS']).reshape((molcas_h5file['BASIS_FUNCTION_IDS'].shape[0],molcas_h5file['BASIS_FUNCTION_IDS'].shape[0]))
    lcao_num_array = lcao_coeff_array.shape[0]

    return n_mo,nucl_index,nucl_coord,bas_fun_type,n_states_neut,tran_den_mat,cont_num,cont_zeta,cont_coeff,lcao_num_array,lcao_coeff_array





if __name__ == "__main__" :
    #fn = '/home/alessio/config/Stephan/up_down' # ON SASHA GREY
    fn = '/Users/stephan/dox/Acu-Stephan/up_down' # ON STEPH MACBOOK
    inactive = 23 # the inactive orbitals
    molcas_h5file = h5.File('/Users/stephan/dox/Acu-Stephan/zNorbornadiene_P005-000_P020-000_P124-190.rasscf.h5','r') # ON STEPH MACBOOK
    #molcas_h5file = h5.File('/home/alessio/config/Stephan/zNorbornadiene_P005-000_P020-000_P124-190.rasscf.h5','r') # ON SASHA
    print("Entering Molcas To Wavepack Routine")
    n_mo,nucl_index,nucl_coord,bas_fun_type,n_states_neut,tran_den_mat,cont_num,cont_zeta,cont_coeff,lcao_num_array,lcao_coeff_array=Molcas_to_Wavepack(molcas_h5file,fn,inactive)
    molcas_h5file.close()
    print("Molcas To Wavepack Routine Done!")

    nes=n_states_neut
    #WVPCK_DATA REPRESENT THE AMPLITUDE ON THE ELECTRONIC STATES FOR THE CONSIDERED TIME. IT IS AN ARRAY WITH SIZE NES IS SINGLE POINT, AND A MATRIX WITH SIZE NES X NGEOM IF GEOMETRY DEPENDENT

    wvpck_data=np.zeros(nes)
    wvpck_data[0]=1;
    
    #TDM IS THE TRANSITION DENSITY MATRIX IN THE BASIS OF MO'S, AVERAGRED OVER THE POPULATIONS IN THE EXCITED STATES.
    tdm=np.zeros((n_mo,n_mo))

    for ies in np.arange(0,nes):
        tdm=tdm+abs(wvpck_data[ies])**2*tran_den_mat[(ies)*n_states_neut+(ies)].reshape((n_mo,n_mo))
        for jes in np.arange(ies+1,nes):
#            print(ies,jes,"are the es")
            tdm=tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*n_states_neut+(jes)].reshape((n_mo,n_mo))

    #ONCE YOU COMPUTED THE AVERGARED TDM, YOU JUST NEED TO EVALUATE THE DENSITY
    xmin=-10.0
    ymin=-10.0
    zmin=-10.0
    dx=0.416
    dy=0.416
    dz=0.416
    nx=64
    ny=64
    nz=64
    lcao_num=lcao_num_array
    cube_array=np.zeros((nx,ny,nz))
    for ix in np.arange(0,nx):
#            print(ix,"/",nx)
        x=xmin+ix*dx
        for iy in np.arange(0,ny):
            y=ymin+iy*dy
            for iz in np.arange(0,nz):
                z=zmin+iz*dz

                val=0
#                lcao_num=len(lcao_num_array[0])
#                lcao_num=molcas_h5file['BASIS_FUNCTION_IDS'].shape
                coord=np.array([x,y,z])
                coordp=coord-nucl_coord[nucl_index-1]
                rp=np.zeros(lcao_num)
                tp=np.zeros(lcao_num)
                fp=np.zeros(lcao_num)
                xp=coordp.T[0]
                yp=coordp.T[1]
                zp=coordp.T[2]
                xp=xp.copy(order='C')
                yp=yp.copy(order='C')
                zp=zp.copy(order='C')
                rp=rp.copy(order='C')
                tp=tp.copy(order='C')
                fp=fp.copy(order='C')

                spher.pcart_to_spher(xp,yp,zp,rp,tp,fp)
                r,t,f=rp.T,tp.T,fp.T
                angular=np.zeros(lcao_num)
                l=bas_fun_type.T[0]
                ml=bas_fun_type.T[1]
                l=l.copy(order='C')
                ml=ml.copy(order='C')
                angular=angular.copy(order='C')
                spher.pspher_harmo(t,f,l,ml,angular)
    
                i=np.arange(0,n_mo)
                phii=mo_value(r,t,f,i,nucl_index,nucl_coord,bas_fun_type,cont_num,cont_zeta,cont_coeff,lcao_num_array,lcao_coeff_array,angular)

                val=np.dot(phii.T,np.matmul(tdm,phii))
    
                cube_array[ix][iy][iz]=cube_array[ix][iy][iz]+val
#                    print(val,ix,iy,iz)
    
    target_file="/Users/stephan/Desktop/density_test_alessio_.cub"
    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,cube_array,nucl_coord)




'''
def ohter():
   ' ''
    This is the old Main of Stephan, that I try to replace with a MAin that works with Molcas
    ' ''
    dataloc = "/Users/Stephan/Desktop/pydensity/LiH_density_grid.h5"

    val = 0
    data = h5.File(dataloc,'r')

    n_mo = data['/electronic_struct_param/n_mo_closed'][0]+data['/electronic_struct_param/n_mo_occ'][0]
    nucl_index = np.asarray(data['/basis_set_info/nucleus_basis_func'])
    nucl_coord = np.asarray(data['/nuclear_coord/nucl_cartesian_array'])
    bas_fun_type = np.asarray(data['/basis_set_info/basis_func_type'])
    cont_num = np.asarray(data['/basis_set_info/contraction_number'])
    cont_zeta = np.asarray(data['/basis_set_info/contraction_zeta'])
    cont_coeff = np.asarray(data['/basis_set_info/contraction_coeff'])
    lcao_coeff_array = np.asarray(data['/lcao_coeff/lcao_mo_coeff'])
    lcao_num_array = np.asarray(np.asarray(data['/lcao_coeff/lcao_mo_coeff']))
    tran_den_mat = np.asarray(data['/lcao_coeff/tran_den_mat_mo'])
    n_states_neut = data['/electronic_struct_param/n_states_neut'][0]
    data.close()

    xmin=-10.0
    ymin=-10.0
    zmin=-10.0
    dx=0.416
    dy=0.416
    dz=0.416
    nx=64
    ny=64
    nz=64
    cube_array=np.zeros((nx,ny,nz))
    for irp in np.arange(0,gsize):
        ir=int(irp*tgsize/gsize)
        print("Computing density at position",irp)
        tdm=np.zeros((n_mo,n_mo))
        ##WAVEPACK DATA TREATMENT
        for ies in np.arange(0,nes):
            tdm=tdm+abs(wvpck_data[ies][irp])**2*tran_den_mat[ir][(ies+es0)*n_states_neut+(ies+es0)].reshape((n_mo,n_mo))
            for jes in np.arange(ies+1,nes):
#                print(ies,jes,"are the es")
                tdm=tdm+2*((wvpck_data[ies][irp]*wvpck_data[jes][irp].conjugate()).real)*tran_den_mat[ir][(ies+es0)*n_states_neut+(jes+es0)].reshape((n_mo,n_mo))
#                            tdm=tran_den_mat[ir][state_1_index*n_states_neut+state_2_index].reshape((n_mo,n_mo))
        ##WAVEPACK DATA TREATMENT
        for ix in np.arange(0,nx):
#            print(ix,"/",nx)
            x=xmin+ix*dx
            for iy in np.arange(0,ny):
                y=ymin+iy*dy
                for iz in np.arange(0,nz):
                    z=zmin+iz*dz

                    val=0
                    lcao_num=len(lcao_num_array[ir][0])
                    coord=np.array([x,y,z])
                    coordp=coord-nucl_coord[ir][nucl_index-1]
                    rp=np.zeros(lcao_num)
                    tp=np.zeros(lcao_num)
                    fp=np.zeros(lcao_num)
                    xp=coordp.T[0]
                    yp=coordp.T[1]
                    zp=coordp.T[2]
                    xp=xp.copy(order='C')
                    yp=yp.copy(order='C')
                    zp=zp.copy(order='C')
                    rp=rp.copy(order='C')
                    tp=tp.copy(order='C')
                    fp=fp.copy(order='C')

                    spher.pcart_to_spher(xp,yp,zp,rp,tp,fp)
                    r,t,f=rp.T,tp.T,fp.T
                    angular=np.zeros(lcao_num)
                    l=bas_fun_type.T[0]
                    ml=bas_fun_type.T[1]
                    l=l.copy(order='C')
                    ml=ml.copy(order='C')
                    angular=angular.copy(order='C')
                    spher.pspher_harmo(t,f,l,ml,angular)

                    i=np.arange(0,n_mo)
                    phii=mo_value(r,t,f,i,nucl_index,nucl_coord[ir],bas_fun_type,cont_num,cont_zeta,cont_coeff,lcao_num_array[ir],lcao_coeff_array[ir],angular)
                    val=np.dot(phii.T,np.matmul(tdm,phii))


                    cube_array[ix][iy][iz]=cube_array[ix][iy][iz]+val

    target_file="/Users/stephan/Desktop/density_time_"+str(time_index)+"_recollision.cub"
    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,dataloc,cube_array)
'''
