#import math
import os
import glob
import pickle
import sys
from argparse import ArgumentParser
import multiprocessing
import h5py as h5
import spherical_util as spher
import fortranformat as ff
import scipy as sp
import yaml
import numpy as np
#from scipy import special
from joblib import Parallel, delayed
from tqdm import tqdm
from orbitals_molcas import Orbitals

def writeH5fileDict(fn, dictionary):
    '''
    writes a h5 file, dictionary edition
    fn :: String <- the output path
    dictionary :: {Values}  <- the content of Hdf5
    '''
    with h5.File(fn, 'w') as hf:
        for key in dictionary.keys():
            hf.create_dataset(key, data=dictionary[key])

def fromBohToAng(n):
    ''' From Bohr to Angstrom conversion - n :: Double '''
    return (n * 0.529177249)

def saveTraj(arrayTraj, labels, filename, convert=None):
    '''
    given a numpy array of multiple coordinates, it prints the concatenated xyz file
    arrayTraj :: np.array(ncoord,natom,3)    <- the coordinates
    labels :: [String] <- ['C', 'H', 'Cl']
    filename :: String <- filepath
    convert :: Bool <- it tells if you need to convert from Boh to Ang (default True)
    '''
    convert = convert or False
    (ncoord,natom,_) = arrayTraj.shape
    fn = filename + '.xyz'
    string = ''
    for geo in range(ncoord):
        string += str(natom) + '\n\n'
        for i in range(natom):
            if convert:
                string += "   ".join([labels[i]] +
                        ['{:10.6f}'.format(fromBohToAng(num)) for num
                    in arrayTraj[geo,i]]) + '\n'
            else:
                string += "   ".join([labels[i]] +
                        ['{:10.6f}'.format(num) for num
                    in arrayTraj[geo,i]]) + '\n'

    with open(fn, "w") as myfile:
        myfile.write(string)


def cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,filename,array_val,nucl_coord):
    file=open(filename,"w")
    file.write("Cube file written using python density utility \n")
    file.write("Norbornadiene electronic density \n")
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nucl_coord.shape[0],xmin,ymin,zmin))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nx,dx,0.000000,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(ny,0.000000,dy,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nz,0.000000,0.000000,dz))

    normb_atom_type = [6,6,6,1,1,1,1,6,6,6,6,1,1,1,1]
    print('warning, atomtype for norbornadiene hardcoded in function cubegen')

    for i in np.arange(0,nucl_coord.shape[0]):
        file.write('{:5} {:11.6f} {:11.6f} {:11.6f} {:11.6f} \n'.format(normb_atom_type[i],1.000000,nucl_coord[i][0],nucl_coord[i][1],nucl_coord[i][2]))

    lineformat=ff.FortranRecordWriter('(1E13.5)')
    for ix in np.arange(0,nx):
        for iy in np.arange(0,ny):
            for iz in np.arange(0,nz):
                file.write(lineformat.write([array_val[ix,iy,iz]]))
                if( (iz + 1) % 6 == 0 and iz != 0):
                    file.write('\n')
            file.write('\n')

    file.close()


def string_active_space_transformer(fn, inactive):
    '''
    from a grepped molcas file to the right vector
    fn :: filepath <- file with a list of occupations in the right order 222ud000
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


def molcas_to_tran_den_mat(molcas_h5file, up_down_file, inactive, cut_states):
    '''
    This is intended to convert Molcas generated hdf5 file data (rasscf) to the format
    used by pydensity.

    molcas_h5file :: h5 file object
    up_down_file :: FilePath
    inactive :: Int <- number of inactive orbital in the casscf problem
    cut_states :: Int  <- number of states you want to consider

    returns the set of data arrays
    '''

    print("Getting CI vectors data")
    _, ci_length = molcas_h5file['CI_VECTORS'].shape
    ci_coefficients = np.asarray(
                 molcas_h5file['CI_VECTORS'][:cut_states]).swapaxes(0,1).flatten()
    MO_index, spin_state = string_active_space_transformer(up_down_file, inactive)

    # PARSE THINGS
    print("Getting MO Occupation and electron data")
    MO_OCCUPATIONS = np.asarray(molcas_h5file['MO_OCCUPATIONS'])
    n_electrons = int(sum(MO_OCCUPATIONS))
    n_mo = MO_OCCUPATIONS[np.nonzero(MO_OCCUPATIONS)].size
    n_states_neut = molcas_h5file['ROOT_ENERGIES'][:cut_states].size # this is ok

    '''
    Now we need to compute the transition density matrix. transition density
    matrix requires the ci vector,the array containing the occupied mos and
    the spin state vector.
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

    tran_den_mat = tran_den_mat.reshape((n_states_neut*n_states_neut, n_mo*n_mo))
    return tran_den_mat


def get_TDM(molcas_h5file_path,updown_file,inactive,cut_states):
    '''
    molcas_h5file_path :: Filepath <- h5 of Molcas
    updown_file :: Filepath <- input up and down file
    inactive :: Int <- inactive orbitals
    cut_states :: Int <- number of states
    '''

    # if TDM is in file, read file, otherwise create a new h5 file with transition density matrix.

    molcas_h5file = h5.File(molcas_h5file_path, 'r')
    if 'TDM' in molcas_h5file.keys():
        # just read the file
        transition_density_matrix = molcas_h5file['TDM']
        print('File {} already contains a TDM'.format(molcas_h5file_path))
    else:
        new_file_name = os.path.splitext(molcas_h5file_path)[0] + '.TDM.h5'
        if os.path.isfile(new_file_name):
            print('File {} already exists'.format(new_file_name))
        else:
            # calculate the 
            transition_density_matrix = molcas_to_tran_den_mat(molcas_h5file,
                    updown_file, inactive, cut_states)
            # HERE HERE you want to write the new h5 with TDM
            # I remove the extension from h5 and put the new one
            with h5.File(new_file_name, 'w') as output_h5_file:
                output_h5_file.create_dataset('TDM',data=transition_density_matrix)
                output_h5_file.close()
    molcas_h5file.close()

def calculate_between_carbons(wvpck_data,molcas_h5file_path,indexes,data):
    '''
    This returns the electronic density summed up on all the points in between carbons
    wvpck_data :: np.array(Double) <- the 1d singular geom multielectronic state wf.
    molcas_h5file_path :: String <- FilePath
    data :: Dictionary
    indexes :: np.array(Int) <-the indexes of the FLATTEN ARRAY in the cartesian cube.
    no_s0 :: Bool <- take out S0 from the computation
    '''
    no_s0 = data['no_s0']
    n_mo, nes, n_core = 31, 8, 23
    tdm_file = h5.File(os.path.splitext(molcas_h5file_path)[0] + '.TDM.h5', 'r')
    tran_den_mat = tdm_file['TDM']
    tdm = np.zeros((n_mo,n_mo))

    # u wanna take out S0?
    if no_s0:
        begin_here = 1
    else:
        begin_here = 0

    for ies in np.arange(begin_here,nes):
        tdm = tdm+abs(wvpck_data[ies])**2*tran_den_mat[(ies)*nes+(ies)].reshape((n_mo,n_mo))
        for jes in np.arange(ies+1,nes):
            #print(ies,jes,"are the es")
            tdm = tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*nes+(jes)].reshape((n_mo,n_mo))

    xmin, ymin, zmin = data['mins']
    nx,ny,nz = data['num_points']
    x = np.linspace(xmin,-xmin,nx)
    y = np.linspace(ymin,-ymin,ny)
    z = np.linspace(zmin,-zmin,nz)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]
    # why meshgrid is always like this? I do not see why this swap (BAC) is always present
    B,A,C = np.meshgrid(x,y,z)
    orbital_object = Orbitals(molcas_h5file_path,'hdf5')
    number_of_points, = indexes.shape
    first = A.flatten()
    second = B.flatten()
    third = C.flatten()
    ffirst = first[indexes]
    ssecond = second[indexes]
    tthird = third[indexes]
    phii = np.empty((n_mo,number_of_points))

    for i in range(n_mo):
        # the mo method calculates the MO given space orbitals
        phii[i,:] = orbital_object.mo(i,first[indexes],second[indexes],third[indexes])

    cube_array = np.zeros(number_of_points)

    if data['take_core_out']:

        for i in range(n_core,n_mo):
            for j in range(n_core,n_mo):
                cube_array += phii[i] * phii[j] * tdm[i,j]

    else:

        for i in range(n_mo):
            for j in range(n_mo):
                cube_array += phii[i] * phii[j] * tdm[i,j]

    final = np.sum(cube_array) * (dx*dy*dz)

    return final



def creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,take_only_active=None):
    '''
    It returns the cube at this geometry.
    wvpck_data :: np.array(Double) <- the 1d singular geom multielectronic state wf.
    data :: Dictionary
    '''
    n_mo, nes, n_core = 31, 8, 23
    tdm_file = h5.File(os.path.splitext(molcas_h5file_path)[0] + '.TDM.h5', 'r')
    tran_den_mat = tdm_file['TDM']
    '''
    tdm is the transition density matrix in the basis of mo's, averaged over the populations
    in the excited states.
    '''
    tdm = np.zeros((n_mo,n_mo))

    #print('\n1) Calculating tdm')

    for ies in np.arange(0,nes):
        tdm = tdm+abs(wvpck_data[ies])**2*tran_den_mat[(ies)*nes+(ies)].reshape((n_mo,n_mo))
        for jes in np.arange(ies+1,nes):
            #print(ies,jes,"are the es")
            tdm = tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*nes+(jes)].reshape((n_mo,n_mo))

     # HERE HERE IMPLEMENT SINGLE ELEMENT
     #ies = 0
     #jes = 5
     #tdm = tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*nes+(jes)].reshape((n_mo,n_mo))


    '''
    once you computed the averaged tdm, you just need to evaluate the density
    this is a box centered in the origin 0,0,0
    '''

    xmin, ymin, zmin = data['mins']
    nx,ny,nz = data['num_points']
    x = np.linspace(xmin,-xmin,nx)
    y = np.linspace(ymin,-ymin,ny)
    z = np.linspace(zmin,-zmin,nz)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]
    B,A,C = np.meshgrid(x,y,z)
    phii = np.empty((n_mo,nx*ny*nz))
    orbital_object = Orbitals(molcas_h5file_path,'hdf5')

    #print('2) Calculating MO')

    #for i in tqdm(range(n_mo)):
    for i in range(n_mo):
        # the mo method calculates the MO given space orbitals
        phii[i,:] = orbital_object.mo(i,A.flatten(),B.flatten(),C.flatten())

    cube_array = np.zeros(nx*ny*nz)

    take_only_active = take_only_active or False

    #print('3) Calculating density\n')

    if take_only_active:

        for i in range(n_core,n_mo):
            for j in range(n_core,n_mo):
                cube_array += phii[i] * phii[j] * tdm[i,j]

    else:

        for i in range(n_mo):
            for j in range(n_mo):
                cube_array += phii[i] * phii[j] * tdm[i,j]

    return cube_array


def create_full_list_of_labels_numpy(list_labels):
    '''
    little helper, from the hardcoded list to the full list of expected files
    this keep the shape of the wavefunction
    '''
    ps = list_labels['phis_lab'].split(' ')
    gs = list_labels['gams_lab'].split(' ')
    ts = list_labels['thes_lab'].split(' ')
    all_lab_in_shape = np.empty((len(ps),len(gs),len(ts)),dtype=object)
    for p,pL in enumerate(ps):
        for g,gL in enumerate(gs):
            for t,tL in enumerate(ts):
                name = 'zNorbornadiene_{}_{}_{}'.format(pL,gL,tL)
                all_lab_in_shape[p,g,t] = name

    return(all_lab_in_shape)


def create_full_list_of_labels(list_labels):
    '''
    little helper, from the hardcoded list to the full list of expected files
    '''
    all_lab = []
    for p in list_labels['phis_lab'].split(' '):
        for g in list_labels['gams_lab'].split(' '):
            for t in list_labels['thes_lab'].split(' '):
                name = 'zNorbornadiene_{}_{}_{}'.format(p,g,t)
                all_lab.append(name)

    return(all_lab)


def process_single_file(full_path_local,updown_file,inactive,cut_states):
    '''
    Function wrapped to be parallel
    get_all_data is an empty IO() function (void in c++)
    single :: the looping function
    updown_file,inactive,cut_states :: actual parameters
                                       of get_all_data function
    '''
    full_tuple = get_all_data(full_path_local,updown_file,inactive,cut_states)
    return (full_tuple)


def abs2(x):
    '''
    x :: complex
    This is a reimplementation of the abs value for complex numbers
    '''
    return x.real**2 + x.imag**2


def read_cube(filename):
    '''
    from cube to dictionary
    filename :: String <- Filepath
    '''
    cube = {}
    transform = np.eye(4)
    with open(filename, 'rb') as f:
        f.readline()
        # Read title and grid origin
        title = str(f.readline().decode('ascii')).strip()
        n, xmin, ymin, zmin = f.readline().split()
        num = int(n)
        n, x, y, z = f.readline().split()
        ngridx = int(n)
        dx = float(x)
        translate = np.array([float(x), float(y), float(z)])
        transform[0,0] = float(x)
        transform[1,0] = float(y)
        transform[2,0] = float(z)
        transform[0,3] = translate[0]
        n, x, y, z = f.readline().split()
        ngridy = int(n)
        dy = float(y)
        transform[0,1] = float(x)
        transform[1,1] = float(y)
        transform[2,1] = float(z)
        transform[1,3] = translate[1]
        n, x, y, z = f.readline().split()
        ngridz = int(n)
        dz = float(z)
        transform[0,2] = float(x)
        transform[1,2] = float(y)
        transform[2,2] = float(z)
        transform[2,3] = translate[2]
        centers = []
        for i in range(abs(num)):
            q, _, x, y, z = str(f.readline().decode('ascii')).split()
            centers.append({'name':'{0}'.format(i), 'Z':int(q), 'xyz':np.array([float(x), float(y), float(z)])})
        cube['centers'] = centers
        rest_of_lines = f.readlines()
        list_of_list_of_floats = [ [ float(x) for x in a.split() ] for a in rest_of_lines ]
        cube['grid'] = np.array([y for x in list_of_list_of_floats for y in x])
        cube['transform'] = transform
        cube['natoms'] = num
        cube['mins'] = [float(xmin), float(ymin), float(zmin)]
        cube['ds'] = [dx, dy, dz]
        cube['ngrids'] = [ ngridx, ngridy, ngridz ]
        return(cube)

def cube_sum_grid_points(path_cube):
    '''
    From the path of one cube, sum up all the elements (calculate the norm)
    '''
    cube = read_cube(path_cube)
    grid_points = cube['grid']
    dx,dy,dz = cube['ds']
    differential = dx * dy * dz
    print('The sum on this cube is {}.'.format(sum(grid_points)*differential))

def vmd_scriptString():
    return '''
display projection Orthographic
display depthcue off
color Display Background white
axes location Off

mol addrep 0
mol new {{{}.xyz}} type {{xyz}} first 0 last -1 step 1 waitfor 1
mol modcolor 0 0 ColorID 16
mol modstyle 0 0 Licorice 0.0200000 10.000000 10.000000

mol addrep 0
mol new {{{}.xyz}} type {{xyz}} first 0 last -1 step 1 waitfor 1
mol modcolor 0 1 ColorID 0
mol modstyle 0 1 VDW 0.020000 12.000000

mol addrep 0
mol new {{{}.xyz}} type {{xyz}} first 0 last -1 step 1 waitfor 1
mol modcolor 0 2 ColorID 1
mol modstyle 0 2 VDW 0.020000 12.000000

mol addrep 0
mol new {{{}}} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}
mol modstyle 0 3 Isosurface 0.001 0 0 0 1 1
mol modmaterial 0 3 Transparent
mol modcolor 0 3 ColorID 0
mol addrep 0
mol modstyle 1 3 Isosurface 0.001 0 0 0 1 1
mol modmaterial 1 3 Transparent
mol modcolor 1 3 ColorID 1

draw text {{0.0 0.0 -4.5}} "NewBond: {:3.5f}"
draw text {{0.0 0.0 -5.0}} "OldBond: {:3.5f}"

'''

def cube_single_bonds(path_cube, r_c):
    '''
    It calculates the single geometry "in between bonds of non overlapping cyls"
    path_cube :: String <- filepath
    r_c :: double <- radius of cylinder

    This function puts serious shame in my programming abilities.
    '''
    ###### points_in_nonoverlapping_cylinder(geom, r, q, kind)
    from create_gridlists import points_in_nonoverlapping_cylinder
    cube = read_cube(path_cube)
    geom = np.vstack([ x['xyz'] for x in cube['centers'] ])
    print(cube.keys())
    print('\nWarning, function cube_single_bonds is SEVERLY hardcoded\n')

    xmin, ymin, zmin = cube['mins']
    dx, dy, dz = cube['ds']
    nx, ny, nz = cube['ngrids']
    differential = dx * dy * dz
    x = np.linspace(xmin,-xmin,nx)
    y = np.linspace(ymin,-ymin,ny)
    z = np.linspace(zmin,-zmin,nz)

    B,A,C = np.meshgrid(x,y,z)
    list_of_points_in_3d = np.stack([A.flatten(),B.flatten(),C.flatten()]).T

    # from now on 1 is newbond and 2 is oldbond
    sel1 = np.where(points_in_nonoverlapping_cylinder(geom, r_c, list_of_points_in_3d, 'blue'))
    sel2 = np.where(points_in_nonoverlapping_cylinder(geom, r_c, list_of_points_in_3d, 'red'))

    list_1 = sel1[0] # single
    list_2 = sel2[0] # double

    cube_values = cube['grid']
    value_1 = sum(cube_values[list_1])*differential
    value_2 = sum(cube_values[list_2])*differential

    first_thing = fromBohToAng(list_of_points_in_3d[sel1])
    second_thing = fromBohToAng(list_of_points_in_3d[sel2])

    label = '{}_{}_non-overlapping'.format(path_cube, r_c)
    label1 = 'new_bond_{}'.format(label)
    label2 = 'old_bond_{}'.format(label)

    saveTraj(np.array([geom]),['C','C','C','H','H','H','H','C','C','C','C','H','H','H','H'], label, convert=True)
    saveTraj(np.array([first_thing]),['H']*len(first_thing), label1)
    saveTraj(np.array([second_thing]),['H']*len(second_thing), label2)

    vmd_script = vmd_scriptString()

    vmd_script_name = '{}.vmd'.format(label)
    with open(vmd_script_name, 'w') as vmds:
         vmds.write(vmd_script.format(label,label1,label2,path_cube,value_1,value_2))

    print('\nPlease type:\n\n vmd -e {} \n\n\n'.format(vmd_script_name))
    print('{} {} {}'.format(value_1,value_2,path_cube))


def cube_difference(path_cube_1, path_cube_2):
    '''
    From the path of two cubes, get the difference
    '''
    root_folder = os.path.dirname(os.path.abspath(path_cube_1))

    label1 = os.path.splitext(path_cube_1)[0]
    label2 = os.path.splitext(path_cube_2)[0]

    output_name = '{}_minus_{}.cube'.format(label1,label2)

    target_file = os.path.join(root_folder,output_name)

    cube1 = read_cube(path_cube_1)
    cube2 = read_cube(path_cube_2)

    xmin, ymin, zmin = cube2['mins']
    dx, dy, dz = cube2['ds']
    nx, ny, nz = cube2['ngrids']

    final_cube = cube1['grid'] - cube2['grid']

    natoms = cube2['natoms']
    centers = cube1['centers']
    nucl_coord = np.zeros((natoms,3))
    for i in range(natoms):
        nucl_coord[i] = centers[i]['xyz']

    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)
    print('File {} written.'.format(os.path.basename(target_file)))


def give_me_stats(time, wf, threshold):
    new_one = abs2(np.asarray(wf))
    pL, gL, tL, nstates = new_one.shape

    calc = 0
    calculate_this = np.zeros((pL,gL,tL), dtype=bool)
    mod_sum = 0
    for p in range(pL):
        for g in range(gL):
            for t in range(tL):
                if np.sum(new_one[p,g,t]) > threshold:
                    mod_sum += np.sum(new_one[p,g,t])
                    calc += 1
                    calculate_this[p,g,t] = True
    norm = np.linalg.norm(wf)
    stringZ = '\nWavefunction at {:6.2f} fs with {} Threshold:\nNumber of points: {:5}\nNorm {:5.2f}\nTotal Norm: {:5.2f}\naccuracy: {:5.2f}%\n\n'
    print(stringZ.format(time, threshold, calc, mod_sum, norm,mod_sum/norm*100))


def parallel_wf(single_file_wf,one_every,p,g,t,hms,file_list,h5file_folder,args,molcas_h5file_path,nucl_coord,data,wf_folder):
    '''
    this is a strange function. It takes a center and a wavefunction
    and creates the total density due to this center
    '''
    print('\n\nI am doing now {}'.format(single_file_wf))
    # cube data

    xmin, ymin, zmin = data['mins']
    nx, ny, nz = data['num_points']
    x = np.linspace(xmin,-xmin,nx)
    y = np.linspace(ymin,-ymin,ny)
    z = np.linspace(zmin,-zmin,nz)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]


    with h5.File(single_file_wf,'r') as wf_file:
        wf_int = wf_file['WF']
        time = wf_file['Time'][0]
        wf = wf_int[15:-15, 15:-15, 30:-30, :]

        hms_p, hms_g, hms_t = hms
        target_file = 'znorb-{}-{}_{}_{}_dis_{}-{}-{}_{:08.3f}.cube'.format(os.path.basename(wf_folder),p+15,g+15,t+30,hms_p, hms_g, hms_t, time)
        # if this particular cube exists already, do not calculate it
        if os.path.isfile(target_file):
           print('File {} already generated'.format(target_file))
        else:
            # in case you want to normalize
            #wvpck_data_not_normalized = np.ndarray.flatten(wf[p-hms:p+hms+1,g-hms:g+hms+1,t-hms:t+hms+1])
            #wvpck_data = wvpck_data_not_normalized / np.linalg.norm(wvpck_data_not_normalized)


            wvpck_data = np.ndarray.flatten(wf[p-hms_p:p+hms_p+1, g-hms_g:g+hms_g+1, t-hms_t:t+hms_t+1])
            file_to_be_processed = np.ndarray.flatten(file_list[p-hms_p:p+hms_p+1, g-hms_g:g+hms_g+1, t-hms_t:t+hms_t+1])
            file_list_abs = [ os.path.join(h5file_folder, single + '.rasscf.h5') for single in file_to_be_processed ]
            if args.active:
                final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,True)
            else:
                final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,False)
            cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)


def command_line_parser():
    '''
    this function deals with command line commands
    '''
    parser = ArgumentParser()
    parser.add_argument("-a", "--active",
                    action="store_true", default=False,
                    help="in single mode, switch active or all")
    parser.add_argument("-u", "--up_down",
                    dest="u",
                    type=str,
                    help="Up_down file")
    parser.add_argument("-t", "--tdm",
                    dest="t",
                    nargs='+',
                    help="A list of rasscf.h5 file to process to create the TDM.")
    parser.add_argument("-d", "--difference",
                    dest="d",
                    nargs='+',
                    help="Two files for which you want the differences")
    parser.add_argument("-i", "--input_multigeom_mode",
                    dest="i",
                    type=str,
                    help="The yml file to set up the geometries")
    parser.add_argument("-c", "--core",
                    dest="c",
                    type=int,
                    help="number of cores for the calculation")
    parser.add_argument("-s", "--single_file_mode",
                    dest="s",
                    type=str,
                    help="The single file path")
    parser.add_argument("-b", "--between-file",
                    dest="b",
                    type=str,
                    help="an yml file with 'Between atoms mode'")
    parser.add_argument("-w", "--wavefunction",
                    dest="w",
                    type=str,
                    help="Wavefunction stats")
    parser.add_argument("-n", "--norm_cube",
                    dest="n",
                    type=str,
                    help="This is to calculate the sum into a cube. if followed by pickle bond file, calculates the norm into bonds subgrids")
    parser.add_argument("-p", "--singlebetween",
                    dest="p",
                    nargs='+',
                    type=str,
                    help="This is the in between bonds for single geometry. It is the CUBE file followed by radius of non overlapping cylinder r_c")
    parser.add_argument("-e", "--e-was-free-letter",
                    dest="e",
                    type=str,
                    help="This creates actual cube of values for newbond and oldbond to be plotted with blender. Takes as argument an yml")
    parser.add_argument("-o", "--orbitals",
                    dest="o",
                    nargs='+',
                    help="Followed by the h5 file name and a list of integer (the orbitals you want to transform in cubes)")
    parser.add_argument("-f", "--follow",
                    dest="f",
                    type=str,
                    help="This is FOLLOW mode. Takes an yml as input. It follows the density on a single or a group of points along one MD.")

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def Main():
    print('warning, n_mo and nes hardcoded on function creating_cube_function_fro_nuclear_list')

    inactive = 23
    cut_states = 8

    args = command_line_parser()
    #print(args)

    # on MAC
    # updown_file = '/Users/stephan/dox/Acu-Stephan/up_down'

    # ON SASHA
    updown_file = '/home/alessio/config/Stephan/up_down'
    if args.u != None:
        updown_file = args.u

    if args.o != None:
        list_of_orbitals = [ int(x) for x in args.o[1:] ]
        name_file = args.o[0]
        print('\nOrbital mode\n{}: extracting orbs {}\n'.format(name_file, list_of_orbitals))
        molcas_h5file_path = os.path.abspath(name_file)
        single_file_data = get_TDM(molcas_h5file_path,updown_file,inactive,cut_states)

        data = { 'mins' : [-10.0,-10.0,-10.0],
                 #'num_points' : [100,100,100]}
                 'num_points' : [64,64,64]}

        xmin, ymin, zmin = data['mins']
        nx, ny, nz = data['num_points']
        x = np.linspace(xmin,-xmin,nx)
        y = np.linspace(ymin,-ymin,ny)
        z = np.linspace(zmin,-zmin,nz)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]

        nucl_coord = np.asarray(h5.File(molcas_h5file_path,'r')['CENTER_COORDINATES'])
        print(nucl_coord)
        for orbital in list_of_orbitals:
            final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,True)
            target_file = os.path.splitext(molcas_h5file_path)[0] + '.orbital.{}.cube'.format(orbital)
            cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)


    if args.e != None:
        # this part is shitty. Sorry future Alessio.
        print('literally everything in option e is bound to Norbornadiene')
        yml_filename = os.path.abspath(args.e)
        data = yaml.load(open(yml_filename,'r'), Loader=yaml.FullLoader)

        num_cores = data['cores']
        h5file_folder = data['folder']

        # grid parameters 
        xmin, ymin, zmin = data['mins']
        nx, ny, nz = data['num_points']
        x = np.linspace(xmin,-xmin,nx)
        y = np.linspace(ymin,-ymin,ny)
        z = np.linspace(zmin,-zmin,nz)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]


        labels = ['S0','S1','S2','S3','S4','S5','S6','S7']

        for label_index, label in enumerate(labels):
            # this wavefunction
            s0 = np.zeros((25,26,100,8))
            s0[:,:,:,label_index] = 1

            # this here is the indexes of the cartesian grid on the indexes of the nucleus
            file_pickle = data['first_second']
            file_list_index = pickle.load(open(file_pickle,'rb'))
            file_list = create_full_list_of_labels_numpy(data)

            # bear with me
            file_to_be_processed = file_list.flatten()
            wf_to_be_processed = s0.reshape(65000,8)
            file_list_abs = [ os.path.join(h5file_folder, single + '.rasscf.h5') for single in file_to_be_processed ]
            for lab in file_list_index:
                diction = {}
                output = 'between-map_{}_{}.h5'.format(label,lab)
                file_list_index_sub = file_list_index[lab]
                list_indexes = file_list_index_sub
                print(len(wf_to_be_processed), len(file_list_abs), len(list_indexes))

                #cut = 64
                #inputs = tqdm(zip(wf_to_be_processed[:cut], file_list_abs[:cut], list_indexes[:cut]), total=len(wf_to_be_processed))

                inputs = tqdm(zip(wf_to_be_processed, file_list_abs, list_indexes), total=len(wf_to_be_processed))
                a_data = Parallel(n_jobs=num_cores)(delayed(calculate_between_carbons)(single_wf, single_file, single_indexes, data) for single_wf, single_file, single_indexes in inputs)

                reshaped_bonds = np.array(a_data).reshape(25,26,100)
                #reshaped_bonds = np.array(a_data).reshape(4,4,4)

                final = np.zeros((55,56,160))# + 999
                final[15:-15,15:-15,30:-30] = reshaped_bonds
                #final[15:19,15:19,30:34] = reshaped_bonds
                diction['values'] = final

                writeH5fileDict(output,diction)


    if args.f != None:
        yml_filename = os.path.abspath(args.f)
        data = yaml.load(open(yml_filename,'r'), Loader=yaml.FullLoader)
        num_cores = data['cores']
        wf_folder = data['wf_folder']
        files_wf = sorted(glob.glob(wf_folder + '/Gauss*.h5'))
        if 'one_every' in data:
            one_every = data['one_every']
        else:
            one_every = 1

        file_list = create_full_list_of_labels_numpy(data)

        center = data['center']
        pe,ge,te = center
        p,g,t = pe-15,ge-15,te-30
        hms = data['how_many_step']
        h5file_folder = data['folder']
        center_label = file_list[p,g,t]
        molcas_h5file_path = os.path.join(h5file_folder, center_label + '.rasscf.h5')
        nucl_coord = np.asarray(h5.File(molcas_h5file_path,'r')['CENTER_COORDINATES'])
        inputs = files_wf[::one_every]
        Parallel(n_jobs=num_cores)(delayed(parallel_wf)(single_file_wf,one_every,p,g,t,hms,file_list,h5file_folder,args,molcas_h5file_path,nucl_coord,data,wf_folder) for single_file_wf in inputs)

    if args.p != None:
        cube_file, r_c = args.p[0], float(args.p[1])
        cube_single_bonds(cube_file, r_c)

    if args.n != None:
        cube_sum_grid_points(args.n)

    if args.w != None:
        print('Waveunction stats mode')
        threshold = 0.000003
        wf_file = h5.File(args.w,'r')
        wf_int = wf_file['WF']
        time = wf_file['Time'][0]
        give_me_stats(time, wf_int, threshold)

    if args.d != None:
        print('difference mode')
        fn1,fn2 = args.d
        cube_difference(fn1,fn2)

    if args.t != None:
        list_of_files = args.t
        if args.c == None:
            #num_cores = multiprocessing.cpu_count()
            num_cores = 4
        else:
            num_cores = args.c
        strintg = ('We are in multiple file TDM creation mode:\n{}\nWith {} cores')
        print(strintg.format(list_of_files,num_cores))

        abs_paths = [os.path.abspath(x) for x in list_of_files]
        inputs = tqdm(abs_paths)
        Parallel(n_jobs=num_cores)(delayed(get_TDM)(i,updown_file,inactive,cut_states) for i in inputs)

    if args.s != None:

        molcas_h5file_path = os.path.abspath(args.s)
        single_file_data = get_TDM(molcas_h5file_path,updown_file,inactive,cut_states)

        data = { 'mins' : [-10.0,-10.0,-10.0],
                 #'num_points' : [100,100,100]}
                 'num_points' : [64,64,64]}

        xmin, ymin, zmin = data['mins']
        nx, ny, nz = data['num_points']
        x = np.linspace(xmin,-xmin,nx)
        y = np.linspace(ymin,-ymin,ny)
        z = np.linspace(zmin,-zmin,nz)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]

        nucl_coord = np.asarray(h5.File(molcas_h5file_path,'r')['CENTER_COORDINATES'])

        '''
        wvpck_data represent the amplitude on the electronic states for the considered time.
        it is an array with size nes is single point, and a matrix with size nes x ngeom if
        geometry dependent
        '''

        # 8 is electronic states
        nstates = 8
        for state in range(nstates):
        #for state in range(0):
            wvpck_data = np.zeros(nstates)
            wvpck_data[state] = 1
            if args.active:
                final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,True)
                target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_ACTIVE_S{}.cube'.format(state)
            else:
                final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data,False)
                target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_S{}.cube'.format(state)

            cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)

        ## 0.7071 = sqrt(2)
        #thing = 0.7071
        #amplit = [(thing,thing),
        #          (thing,thing*1j),
        #          (thing,-thing),
        #          (thing,-thing*1j)]
        #for time in range(4):
        #    target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_TIME_{}.cube'.format(time)
        #    wvpck_data = np.zeros(8, dtype=complex)
        #    wvpck_data[0] = amplit[time][0]
        #    wvpck_data[6] = amplit[time][1]
        #    print('This state at time {} has {}={} and {}={}'.format(time, 4, amplit[time][0], 5, amplit[time][1]))
        #    final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data)
        #    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)

    if args.b != None:
        # we enter in between mode. This mode means that we will count only grid points in cartesian
        # space that belongs to some rules, for which we already precalculated the index
        # This means that we DO NOT use all the grid space.
        yml_filename = os.path.abspath(args.b)
        data = yaml.load(open(yml_filename,'r'), Loader=yaml.FullLoader)
        num_cores = data['cores']
        wf_folder = data['wf_folder']
        files_wf = sorted(glob.glob(wf_folder + '/Gauss*.h5'))

        # this here is the indexes of the cartesian grid on the indexes of the nucleus
        file_pickle = data['first_second']
        file_list_index = pickle.load(open(file_pickle,'rb'))

        if data['take_core_out']:
            print("\nI will take into account only ACTIVE SPACE\n")

        if 'one_every' in data:
            one_every = data['one_every']
        else:
            one_every = 10

        threshold = data['threshold']
        h5file_folder = data['folder']
        file_list = create_full_list_of_labels_numpy(data)

        for single_file_wf in files_wf[::one_every]:
            print('\n\nI am doing now {}'.format(single_file_wf))
            with h5.File(single_file_wf,'r') as wf_file:
                wf_int = wf_file['WF']
                time = wf_file['Time'][0]
                give_me_stats(time,wf_int,threshold)
                #wf = wf_int[13:16, 14:17, 20:23, :]
                wf = wf_int[15:-15, 15:-15, 30:-30, :]
                #print(wf.shape)

                ## new file list thing
                sums = np.sum(abs2(wf),axis=3)
                trues_indexes = np.where(sums>threshold)
                # the following operation flatten the lists, because np.where returns a flat list
                wf_to_be_processed = wf[trues_indexes]
                file_to_be_processed = file_list[trues_indexes]
                file_list_abs = [ os.path.join(h5file_folder, single + '.rasscf.h5') for single in file_to_be_processed ]


                for lab in file_list_index:
                    fn = os.path.join(wf_folder,lab + '.dat')
                    time_string = '{:6.3f}'.format(time)
                    is_there = False
                    if os.path.exists(fn):
                        print('{} exists...'.format(fn))
                        with open(fn,'r') as f:
                            for line in f.readlines():
                                # this shit below does not work
                                if abs(time - float(line.split()[0])) < 0.00001:
                                    is_there = True
                    if is_there:
                        print('It seems like {} is already calculated'.format(time_string))
                    else:
                        print('{} at time {} does not exist...'.format(fn,time_string))
                        file_list_index_sub = file_list_index[lab]
                        reshaped_file_list_index = file_list_index_sub.reshape(25,26,100)
                        list_indexes = reshaped_file_list_index[trues_indexes]

                        print('Using {} I will process {} files with {} cores'.format(threshold, len(file_list_abs), num_cores))

                        ## HERE HERE you can put option to take out S0
                        ##             parallel version
                        inputs = tqdm(zip(wf_to_be_processed, file_list_abs, list_indexes), total=len(wf_to_be_processed))
                        a_data = Parallel(n_jobs=num_cores)(delayed(calculate_between_carbons)(single_wf, single_file, single_indexes, data) for single_wf, single_file, single_indexes in inputs)

                        ###            serial version (debug)
                        #inputs = zip(wf_to_be_processed, file_list_abs, list_indexes)
                        #for single_wf, single_file, single_indexes in inputs:
                        #      print(single_wf, single_file, single_indexes)
                        #      calculate_between_carbons(single_wf, single_file, single_indexes, data)

                        final_sum = sum(a_data)
                        with open(fn,'a') as filZ:
                            filZ.write('{} {:7.4f}\n'.format(time_string,final_sum))

    if args.i != None:
        # activate folder mode
        yml_filename = os.path.abspath(args.i)
        data = yaml.load(open(yml_filename,'r'), Loader=yaml.FullLoader)
        num_cores = data['cores']
        wf_folder = data['wf_folder']
        files_wf = sorted(glob.glob(wf_folder + '/Gauss*.h5'))

        if 'one_every' in data:
            one_every = data['one_every']
        else:
            one_every = 10

        threshold = data['threshold']

        ## new file list thing
        h5file_folder = data['folder']
        file_list = create_full_list_of_labels_numpy(data)

        xmin, ymin, zmin = data['mins']
        nx, ny, nz = data['num_points']
        x = np.linspace(xmin,-xmin,nx)
        y = np.linspace(ymin,-ymin,ny)
        z = np.linspace(zmin,-zmin,nz)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]

        for single_file_wf in files_wf[::one_every]:
            print('\n\nI am doing now {}'.format(single_file_wf))
            with h5.File(single_file_wf,'r') as wf_file:
                wf_int = wf_file['WF']
                time = wf_file['Time'][0]
                give_me_stats(time,wf_int,threshold)
                #wf = wf_int[13:16, 14:17, 20:23, :]
                wf = wf_int[15:-15, 15:-15, 30:-30, :]

                # Wanna check if output cube file already exists, before doing calculations
                if 'output_file' in data:
                    target_file = data['output_file']
                else:
                    #target_file = os.path.splitext(data['wf'])[1] + '.density.cube'
                    target_file = os.path.basename(wf_folder) + '_time-{:07.3f}.density.cube'.format(time)
                if not os.path.isfile(target_file):
                    # these lines are magic. wf, then absolute value (abs2), then sum along the 
                    # eletronic states, (axis=3)
                    # then get the index of this with np.where
                    # BECAUSE file_list has the same shape than sums, I can use those indexes 
                    # to extract the file name I want in 1D (file_to_be_processed).
                    sums = np.sum(abs2(wf),axis=3)
                    trues_indexes = np.where(sums>threshold)
                    wf_to_be_processed = wf[trues_indexes]
                    sums_to_be_processed = sums[trues_indexes] # I use this to weight geometries
                    file_to_be_processed = file_list[trues_indexes]
                    #print(sums,trues_indexes,file_to_be_processed)

                    file_list_abs = [ os.path.join(h5file_folder, single + '.rasscf.h5') for single in file_to_be_processed]
                    print('Using {} I will process {} files with {} cores'.format(threshold, len(file_list_abs), num_cores))

                    ## not_parallel one
                    #final_cube = np.zeros(nx*ny*nz)
                    #for single_wf, single_file in zip(wf_to_be_processed, file_list_abs):
                    #    final_cube += creating_cube_function_fro_nuclear_list(single_wf, single_file, data)

                    # parallel version
                    inputs = tqdm(zip(wf_to_be_processed, file_list_abs),total=len(wf_to_be_processed))
                    a_data = Parallel(n_jobs=num_cores)(delayed(creating_cube_function_fro_nuclear_list)(single_wf, single_file, data) for single_wf, single_file in inputs)
                    final_cube = sum(a_data)

                    norm_of_abs = sum(sums_to_be_processed)

                    to_be_summed = []
                    for file_geom, single_abs in zip(file_list_abs, sums_to_be_processed):
                        geom = np.asarray(h5.File(file_geom,'r')['CENTER_COORDINATES'])
                        mult = single_abs * geom
                        to_be_summed.append(mult)
                    nucl_coord = sum(to_be_summed)/norm_of_abs

                    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)
                else:
                    print('File {} exists...'.format(target_file))


if __name__ == "__main__" :
    Main()

