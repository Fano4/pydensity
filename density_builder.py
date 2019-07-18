'''
This utility reads an h5 file produced by the PWAPIC code and .wvpck files generated by the Wavepack code and generates a cube file containing the time-dependent electronic density corresponding to the molecule whose dynamics has been computed using Wavepack.
'''

#import math
import os
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


def cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,filename,array_val,nucl_coord):
    file=open(filename,"w")
    file.write("Cube file written using python density utility \n")
    file.write("Norbornadiene electronic density \n")
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nucl_coord.shape[0],xmin,ymin,zmin))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nx,dx,0.000000,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(ny,0.000000,dy,0.000000))
    file.write('{:5} {:11.6f} {:11.6f} {:11.6f} \n'.format(nz,0.000000,0.000000,dz))

    normb_atom_type = [6,6,6,1,1,1,1,6,6,6,6,1,1,1,1]
    print('warning, atomtype for norbornadiene hardcoded')

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



def creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data):
    '''
    wvpck_data :: np.array(Double) <- the 1d singular geom multielectronic state wf.
    return_tuple :: Tuple <- all the rest of the data needed for the cube creation
    target_file :: Filepath <- output cube
    '''
    n_mo, nes = 31, 8
    tdm_file = h5.File(os.path.splitext(molcas_h5file_path)[0] + '.TDM.h5', 'r')
    tran_den_mat = tdm_file['TDM']
    '''
    tdm is the transition density matrix in the basis of mo's, averaged over the populations
    in the excited states.
    '''
    tdm = np.zeros((n_mo,n_mo))

    for ies in np.arange(0,nes):
        tdm = tdm+abs(wvpck_data[ies])**2*tran_den_mat[(ies)*nes+(ies)].reshape((n_mo,n_mo))
        for jes in np.arange(ies+1,nes):
            #print(ies,jes,"are the es")
            tdm = tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*nes+(jes)].reshape((n_mo,n_mo))
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
    for i in range(n_mo):
        # the mo method calculates the MO given space orbitals
        phii[i,:] = orbital_object.mo(i,A.flatten(),B.flatten(),C.flatten())

    cube_array = np.zeros(nx*ny*nz)
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


def give_me_stats(time, wf, threshold):
    pL,gL,tL,_ = wf.shape
    new_one = abs2(wf)
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
    print('{:6.2f} {:5} {:5.2f} {:5.2f} {:5.2f}%'.format(time, calc, mod_sum, norm,mod_sum/norm*100))
    return calculate_this


def command_line_parser():
    '''
    this function deals with command line commands
    '''
    parser = ArgumentParser()
    parser.add_argument("-t", "--tdm",
                    dest="t",
                    nargs='+',
                    help="A list of rasscf.h5 file to process")
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
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def Main():
    # on MAC
    # updown_file = '/Users/stephan/dox/Acu-Stephan/up_down'

    # ON SASHA
    updown_file = '/home/alessio/config/Stephan/up_down'
    inactive = 23
    cut_states = 8

    args = command_line_parser()

    if args.t != None:
        list_of_files = args.t
        if args.c != None:
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
        for state in range(8):
            target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_S{}.cube'.format(state)
            wvpck_data = np.zeros(8)
            wvpck_data[state] = 1
            final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data)
            cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)
        # 0.7071 = sqrt(2)
        thing = 0.7071
        amplit = [(thing,thing),
                  (thing,thing*1j),
                  (thing,-thing),
                  (thing,-thing*1j)]
        for time in range(4):
            target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_TIME_{}.cube'.format(time)
            wvpck_data = np.zeros(8, dtype=complex)
            wvpck_data[0] = amplit[time][0]
            wvpck_data[6] = amplit[time][1]
            print('This state at time {} has {}={} and {}={}'.format(time, 4, amplit[time][0], 5, amplit[time][1]))
            final_cube = creating_cube_function_fro_nuclear_list(wvpck_data,molcas_h5file_path,data)
            cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)

    if args.i != None:
        # activate folder mode

        yml_filename = os.path.abspath(args.i)
        data = yaml.load(open(yml_filename,'r'))
        num_cores = data['cores']
        threshold = data['threshold']
        wf_file = h5.File(data['wf'],'r')
        wf_int = wf_file['WF']
        time = wf_file['Time'][0]
        wf = wf_int[13:16, 14:17, 20:23, :]


        print('\n\n\n!!!! WARNING HARDCODED CUTS !!!!\n\n\n')


        ## new file list thing
        h5file_folder = data['folder']
        file_list = create_full_list_of_labels_numpy(data)

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

        xmin, ymin, zmin = data['mins']
        nx, ny, nz = data['num_points']
        x = np.linspace(xmin,-xmin,nx)
        y = np.linspace(ymin,-ymin,ny)
        z = np.linspace(zmin,-zmin,nz)
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        dz = z[1]-z[0]

        ## not_parallel one
        #final_cube = np.zeros(nx*ny*nz)
        #for single_wf, single_file in zip(wf_to_be_processed, file_list_abs):
        #    final_cube += creating_cube_function_fro_nuclear_list(single_wf, single_file, data)

        # parallel version
        a_data = Parallel(n_jobs=num_cores)(delayed(creating_cube_function_fro_nuclear_list)(single_wf, single_file, data) for single_wf, single_file in zip(wf_to_be_processed, file_list_abs))
        final_cube = sum(a_data)

        target_file = os.path.splitext(data['wf'])[0] + '.density.cube'

        norm_of_abs = sum(sums_to_be_processed)

        to_be_summed = []
        for file_geom, single_abs in zip(file_list_abs, sums_to_be_processed):
            geom = np.asarray(h5.File(file_geom,'r')['CENTER_COORDINATES'])
            mult = single_abs * geom
            to_be_summed.append(mult)
        nucl_coord = sum(to_be_summed)/norm_of_abs

        cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,final_cube.reshape(nx, ny, nz),nucl_coord)


if __name__ == "__main__" :
    Main()

