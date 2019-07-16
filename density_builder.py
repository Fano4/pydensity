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


def creating_cube_function_fro_nuc(wvpck_data,return_tuple,molcas_h5file_path,target_file):
    '''
    return_tuple :: Tuple <- all the data needed for the cube creation
    target_file :: Filepath <- output cube
    '''
    n_mo,nucl_index,nucl_coord,bas_fun_type,n_states_neut,tran_den_mat,cont_num,cont_zeta,cont_coeff,lcao_num_array,lcao_coeff_array = return_tuple
    print('Writing cube {}'.format(target_file))
    nes = n_states_neut

    '''
    tdm is the transition density matrix in the basis of mo's, averaged over the populations
    in the excited states.
    '''

    tdm = np.zeros((n_mo,n_mo))

    for ies in np.arange(0,nes):
        tdm = tdm+abs(wvpck_data[ies])**2*tran_den_mat[(ies)*n_states_neut+(ies)].reshape((n_mo,n_mo))
        for jes in np.arange(ies+1,nes):
            #print(ies,jes,"are the es")
            tdm = tdm+2*((wvpck_data[ies]*wvpck_data[jes].conjugate()).real)*tran_den_mat[(ies)*n_states_neut+(jes)].reshape((n_mo,n_mo))

    '''
    once you computed the averaged tdm, you just need to evaluate the density
    this is a box centered in the origin 0,0,0
    '''

    xmin = -10.0
    ymin = -10.0
    zmin = -10.0
    dx = 0.31746032
    dy = 0.31746032
    dz = 0.31746032
    nx = 64
    ny = 64
    nz = 64
    cube_array = np.zeros((nx,ny,nz))
    x = np.linspace(-10.0,10.0,64)
    y = np.linspace(-10.0,10.0,64)
    z = np.linspace(-10.0,10.0,64)
    B,A,C = np.meshgrid(x,y,z)
    phii = np.empty((n_mo,64*64*64))
    orbital_object = Orbitals(molcas_h5file_path,'hdf5')
    for i in range(n_mo):
        # the mo method calculates the MO given space orbitals
        phii[i,:] = orbital_object.mo(i,A.flatten(),B.flatten(),C.flatten())

    cube_array = np.zeros(64*64*64)
    for i in range(n_mo):
        for j in range(n_mo):
            # np.tensordot(phii,np.tensordot(tdm,phii,axes=0),axes=0)
            cube_array += phii[i] * phii[j] * tdm[i,j]

    cubegen(xmin,ymin,zmin,dx,dy,dz,nx,ny,nz,target_file,cube_array.reshape(64,64,64),nucl_coord * 0.529)


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


def command_line_parser():
    '''
    this function deals with command line commands
    '''
    parser = ArgumentParser()
    parser.add_argument("-l", "--list-tdm",
                    dest="l",
                    nargs='+',
                    help="A list of rasscf.h5 file to process")
    parser.add_argument("-t", "--tdm",
                    dest="t",
                    type=str,
                    help="The rasscf h5 file without TDM")
    parser.add_argument("-c", "--core",
                    dest="c",
                    type=int,
                    help="number of cores for the calculation")
    parser.add_argument("-f", "--folder_tdm",
                    dest="f",
                    type=str,
                    help="A folder of rasscf h5 files without TDM")
    parser.add_argument("-w", "--wavefunction",
                    dest="w",
                    type=str,
                    help="The WF h5 file")
    parser.add_argument("-i", "--input_multigeom_mode",
                    dest="i",
                    type=str,
                    help="The yml file to set up the geometries")
    parser.add_argument("-s", "--single_file_mode",
                    dest="s",
                    type=str,
                    help="The single file path")
    parser.add_argument("-g", "--global_file_mode",
                    dest="g",
                    type=str,
                    help="The global pickle file path")
    if len(sys.argv)==1:
        parser.print_help()
    return parser.parse_args()


def Main():
    # molcas_h5file_path <- THIS VARIABLE IS NOW IS GIVEN BY THE -s option
    # $ python density_builder.py -s 'molcas_h5file_path'
    # molcas_h5file_path = '/Users/stephan/dox/Acu-Stephan/zNorbornadiene_P005-000_P020-000_P124-190.rasscf.h5'
    # target_file <- THIS VARIABLE IS NOW GENERATED AUTOMATICALLY DEPENDING ON molcas_h5file_path name
    # target_file = "/Users/stephan/Desktop/density_test_alessio_.cub"



    # on MAC
    # updown_file = '/Users/stephan/dox/Acu-Stephan/up_down'

    # ON SASHA
    updown_file = '/home/alessio/config/Stephan/up_down'

    inactive = 23
    cut_states = 8

    args = command_line_parser()

    if args.l != None:
        list_of_files = args.l
        if args.c != None:
            #num_cores = multiprocessing.cpu_count()
            num_cores = 4
        else:
            num_cores = len(list_of_files)
        strintg = ('We are in multiple file TDM creation mode:\n{}\nWith {} cores')
        print(strintg.format(list_of_files,num_cores))

        abs_paths = [os.path.abspath(x) for x in list_of_files]
        inputs = tqdm(abs_paths)
        Parallel(n_jobs=num_cores)(delayed(get_TDM)(i,updown_file,inactive,cut_states) for i in inputs)

    if args.t != None:
        print('we are in TDM creation mode')
        molcas_h5file_path = os.path.abspath(args.t)
        get_TDM(molcas_h5file_path, updown_file, inactive, cut_states)

    if args.f != None:
        if args.c != None:
            #num_cores = multiprocessing.cpu_count()
            num_cores = 16
        else:
            num_cores = args.c

        # new file list thing
        h5file_folder = args.f

        strintg = ('We are in folder TDM creation mode\n\n{}\nWith {} cores')
        print(strintg.format(h5file_folder,num_cores))

        abs_path = os.path.abspath(h5file_folder)
        file_list_abs = [os.path.join(abs_path, f) for f in os.listdir(abs_path)]
        inputs = tqdm(file_list_abs)

        Parallel(n_jobs=num_cores)(delayed(get_TDM)(i,updown_file,inactive,cut_states) for i in inputs)


    if args.g != None:
        # activate Global mode.
        pickle_file_name = os.path.abspath(args.g)
        return_tuple = pickleLoad(pickle_file_name)
        if args.w == None:
            print('\nyou have to provide Wavefunction file\n')
        else:
            wf_file_name = os.path.abspath(args.w)
            print('reading wf {}'.format(wf_file_name))
            wf_h5_file = h5.File(wf_file_name, 'r')
            wf_ext = np.asarray(wf_h5_file['WF'])

            # wf = wf_ext[15:-15, 15:-15, 30:-30, :]
            wf_int = wf_ext[15:-15, 15:-15, 30:-30, :]
            wf = wf_int[13:16, 14:17, 20:23, :]
            print('\n\n\n!!!! WARNING HARDCODED CUTS !!!!\n\n\n')
            phiL,gamL,theL,nstates = wf.shape
            reshaped_wf = wf.reshape(phiL*gamL*theL,nstates)
            print('Wavefunction is {}'.format(reshaped_wf.shape))


    if args.s != None:
        # activate single file mode, this is the code as you left it.
        # just that get_all_data and creating_cube_function are now two different phases
        molcas_h5file_path = os.path.abspath(args.s)

        # get_all_data will create the pickle if not present OR use the pickle if present
        single_file_data = get_all_data(molcas_h5file_path,updown_file,inactive,cut_states)

        # 8 is electronic states
        '''
        wvpck_data represent the amplitude on the electronic states for the considered time.
        it is an array with size nes is single point, and a matrix with size nes x ngeom if
        geometry dependent
        '''
        for state in range(8):
            target_file = os.path.splitext(molcas_h5file_path)[0] + '.testsingle_S{}.cube'.format(state)
            wvpck_data = np.zeros(8)
            wvpck_data[state] = 1
            creating_cube_function_fro_nuc(wvpck_data,single_file_data,molcas_h5file_path,target_file)
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
            creating_cube_function_fro_nuc(wvpck_data,single_file_data,molcas_h5file_path,target_file)


    if args.i != None:
        # activate folder mode

        yml_filename = os.path.abspath(args.i)
        data = yaml.load(open(yml_filename,'r'))
        pickle_global_file_name = os.path.splitext(yml_filename)[0] + '.global.pickle'
        num_cores = multiprocessing.cpu_count()

        # new file list thing
        h5file_folder = data['folder']
        file_list = create_full_list_of_labels(data)
        file_list_abs = [ os.path.join(h5file_folder, single + '.rasscf.h5') for single in file_list ]
        inputs = tqdm(file_list_abs)

        # code form here is REALLY SHITTY, sorry my man
        # this seems to create something in the right order
        a_data = Parallel(n_jobs=num_cores)(delayed(process_single_file)(i,updown_file,inactive,cut_states) for i in inputs)

        # This code below just outputs some statistics of what changes between tuples. That is index 2,5,10
        for i in range(len(a_data[0])):
            boole = np.all(a_data[0][i]==a_data[1][i])
            if boole:
                print('{:2} TRUE  {}'.format(i,type(a_data[0][i])))
            else:
                print('{:2} FALSE {} {}'.format(i,type(a_data[0][i]),a_data[0][i].shape))


        # geom is index 2
        filesN = len(file_list)
        natoms, _ = a_data[0][2].shape
        global_geom = np.empty((filesN,natoms,3))

        # something_else is index 5
        something_else_1, something_else_2 = a_data[0][5].shape
        global_something_else = np.empty((filesN, something_else_1, something_else_2))

        # TDM is index 10
        nmo, _ = a_data[0][10].shape
        global_tdm = np.empty((filesN, nmo, nmo))

        for i,single_tuple in enumerate(a_data):
            global_geom[i] = single_tuple[2]
            global_something_else[i] = single_tuple[5]
            global_tdm[i] = single_tuple[10]

        # global data is list now
        global_data = list(a_data[0])
        global_data[2] = global_geom
        global_data[5] = global_something_else
        global_data[10] = global_tdm

        for i in range(len(global_data)):
                print('{:2} {} {}'.format(i,type(global_data[i]),global_data[10].shape))

        print('Creating {}'.format(pickle_global_file_name))
        pickleSave(pickle_global_file_name,global_data)

        print('\nI did this using {} cores'.format(num_cores))

if __name__ == "__main__" :
    Main()

