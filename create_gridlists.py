'''
between carbons
'''
import numpy as np
import os
import quantumpropagator as qp
from tqdm import tqdm
from argparse import ArgumentParser

def points_in_sphere(pt1, r, q):
    '''
    given a 1d vector of points q, returns a boolean telling if the point is into the
    sphere centred at pt1 and radius r.
    '''

    vec = pt1 - q
    norm = np.linalg.norm(vec, axis=1)
    boolean1 = np.abs(norm) < r
    return boolean1

def points_in_cylinder(pt1, pt2, r, q, cyl_shrink):
    '''
    given a 1d vector of points q, returns a boolean telling if the point is into the
    cylinder defined by the two points pt1 and pt2 and radius r.
    cyl_shrink is the distance between the two atoms at which the cylinder starts
    '''

    vec = pt2 - pt1
    vec_norm = np.linalg.norm(vec)
    versor = vec / vec_norm
    pt1_2 = pt1 + (versor * cyl_shrink)
    pt2_2 = pt1 + (versor * (vec_norm-cyl_shrink))

    vec2 = pt2_2 - pt1_2
    vec2_norm = np.linalg.norm(vec2)

    const = r * vec2_norm
    boolean1 = np.dot(q - pt1_2, vec2) >= 0
    boolean2 = np.dot(q - pt2_2, vec2) <= 0
    boolean3 = np.linalg.norm(np.cross(q - pt1_2, vec2), axis=1) <= const

    return boolean1*boolean2*boolean3

def points_in_nonoverlapping_cylinder(geom, r, q, kind):
    '''
    given a 1d vector of points q, returns a boolean telling if the point is into the
    cylinder defined by the two points pt1 and pt2 and radius r.
    cyl_shrink is the distance between the two atoms at which the cylinder starts
    '''

    pt0 = geom[0]
    pt2 = geom[7]
    pt3 = geom[9]
    pt4 = geom[8]
    pt5 = geom[10]
    vec1 = pt0 - pt2
    vec2 = pt0 - pt3
    vec3 = pt0 - pt4
    vec4 = pt0 - pt5
    direction1 = np.cross(vec1,vec2)
    direction2 = np.cross(vec3,vec4)
    if kind == 'blue':
        boolean4 = np.dot(q,direction1) > 0
        boolean5 = np.dot(q,direction2) < 0
        pt1 = pt5
        pt2 = pt3
    elif kind == 'red':
        boolean4 = np.dot(q,direction1) > 0
        boolean5 = np.dot(q,direction2) > 0
        pt1 = pt5
        pt2 = pt2
    else:
        sys.err('what?')

    cyl_shrink = 0.0
    vec = pt2 - pt1
    vec_norm = np.linalg.norm(vec)
    versor = vec / vec_norm
    pt1_2 = pt1 + (versor * cyl_shrink)
    pt2_2 = pt1 + (versor * (vec_norm-cyl_shrink))

    vec2 = pt2_2 - pt1_2
    vec2_norm = np.linalg.norm(vec2)

    const = r * vec2_norm
    boolean1 = np.dot(q - pt1_2, vec2) >= 0
    boolean2 = np.dot(q - pt2_2, vec2) <= 0
    boolean3 = np.linalg.norm(np.cross(q - pt1_2, vec2), axis=1) <= const

    return boolean1*boolean2*boolean3*boolean4*boolean5

def points_in_blue_red_regions(geom,q):
    '''
    This is done to create two non overlapping red and blue regions.
    '''
    pt1 = geom[0]
    pt2 = geom[7]
    pt3 = geom[9]
    pt4 = geom[8]
    pt5 = geom[10]
    vec1 = pt1 - pt2
    vec2 = pt1 - pt3
    vec3 = pt1 - pt4
    vec4 = pt1 - pt5
    direction1 = np.cross(vec1,vec2)
    direction2 = np.cross(vec3,vec4)
    boolean1 = np.dot(q,direction1) < 0
    boolean2 = np.dot(q,direction2) < 0
    boolean3 = np.dot(q,np.array([0.0,0.0,1.0])) < 4
    boolean4 = np.dot(q,np.array([0.0,0.0,1.0])) > -4
    boolean5 = np.dot(q,np.array([1.0,0.0,0.0])) < 4
    return boolean1 * boolean2 * boolean3 * boolean4 * boolean5


def read_single_arguments():
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple for the 3D grid propagator.
    '''
    d = 'This script will launch a Grid quantum propagation'
    parser = ArgumentParser(description=d)

    parser.add_argument('-g','--geometrypickle',
                        type=str,
                        required=True,
                        help='the path of geom.p pickle file',
                        )
    parser.add_argument('-p','--parameters',
                        nargs='+',
                        help='3 numbers: r_c, r_s, shrink. they have to be in BOHR',
                        )
    args = parser.parse_args()

    return args


def main():
    args = read_single_arguments()

    geom_file = args.geometrypickle
    geoms = qp.pickleLoad(geom_file)
    pL, gL, tL, atomN, _ = geoms.shape
    print('File {} loaded...'.format(geom_file))
    print('\nPhi: {}\nGamma: {}\nTheta: {}\nTotal points: {}\n'.format(pL, gL, tL, pL*gL*tL))

    xmin, ymin, zmin = -10,-10,-10
    nx, ny, nz = 64,64,64
    x = np.linspace(xmin,-xmin,nx)
    y = np.linspace(ymin,-ymin,ny)
    z = np.linspace(zmin,-zmin,nz)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    dz = z[1]-z[0]

    B,A,C = np.meshgrid(x,y,z)
    list_of_points_in_3d = np.stack([A.flatten(),B.flatten(),C.flatten()]).T
    #print(list_of_points_in_3d.shape)

    if not args.parameters:
        r_c = 1.6
        list_1 = np.empty(pL*gL*tL,dtype=object)
        list_2 = np.empty(pL*gL*tL,dtype=object)
        counter = 0
        for p in tqdm(range(pL)[:]):
            for g in tqdm(range(gL)[:], leave=False):
                for t in tqdm(range(tL)[:], leave=False):
                    single = geoms[p,g,t]
                    #a = np.where(points_in_blue_red_regions(single,list_of_points_in_3d))
                    a = np.where(points_in_nonoverlapping_cylinder(single, r_c, list_of_points_in_3d, 'red'))
                    b = np.where(points_in_nonoverlapping_cylinder(single, r_c, list_of_points_in_3d, 'blue'))
                    list_1[counter] = a[0]
                    list_2[counter] = b[0]

                    if counter % 1000000 == 6:
                        first_thing = qp.fromBohToAng(list_of_points_in_3d[a])
                        second_thing = qp.fromBohToAng(list_of_points_in_3d[b])
                        qp.saveTraj(np.array([single]),['C','C','C','H','H','H','H','C','C','C','C','H','H','H','H'], '{}'.format(counter),convert=True)
                        qp.saveTraj(np.array([first_thing]),['H']*len(first_thing),'selected{}'.format(counter))
                        qp.saveTraj(np.array([second_thing]),['H']*len(second_thing),'selected2{}'.format(counter))
                    counter+=1
        dic2 = {}
        dic2['list_1'] = list_1
        dic2['list_2'] = list_2
        name_final_pickle = 'r_c-{}-non-overlapping.p'.format(r_c)
        qp.pickleSave(name_final_pickle, dic2)

    else:
        # this should raise up error when there are not 3 values.
        r_c, r_s, cyl_shrink = [ float(x) for x in args.parameters ]

        # remember that atom numeration starts from 0 here
        uno = 10
        due = 9
        tre = 8
        qua = 7

        list_1 = np.empty(pL*gL*tL,dtype=object)
        list_2 = np.empty(pL*gL*tL,dtype=object)
        list_3 = np.empty(pL*gL*tL,dtype=object)
        list_4 = np.empty(pL*gL*tL,dtype=object)
        counter = 0
        for p in tqdm(range(pL)[:]):
            for g in tqdm(range(gL)[:], leave=False):
                for t in tqdm(range(tL)[:], leave=False):
                    single = geoms[p,g,t]

                    pt1 = single[uno]
                    pt2 = single[due]
                    pt3 = single[tre]
                    pt4 = single[qua]
                    a = np.where(points_in_cylinder(pt1, pt2, r_c, list_of_points_in_3d, cyl_shrink))
                    b = np.where(points_in_cylinder(pt3, pt4, r_c, list_of_points_in_3d, cyl_shrink))
                    c = np.where(points_in_cylinder(pt1, pt4, r_c, list_of_points_in_3d, cyl_shrink))
                    d = np.where(points_in_cylinder(pt2, pt3, r_c, list_of_points_in_3d, cyl_shrink))
                    e = np.where(points_in_sphere(pt1, r_s, list_of_points_in_3d))
                    f = np.where(points_in_sphere(pt2, r_s, list_of_points_in_3d))

                    list_1[counter] = np.concatenate((a[0],b[0])) # single
                    list_2[counter] = np.concatenate((c[0],d[0])) # double
                    list_3[counter] = e[0]
                    list_4[counter] = f[0]

                    if counter % 1000000 == 30000: # one each 1000 (I do not want this to trigger)
                        first_thing = qp.fromBohToAng(np.concatenate((list_of_points_in_3d[a],list_of_points_in_3d[b])))
                        second_thing = qp.fromBohToAng(np.concatenate((list_of_points_in_3d[c],list_of_points_in_3d[d])))
                        fourth_thing = qp.fromBohToAng(list_of_points_in_3d[e])
                        fifth_thing = qp.fromBohToAng(list_of_points_in_3d[f])

                        #DEBUG
                        vec = pt2 - pt1
                        vec_norm = np.linalg.norm(vec)
                        versor = vec / vec_norm
                        pt1_2 = pt1 + (versor * cyl_shrink)
                        pt2_2 = pt1 + (versor * (vec_norm-cyl_shrink))
                        third_thing = qp.fromBohToAng(np.stack([pt1,pt2,pt1_2,pt2_2]))
                        qp.saveTraj(np.array([third_thing]),['H']*len(third_thing),'cayo{}'.format(counter))
                        ######

                        qp.saveTraj(np.array([single]),['C','C','C','H','H','H','H','C','C','C','C','H','H','H','H'], '{}'.format(counter),convert=True)
                        qp.saveTraj(np.array([first_thing]),['H']*len(first_thing),'poi{}'.format(counter))
                        qp.saveTraj(np.array([second_thing]),['H']*len(second_thing),'poiT{}'.format(counter))
                        qp.saveTraj(np.array([fourth_thing]),['H']*len(fourth_thing),'carbon{}'.format(counter))
                        qp.saveTraj(np.array([fifth_thing]),['H']*len(fifth_thing),'carbonT{}'.format(counter))
                    counter+=1
        dic2 = {}
        dic2['list_1'] = list_1
        dic2['list_2'] = list_2
        dic2['list_3'] = list_3
        dic2['list_4'] = list_4
        name_final_pickle = 'r_c-{}-r_s-{}-cs-{}-list.p'.format(r_c,r_s,cyl_shrink)
        qp.pickleSave(name_final_pickle, dic2)

        # this file now contains two lists. Each one of those is 65000 long. In each of those 65000 there is a list of grid points in the flattened list. 

        # We need to pass those grid points to the routine of density, so we can sum up...


if __name__ == "__main__":
    #geom_file = '/home/users/alessioval/densities/between_carbon/geoms.p'
    #if os.path.isfile(geom_file):
    #    geoms = qp.pickleLoad(geom_file)
    #    print('File {} loaded...'.format(geom_file))
    #else:
    #    fold = '/home/users/alessioval/x-May-2019/initialData'
    #    fn1 = 'datanewoneWithNACnow.npy'
    #    file_path = os.path.join(fold,fn1)
    #    data = np.load(file_path)
    #    dictionary = data[()]
    #    geoms = dictionary['geoCUBE']
    #    qp.pickleSave(geom_file,geoms)
    #    print('File {} written...'.format(geom_file))
    #main(geoms)
    main()
