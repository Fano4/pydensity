'''
between carbons
'''
import numpy as np
import os
import quantumpropagator as qp
from tqdm import tqdm


def points_in_cylinder(pt1, pt2, r, q):
    '''
    given a 1d vector of points q, returns a boolean telling if the point is into the
    cylinder defined by the two points pt1 and pt2 and ray r.
    '''
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    boolean1 = np.dot(q - pt1, vec) >= 0
    boolean2 = np.dot(q - pt2, vec) <= 0
    boolean3 = np.linalg.norm(np.cross(q - pt1, vec), axis=1) <= const
    return boolean1*boolean2*boolean3

def main(geoms):
    pL, gL, tL, atomN, _ = geoms.shape
    print('\nPhi: {}\nGamma: {}\nTheta: {}\nTotal points: {}\n'.format(pL, gL, tL, pL*gL*tL))
    uno = 10
    due = 9
    tre = 8
    qua = 7
    r = 0.6
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
    print(list_of_points_in_3d.shape)

    list_first = np.empty(pL*gL*tL,dtype=object)
    list_second = np.empty(pL*gL*tL,dtype=object)
    counter = 0
    for p in tqdm(range(pL)[:]):
        for g in tqdm(range(gL)[:], leave=False):
            for t in tqdm(range(tL)[:], leave=False):
                single = geoms[p,g,t]

                pt1 = single[uno]
                pt2 = single[due]
                pt3 = single[tre]
                pt4 = single[qua]
                a = np.where(points_in_cylinder(pt1, pt2, r, list_of_points_in_3d))
                b = np.where(points_in_cylinder(pt3, pt4, r, list_of_points_in_3d))
                c = np.where(points_in_cylinder(pt1, pt4, r, list_of_points_in_3d))
                d = np.where(points_in_cylinder(pt2, pt3, r, list_of_points_in_3d))

                list_first[counter] = np.concatenate((a[0],b[0]))
                list_second[counter] = np.concatenate((c[0],d[0]))
                if counter % 1000000 == 'alle': # one each 1000
                    first_thing = qp.fromBohToAng(np.concatenate((list_of_points_in_3d[a],list_of_points_in_3d[b])))
                    second_thing = qp.fromBohToAng(np.concatenate((list_of_points_in_3d[c],list_of_points_in_3d[d])))
                    qp.saveTraj(np.array([single]),['C','C','C','H','H','H','H','C','C','C','C','H','H','H','H'], '{}'.format(counter),convert=True)
                    qp.saveTraj(np.array([first_thing]),['H']*len(first_thing),'poi{}'.format(counter))
                counter+=1
    dic2 = {}
    dic2['list_first'] = list_first
    dic2['list_second'] = list_second
    qp.pickleSave('first_second.p',dic2)
    # this file now contains two lists. Each one of those is 65000 long. In each of those 65000 there is a list of grid points in the flattened list. 
    # We need to pass those grid points to the routine of density, so we can sum up...


if __name__ == "__main__":
    geom_file = '/home/users/alessioval/densities/between_carbon/geoms.p'
    if os.path.isfile(geom_file):
        geoms = qp.pickleLoad(geom_file)
        print('File {} loaded...'.format(geom_file))
    else:
        fold = '/home/users/alessioval/x-May-2019/initialData'
        fn1 = 'datanewoneWithNACnow.npy'
        file_path = os.path.join(fold,fn1)
        data = np.load(file_path)
        dictionary = data[()]
        geoms = dictionary['geoCUBE']
        qp.pickleSave(geom_file,geoms)
        print('File {} written...'.format(geom_file))
    main(geoms)