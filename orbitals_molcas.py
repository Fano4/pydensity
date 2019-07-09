'''
Pegamoyd my love
'''

import numpy as np
import h5py

from fractions import Fraction

import os
import os.path
import codecs
import re
import struct
import traceback
import time
from copy import deepcopy
from socket import gethostname
from datetime import datetime
from tempfile import mkdtemp
from shutil import rmtree
from functools import partial
from collections import OrderedDict
from itertools import zip_longest

class Orbitals(object):

  def __init__(self, orbfile, ftype):
    self.inporb = None
    self.file = orbfile
    self.type = ftype
    self.eps = np.finfo(np.float).eps
    self.wf = 'SCF'
    if (self.type == 'hdf5'):
      self.read_h5_basis()
      self.read_h5_MO()
      self.inporb = 'gen'
      self.h5file = self.file
    elif (self.type == 'molden'):
      self.read_molden_basis()
      self.read_molden_MO()

  # Read basis set from an HDF5 file
  def read_h5_basis(self):
    with h5py.File(self.file, 'r') as f:
      sym = f.attrs['NSYM']
      self.N_bas = f.attrs['NBAS']
      self.irrep = [i.decode('ascii').strip() for i in f.attrs['IRREP_LABELS']]
      # First read the centers and their properties
      if (sym > 1):
        labels = f['DESYM_CENTER_LABELS'][:]
        charges = f['DESYM_CENTER_CHARGES'][:]
        coords = f['DESYM_CENTER_COORDINATES'][:]
        self.mat = np.reshape(f['DESYM_MATRIX'][:], (sum(self.N_bas), sum(self.N_bas))).T
      else:
        labels = f['CENTER_LABELS'][:]
        charges = f['CENTER_CHARGES'][:]
        coords = f['CENTER_COORDINATES'][:]
      self.centers = [{'name':str(l.decode('ascii')).strip(), 'Z':int(q), 'xyz':x} for l,q,x in zip(labels, charges, coords)]
      self.geomcenter = (np.amin(coords, axis=0) + np.amax(coords, axis=0))/2
      # Then read the primitives and assign them to the centers
      prims = f['PRIMITIVES'][:]    # (exponent, coefficient)
      prids = f['PRIMITIVE_IDS'][:] # (center, l, shell)
      # The basis_id contains negative l if the shell is Cartesian
      if (sym > 1):
        basis_function_ids = 'DESYM_BASIS_FUNCTION_IDS'
      else:
        basis_function_ids = 'BASIS_FUNCTION_IDS'
      bf_id = np.rec.fromrecords(np.insert(f[basis_function_ids][:], 4, -1, axis=1), names='c, s, l, m, tl') # (center, shell, l, m, true-l)
      bf_cart = set([(b['c'], b['l'], b['s']) for b in bf_id if (b['l'] < 0)])
      # Add contaminants, which are found as lower l basis functions after higher l ones
      # The "tl" field means the "l" from which exponents and coefficients are to be taken, or "true l"
      ii = [sum(self.N_bas[:i]) for i in range(len(self.N_bas))]
      if (sym > 1):
        sbf_id = np.rec.fromrecords(np.insert(f['BASIS_FUNCTION_IDS'][:], 4, -1, axis=1), names='c, s, l, m, tl')
      else:
        sbf_id = bf_id
      for i,nb in zip(ii, self.N_bas):
        prev = {'c': -1, 'l': -1, 's': -1, 'tl': 0, 'm': None}
        for b in sbf_id[i:i+nb]:
          if (b['c']==prev['c'] and abs(b['l']) < abs(prev['tl'])):
            b['tl'] = prev['tl']
          else:
            b['tl'] = b['l']
          prev = {n: b[n] for n in b.dtype.names}
      if (sym > 1):
        for i,b in enumerate(self.mat.T):
          bb = np.array(sbf_id[i])
          nz = np.nonzero(b)[0]
          assert (np.all(bf_id[nz][['l','s','m']] == bb[['l','s','m']]))
          for j in nz:
            bf_id[j]['tl'] = bb['tl']
      # Workaround for bug in HDF5 files where p-type contaminants did all have m=0
      p_shells, p0_counts = np.unique([np.array(b)[['c','s','tl']] for b in bf_id if ((b['l']==1) and (b['m']==0))], return_counts=True)
      if (np.any(p0_counts > 1)):
        if (sym > 1):
          # can't fix it with symmetry
          error = 'Bad m for p contaminants. The file could have been created by a buggy or unsupported OpenMolcas version'
          raise Exception(error)
        else:
          m = -1
          for i in np.where(bf_id['l']==1)[0]:
            bi = list(p_shells).index(bf_id[i][['c','s','tl']])
            if (p0_counts[bi] > 1):
              bf_id[i]['m'] = m
              m += 1
              if (m > 1):
                m = -1
      # Count the number of m per basis to make sure it matches with the expected type
      counts = {}
      for b in bf_id:
        key = (b['c'], b['l'], b['s'], b['tl'])
        counts[key] = counts.get(key, 0)+1
      for f,n in counts.items():
        l = f[1]
        if (((l >= 0) and (n != 2*l+1)) or ((l < 0) and (n != (-l+1)*(-l+2)/2))):
          error = 'Inconsistent basis function IDs. The file could have been created by a buggy or unsupported OpenMolcas version'
          raise Exception(error)
      # Maximum angular momentum in the whole basis set,
      maxl = max([p[1] for p in prids])
      for i,c in enumerate(self.centers):
        c['basis'] = []
        c['cart'] = {}
        for l in range(maxl+1):
          ll = []
          # number of shells for this l and center
          maxshell = max([0] + [p[2] for p in prids if ((p[0] == i+1) and (p[1] == l))])
          for s in range(maxshell):
            # find out if this is a Cartesian shell (if the l is negative)
            # note that Cartesian shells never have (nor are) contaminants,
            # and since contaminants come after regular shells,
            # it should be safe to save just l and s
            if ((i+1, -l, s+1) in bf_cart):
              c['cart'][(l, s)] = True
            # get exponents and coefficients
            ll.append([0, [pp.tolist() for p,pp in zip(prids, prims) if ((p[0] == i+1) and (p[1] == l) and (p[2] == s+1))]])
          c['basis'].append(ll)
        # Add contaminant shells, that is, additional shells for lower l, with exponents and coefficients
        # from a higher l, and with some power of r**2
        for l in range(maxl-1):
          # find basis functions for this center and l, where l != tl
          cont = [(b['l'],b['tl'],b['s']) for b in bf_id[np.logical_and(bf_id['c']==i+1, bf_id['l']==l)] if (b['l'] != b['tl'])]
          # get a sorted unique set
          cont = sorted(set(cont))
          # copy the exponents and coefficients from the higher l and set the power of r**2
          for j in cont:
            new = deepcopy(c['basis'][j[1]][j[2]-1])
            new[0] = (j[1]-j[0])//2
            c['basis'][l].append(new)
      # At this point each center[i]['basis'] is a list of maxl items, one for each value of l,
      # each item is a list of shells,
      # each item is [power of r**2, primitives],
      # each "primitives" is a list of [exponent, coefficient]
      # Now get the indices for sorting all the basis functions (2l+1 or (l+1)(l+2)/2 for each shell)
      # by center, l, m, "true l", shell
      # To get the correct sorting for Cartesian shells, invert l
      for b in bf_id:
        if (b['l'] < 0):
          b['l'] *= -1
      self.bf_sort = np.argsort(bf_id, order=('c', 'l', 'm', 'tl', 's'))
      # And sph_c can be computed
      self.set_sph_c(maxl)
    # Reading the basis set invalidates the orbitals, if any
    self.MO = None
    self.MO_a = None
    self.MO_b = None

  # Read molecular orbitals from an HDF5 file
  def read_h5_MO(self):
    with h5py.File(self.file, 'r') as f:
      # Read the orbital properties
      if ('MO_ENERGIES' in f):
        mo_en = f['MO_ENERGIES'][:]
        mo_oc = f['MO_OCCUPATIONS'][:]
        mo_cf = f['MO_VECTORS'][:]
        if ('MO_TYPEINDICES' in f):
          mo_ti = f['MO_TYPEINDICES'][:]
        else:
          mo_ti = [b'?' for i in mo_oc]
      else:
        mo_en = []
        mo_oc = []
        mo_cf = []
        mo_ti = []
      if ('MO_ALPHA_ENERGIES' in f):
        mo_en_a = f['MO_ALPHA_ENERGIES'][:]
        mo_oc_a = f['MO_ALPHA_OCCUPATIONS'][:]
        mo_cf_a = f['MO_ALPHA_VECTORS'][:]
        if ('MO_ALPHA_TYPEINDICES' in f):
          mo_ti_a = f['MO_ALPHA_TYPEINDICES'][:]
        else:
          mo_ti_a = [b'?' for i in mo_oc_a]
      else:
        mo_en_a = []
        mo_oc_a = []
        mo_cf_a = []
        mo_ti_a = []
      if ('MO_BETA_ENERGIES' in f):
        mo_en_b = f['MO_BETA_ENERGIES'][:]
        mo_oc_b = f['MO_BETA_OCCUPATIONS'][:]
        mo_cf_b = f['MO_BETA_VECTORS'][:]
        if ('MO_BETA_TYPEINDICES' in f):
          mo_ti_b = f['MO_BETA_TYPEINDICES'][:]
        else:
          mo_ti_b = [b'?' for i in mo_oc_b]
      else:
        mo_en_b = []
        mo_oc_b = []
        mo_cf_b = []
        mo_ti_b = []
      mo_ti = [str(i.decode('ascii')) for i in mo_ti]
      mo_ti_a = [str(i.decode('ascii')) for i in mo_ti_a]
      mo_ti_b = [str(i.decode('ascii')) for i in mo_ti_b]
      self.MO = [{'ene':e, 'occup':o, 'type':t} for e,o,t in zip(mo_en, mo_oc, mo_ti)]
      self.MO_a = [{'ene':e, 'occup':o, 'type':t} for e,o,t in zip(mo_en_a, mo_oc_a, mo_ti_a)]
      self.MO_b = [{'ene':e, 'occup':o, 'type':t} for e,o,t in zip(mo_en_b, mo_oc_b, mo_ti_b)]
      # Read the coefficients
      ii = [sum(self.N_bas[:i]) for i in range(len(self.N_bas))]
      j = 0
      for i,b,s in zip(ii, self.N_bas, self.irrep):
        for orb,orb_a,orb_b in zip_longest(self.MO[i:i+b], self.MO_a[i:i+b], self.MO_b[i:i+b]):
          if (orb):
            orb['sym'] = s
            orb['coeff'] = np.zeros(sum(self.N_bas))
            orb['coeff'][i:i+b] = mo_cf[j:j+b]
          if (orb_a):
            orb_a['sym'] = s
            orb_a['coeff'] = np.zeros(sum(self.N_bas))
            orb_a['coeff'][i:i+b] = mo_cf_a[j:j+b]
          if (orb_b):
            orb_b['sym'] = s
            orb_b['coeff'] = np.zeros(sum(self.N_bas))
            orb_b['coeff'][i:i+b] = mo_cf_b[j:j+b]
          j += b
      # Desymmetrize the MOs
      if (len(self.N_bas) > 1):
        for orb in self.MO + self.MO_a + self.MO_b:
          orb['coeff'] = np.dot(self.mat, orb['coeff'])
      self.roots = ['Average']
      self.dm = [np.diag([o['occup'] for o in self.MO if (o['type'] in ['1', '2', '3'])])]
      self.sdm = None
      self.tdm = None
      self.H_eff = None
      mod = None
      if ('MOLCAS_MODULE' in f.attrs):
        mod = f.attrs['MOLCAS_MODULE'].decode('ascii')
      if (mod == 'CASPT2'):
        self.wf = 'PT2'
        self.roots[0] = 'Reference'
        # For CASPT2 the density matrices are symmetry-blocked, and for all orbitals (not F or D),
        # here we convert them to square, with full size
        self.dm = np.array([np.diag([o['occup'] for o in self.MO if (o['type'] not in ['F', 'D'])])])
        if ('DENSITY_MATRIX' in f):
          rootids = f.attrs['STATE_ROOTID'][:]
          for root in range(len(rootids)):
            self.dm = np.concatenate((self.dm, [np.zeros_like(self.dm[0])]))
            nMO = [(sum(self.N_bas[:i]), sum(self.N_bas[:i+1])) for i in range(len(self.N_bas))]
            j = 0
            k = 0
            for i,nbas in zip(nMO, self.N_bas):
              n = len([o for o in self.MO[i[0]:i[1]] if (o['type'] not in ['F', 'D'])])
              j1 = int(n*(n+1)/2)
              dm = np.zeros((n, n))
              dm[np.tril_indices(n, 0)] = f['DENSITY_MATRIX'][root,j:j+j1]
              dm = dm + np.tril(dm, -1).T
              self.dm[-1][k:k+n,k:k+n] = dm
              j += j1
              k += n
          # For MS-CASPT2, the densities are SS, but the energies are MS,
          # so take the energies from the effective Hamiltonian matrix instead
          if ('H_EFF' in f):
            self.H_eff = f['H_EFF'][:]
            self.roots.extend(['{0}: {1:.6f}'.format(i, e) for i,e in zip(rootids, np.diag(self.H_eff))])
            self.msroots = ['Reference']
            self.msroots.extend(['{0}: {1:.6f}'.format(i+1, e) for i,e in enumerate(f['STATE_PT2_ENERGIES'])])
          else:
            self.roots.extend(['{0}: {1:.6f}'.format(i, e) for i,e in zip(rootids, f['STATE_PT2_ENERGIES'])])
      else:
        if ('DENSITY_MATRIX' in f):
          self.dm = np.concatenate((self.dm, f['DENSITY_MATRIX'][:]))
          rootids = [i+1 for i in range(f.attrs['NROOTS'])]
          self.roots.extend(['{0}: {1:.6f}'.format(i, e) for i,e in zip(rootids, f['ROOT_ENERGIES'])])
        if ('SPINDENSITY_MATRIX' in f):
          sdm = f['SPINDENSITY_MATRIX'][:]
          if (not np.allclose(sdm, np.zeros_like(sdm))):
            sdm = np.insert(sdm, 0, np.mean(sdm, axis=0), axis=0)
            self.sdm = sdm
        if ('TRANSITION_DENSITY_MATRIX' in f):
          tdm = f['TRANSITION_DENSITY_MATRIX'][:]
          if (not np.allclose(tdm, np.zeros_like(tdm))):
            self.tdm = tdm
      # Read the optional notes
      if ('Pegamoid_notes' in f):
        self.notes = f['Pegamoid_notes'][:]

  # Read basis set from a Molden file
  def read_molden_basis(self):
    with open(self.file, 'r') as f:
      # Molden supports up to g functions, and by default all are Cartesian
      ang_labels = 'spdfg'
      cart = [False, True, True, True, True]
      # Specify the order of the Cartesian components according to the convention (see ang)
      order = []
      order.append([0])
      order.append([-1, 0, 1])
      order.append([-2, 1, 3, -1, 0, 2])
      order.append([-3, 3, 6, 0, -2, -1, 2, 5, 4, 1])
      order.append([-4, 6, 10, -3, -2, 2, 7, 5, 9, -1, 1, 8, 0, 3, 4])
      maxl = 0
      done = True
      if (re.search(r'\[MOLDEN FORMAT\]', f.readline(), re.IGNORECASE)):
        done = False
      line = ' '
      while ((not done) and (line != '')):
        line = f.readline()
        # Read the geometry
        if re.search(r'\[N_ATOMS\]', line, re.IGNORECASE):
          num = int(f.readline())
        elif re.search(r'\[ATOMS\]', line, re.IGNORECASE):
          unit = 1
          if (re.search(r'Angs', line, re.IGNORECASE)):
            unit = 1/0.52917721092
          self.centers = []
          for i in range(num):
            l, _, q, x, y, z = f.readline().split()
            self.centers.append({'name':l, 'Z':int(q), 'xyz':np.array([float(x), float(y), float(z)])*unit})
          self.geomcenter = (np.amin([c['xyz'] for c in self.centers], axis=0) + np.amax([c['xyz'] for c in self.centers], axis=0))/2
        # Read tags for spherical shells
        elif re.search(r'\[5D\]', line, re.IGNORECASE):
          cart[2] = False
          cart[3] = False
        elif re.search(r'\[5D7F\]', line, re.IGNORECASE):
          cart[2] = False
          cart[3] = False
        elif re.search(r'\[5D10F\]', line, re.IGNORECASE):
          cart[2] = False
          cart[3] = True
        elif re.search(r'\[7F\]', line, re.IGNORECASE):
          cart[3] = False
        elif re.search(r'\[9G\]', line, re.IGNORECASE):
          cart[4] = False
        # Read basis functions: a series of blank-separated blocks
        # starting with center number and followed by all the shells,
        # each is the angular momentum letter and number of primitives,
        # plus this number of exponents and coefficients.
        elif re.search(r'\[GTO\]', line, re.IGNORECASE):
          bf_id = []
          while (True):
            save = f.tell()
            # First find out if this is another center, or the basis set
            # specification has finished
            try:
              n = int(f.readline())
            except:
              f.seek(save)
              break
            self.centers[n-1]['cart'] = []
            basis = {}
            # Read the shells for this center
            while (True):
              try:
                l, nprim = f.readline().split()
                nprim = int(nprim)
                # The special label "sp" has the same exponent
                # but different coefficients for s and p functions
                if (l.lower() == 'sp'):
                  if (0 not in basis):
                    basis[0] = []
                  basis[0].append([0, []])
                  if (1 not in basis):
                    basis[1] = []
                  basis[1].append([0, []])
                  for i in range(nprim):
                    e, c1, c2 = (float(i) for i in f.readline().split())
                    basis[0][-1][1].append([e, c1])
                    basis[1][-1][1].append([e, c2])
                  bf_id.append([n, len(basis[0]), 0, 0])
                  if (cart[0]):
                    self.centers[n-1]['cart'].append((0, len(basis[0])-1))
                  if (cart[1]):
                    self.centers[n-1]['cart'].append((1, len(basis[1])-1))
                  for i in order[1]:
                    bf_id.append([n, len(basis[1]), -1, i])
                else:
                  l = ang_labels.index(l.lower())
                  if (l not in basis):
                    basis[l] = []
                  basis[l].append([0, []])
                  # Read exponents and coefficients
                  for i in range(nprim):
                    e, c = (float(i) for i in f.readline().split())
                    basis[l][-1][1].append([e, c])
                  # Set up the basis_id
                  if (cart[l]):
                    self.centers[n-1]['cart'].append((l, len(basis[l])-1))
                    for i in order[l]:
                      bf_id.append([n, len(basis[l]), -l, i])
                  else:
                    for i in range(l+1):
                      bf_id.append([n, len(basis[l]), l, i])
                      if (i > 0):
                        bf_id.append([n, len(basis[l]), l, -i])
              except:
                break
            nl = max(basis.keys())
            maxl = max(maxl, nl)
            self.centers[n-1]['basis'] = [[] for i in range(nl+1)]
            for i in basis.keys():
              self.centers[n-1]['basis'][i] = basis[i][:]
          # At this point each center[i]['basis'] is a list of maxl items, one for each value of l,
          # each item is a list of shells,
          # each item is [power of r**2, primitives],
          # each "primitives" is a list of [exponent, coefficient]
        elif re.search(r'\[MO\]', line, re.IGNORECASE):
          done = True
      # Now get the normalization factors and invert l for Cartesian shells
      # The factor is 1/sqrt(N(lx)*N(ly)*N(lz)), where N(x) is the double factorial of 2*x-1
      # N(0)=1, N(1)=1, N(2)=1*3, N(3)=1*3*5, ...
      self.fact = np.full(len(bf_id), 1.0)
      bf_id = np.rec.fromrecords(bf_id, names='c, s, l, m')
      for i,b in enumerate(bf_id):
        if (b['l'] < 0):
          b['l'] *= -1
          ly = int(np.floor((np.sqrt(8*(b['m']+b['l'])+1)-1)/2))
          lz = b['m']+b['l']-ly*(ly+1)//2
          lx = b['l']-ly
          ly -= lz
          lx = self._binom(2*lx, lx)*np.math.factorial(lx)//2**lx
          ly = self._binom(2*ly, ly)*np.math.factorial(ly)//2**ly
          lz = self._binom(2*lz, lz)*np.math.factorial(lz)//2**lz
          self.fact[i] = 1.0/np.sqrt(float(lx*ly*lz))
      # And get the indices for sorting the basis functions by center, l, m, shell
      self.bf_sort = np.argsort(bf_id, order=('c', 'l', 'm', 's'))
      self.head = f.tell()
      self.N_bas = [len(bf_id)]
      self.set_sph_c(maxl)
    # Reading the basis set invalidates the orbitals, if any
    self.MO = None
    self.MO_a = None
    self.MO_b = None

  # Read molecular orbitals from a Molden file
  def read_molden_MO(self):
    self.MO = []
    self.MO_a = []
    self.MO_b = []
    # Each orbital is a header with properties and a list of coefficients
    with open(self.file, 'r') as f:
      f.seek(self.head)
      while (True):
        try:
          sym = re.sub(r'^\d*', '', f.readline().split()[1])
          ene = float(f.readline().split()[1])
          spn = 'b' if (f.readline().split()[1] == 'Beta') else 'a'
          occ = float(f.readline().split()[1])
          cff = np.zeros(sum(self.N_bas))
          for i in range(sum(self.N_bas)):
            n, c = f.readline().split()
            cff[int(n)-1] = float(c)
          # Save the orbital as alpha or beta
          if (spn == 'b'):
            self.MO_b.append({'ene':ene, 'occup':occ, 'sym':sym, 'type':'?', 'coeff':self.fact*cff})
          else:
            self.MO_a.append({'ene':ene, 'occup':occ, 'sym':sym, 'type':'?', 'coeff':self.fact*cff})
        except:
          break
    # Build the list of irreps from the orbitals
    self.irrep = []
    for o in self.MO_a + self.MO_b:
      if (o['sym'] not in self.irrep):
        self.irrep.append(o['sym'])
    if (not self.MO_b):
      self.MO = deepcopy(self.MO_a)
      self.MO_a = []

  # Read molecular orbitals from an InpOrb file
  def read_inporb_MO(self, infile):
    if (self.type != 'hdf5'):
      return 'Current file is not HDF5'
    self.file = infile
    self.inporb = 0
    fortrannums = re.compile(r'-?\d*\.\d*[EeDd][+-]\d*(?!\.)')
    sections = {}
    with open(infile, 'r') as f:
      line = f.readline()
      # First read the header section (which must be the first) and
      # make sure the number of basis functions matches the current values
      while ((not line.startswith('#INFO')) and (line != '')):
        line = f.readline()
      sections['INFO'] = True
      line = f.readline()
      uhf, nsym, _ = (int(i) for i in f.readline().split())
      N_bas = np.array([int(i) for i in f.readline().split()])
      nMO = np.array([int(i) for i in f.readline().split()])
      if (not np.array_equal(N_bas, self.N_bas)):
        return 'Incompatible InpOrb data'
      # Decide whether or not beta orbitals will be read
      if (uhf):
        self.MO_b = deepcopy(self.MO)
      else:
        self.MO_a = []
        self.MO_b = []
      ii = [sum(self.N_bas[:i]) for i in range(len(self.N_bas))]
      # Read until EOF
      while (line != ''):
        # Find next section
        while ((not line.startswith('#')) and (line != '')):
          line = f.readline()
        # Read orbital coefficients, only the non-zero (by symmetry)
        # coefficients are written in the file
        if (line.startswith('#ORB')):
          sections['ORB'] = True
          line = '\n'
          j = 0
          for i,b,s in zip(ii, self.N_bas, self.irrep):
            for orb in self.MO[i:i+b]:
              orb['sym'] = s
              orb['coeff'] = np.zeros(sum(self.N_bas))
              cff = []
              f.readline()
              while (len(cff) < b):
                line = f.readline()
                if (re.search(r'\.[^ ]*\.', line)):
                  cff.extend(fortrannums.findall(line))
                else:
                  cff.extend(line.split())
              orb['coeff'][i:i+b] = [float(c) for c in cff]
              j += b
        elif (line.startswith('#UORB')):
          sections['UORB'] = True
          line = '\n'
          if (uhf):
            j = 0
            for i,b,s in zip(ii, self.N_bas, self.irrep):
              for orb in self.MO_b[i:i+b]:
                orb['sym'] = s
                orb['coeff'] = np.zeros(sum(self.N_bas))
                cff = []
                f.readline()
                while (len(cff) < b):
                  line = f.readline()
                  if (re.search(r'\.[^ ]*\.', line)):
                    cff.extend(fortrannums.findall(line))
                  else:
                    cff.extend(line.split())
                orb['coeff'][i:i+b] = [float(c) for c in cff]
                j += b
        # Read the occupations
        elif (line.startswith('#OCC')):
          sections['OCC'] = True
          line = '\n'
          f.readline()
          occ = []
          for i,b in zip(ii, self.N_bas):
            while (len(occ) < i+b):
              line = f.readline()
              if (re.search(r'\.[^ ]*\.', line)):
                occ.extend(fortrannums.findall(line))
              else:
                occ.extend(line.split())
        elif (line.startswith('#UOCC')):
          sections['UOCC'] = True
          line = '\n'
          if (uhf):
            f.readline()
            for i,b in zip(ii, self.N_bas):
              while (len(occ) < len(self.MO)+i+b):
                line = f.readline()
                if (re.search(r'\.[^ ]*\.', line)):
                  occ.extend(fortrannums.findall(line))
                else:
                  occ.extend(line.split())
        # Read the energies
        elif (line.startswith('#ONE')):
          sections['ONE'] = True
          line = '\n'
          f.readline()
          ene = []
          for i,b in zip(ii, self.N_bas):
            while (len(ene) < i+b):
              line = f.readline()
              if (re.search(r'\.[^ ]*\.', line)):
                ene.extend(fortrannums.findall(line))
              else:
                ene.extend(line.split())
        elif (line.startswith('#UONE')):
          sections['UONE'] = True
          line = '\n'
          if (uhf):
            f.readline()
            for i,b in zip(ii, self.N_bas):
              while (len(ene) < len(self.MO)+i+b):
                line = f.readline()
                if (re.search(r'\.[^ ]*\.', line)):
                  ene.extend(fortrannums.findall(line))
                else:
                  ene.extend(line.split())
        # Read the orbital types (same for alpha and beta)
        elif (line.startswith('#INDEX')):
          sections['INDEX'] = True
          line = '\n'
          idx = ''
          for i,b in zip(ii, self.N_bas):
            line = f.readline()
            while (len(idx) < i+b):
              idx += f.readline().split()[1]
          for i,o in enumerate(self.MO):
            o['type'] = idx[i].upper()
            o.pop('newtype', None)
          for i,o in enumerate(self.MO_b):
            o['type'] = idx[i].upper()
            o.pop('newtype', None)
        elif (line.startswith('#')):
          line = '\n'
      # Desymmetrize the orbital coefficients
      if (sections.get('ORB')):
        if (uhf and (not sections.get('UORB'))):
          return 'No UORB section'
        if (len(self.N_bas) > 1):
          for orb in self.MO + self.MO_b:
            orb['coeff'] = np.dot(self.mat, orb['coeff'])
      else:
        return 'No ORB section'
      # Assign occupations
      if (sections.get('OCC')):
        if (uhf and (not sections.get('UOCC'))):
          return 'No UOCC section'
        for i,o in enumerate(self.MO + self.MO_b):
          o['occup'] = float(occ[i])
      else:
        for o in self.MO + self.MO_b:
          o['occup'] = 0.0
      # Assign energies
      if (sections.get('ONE')):
        if (uhf and (not sections.get('UONE'))):
          return 'No UONE section'
        for i,o in enumerate(self.MO + self.MO_b):
          o['ene'] = float(ene[i])
      else:
        for o in self.MO + self.MO_b:
          o['ene'] = 0.0
    # Clear types
    if (not sections.get('INDEX')):
      for o in self.MO + self.MO_b:
        o['type'] = '?'
        o.pop('newtype', None)
    for o in self.MO + self.MO_b:
      o.pop('root_coeff', None)
      o.pop('root_coeffl', None)
      o.pop('root_coeffr', None)
      o.pop('root_occup', None)
      o.pop('root_type', None)
      o.pop('root_ene', None)

    if (self.MO_b):
      self.MO_a = deepcopy(self.MO)
      self.MO = []
    self.roots = ['InpOrb']
    self.dm = [np.diag([o['occup'] for o in self.MO if (o['type'] in ['1', '2', '3'])])]
    self.sdm = None
    self.tdm = None
    self.H_eff = None

    return True

  # Set the Cartesian coefficients for spherical harmonics
  def set_sph_c(self, maxl):
    # Get the coefficients for each value of l,m
    self.sph_c = []
    for l in range(maxl+1):
      s = {}
      for m in range(-l, l+1):
        s[m] = []
        # Go through all possible lx+ly+lz=l
        for lx in range(l+1):
          for ly in range(l-lx+1):
            lz = l - lx - ly
            # Get the coefficient (c_sph returns the square as a fraction with the sign)
            c = float(self._c_sph(l, m, lx, ly, lz))
            c = np.sign(c)*np.sqrt(abs(c))
            if (c != 0):
              s[m].append([c, [lx, ly, lz]])
      self.sph_c.append(s)
    # Now sph_c is a list of items for each l,
    # each item is a dict for each m,
    # each item is a list for each non-zero contribution,
    # each item is a list of coefficient and [lx, ly, lz] (x**lx * y**ly * z**lz)

  # Compute the angular component with quantum numbers l,m in an x,y,z grid
  # If cart=True, this is for a Cartesian shell
  def ang(self, x, y, z, l, m, cart=False):
    if (cart):
      # For Cartesian shells, m does not actually contain m, but:
      # m = T(ly+lz)-(lx+ly), where T(n) = n*(n+1)/2 is the nth triangular number
      ly = int(np.floor((np.sqrt(8*(m+l)+1)-1)/2))
      lz = m+l-ly*(ly+1)//2
      lx = l-ly
      ly -= lz
      assert (lx >= 0) and (ly >= 0) and (lz >= 0)
      c = np.sqrt(2**l)
      ang = c * x**lx * y**ly * z**lz
    else:
      ang = 0
      # Once sph_c has been computed, this is trivial
      for c in self.sph_c[l][m]:
        ang += c[0] * (x**c[1][0] * y**c[1][1] * z**c[1][2])
    return ang

  # Compute the radial component, with quantum number l, given the values of r**2 (as r2),
  # for a list of primitive Gaussians (exponents and coefficients, as ec)
  # and an optional power of r**2 (for contaminants)
  def rad(self, r2, l, ec, p=0, cache=None):
    rad = 0
    # For contaminants, the radial part is multiplied by r**(2*p)
    # and the normalization must be corrected, noting that the
    # angular part already includes a factor r**l
    if (p > 0):
      m = Fraction(2*l+1, 2*l+4*p+1)
      for i in range(2*l+1, 2*l+4*p, 2):
        m /= i
      m = np.sqrt(float(m))
      prad = np.power(r2, p)
    for e,c in ec:
      if (c != 0.0):
        if ((cache is None) or ((e,p) not in cache)):
          N = np.power((2*e)**(3+2*l)/np.pi**3, 0.25)
          if (p > 0):
            N *= m*np.power(4*e, p)
            cch = N * np.exp(-e*r2)*prad
          else:
            cch = N * np.exp(-e*r2)
          if (cache is not None):
            cache[(e,p)] = np.copy(cch)
        else:
          cch = cache[(e,p)]
        rad += c*cch
    return rad

  # Compute an atomic orbital as product of angular and radial components
  def ao(self, x, y, z, ec, l, m, p=0):
    ang = self.ang(x, y, z, l, m)
    r2 = x**2+y**2+z**2
    rad = self.rad(r2, l, ec, p)
    return ang*rad

  # Compute a molecular orbital, as linear combination of atomic orbitals
  # at different centers. It can use a cache of atomic orbitals to avoid
  # recomputing them. "spin" specifies if the coefficients will be taken
  # from self.MO (alpha) or self.MO_b (beta)
  def mo(self, n, x, y, z, spin='n', cache=None, callback=None, interrupt=False):
    mo = np.zeros_like(x)
    # Reorder MO coefficients
    if (spin == 'b'):
      MO = self.MO_b[n]
    elif (spin == 'a'):
      MO = self.MO_a[n]
    else:
      MO = self.MO[n]
    if ('root_coeff' in MO):
      MO = MO['root_coeff'][self.bf_sort]
    else:
      MO = MO['coeff'][self.bf_sort]

    if (callback is None):
      actions = [True]
    else:
      num = 0
      total = 0
      actions = [False, True]

    npoints = x.size
    if (cache is not None):
      chunk_size = cache.shape[1]
    use_cache = (cache is not None) and (chunk_size >= npoints)

    for compute in actions:
      f = 0
      # For each center, the relative x,y,z and r**2 are different
      for c in self.centers:
        x0, y0, z0 = [None]*3
        r2 = None
        # For each center, l and shell we have different radial parts
        for l,ll in enumerate(c['basis']):
          # Since all shells are computed for each m value, but the radial
          # part does not depend on m, we will save the radial part for
          # each shell to reuse it. This is a dict and not a list because
          # some shells could be skipped altogether
          rad_l = {}
          prim_cache = {}
          # For each center, l and m we have different angular parts
          # (the range includes both spherical and Cartesian indices)
          #for m in range(-l, l*(l+1)+1):
          for m in range(-l, l*(l+1)//2+1):
            ao_ang = None
            cart = None
            # Now each shell is an atomic orbital (basis function)
            for s,p in enumerate(ll):
              if (interrupt):
                return mo
              # Skip when out of range for spherical shells
              # Also invalidate the angular part if for some reason
              # there is a mixture of types among shells
              if ((l, s) in c['cart']):
                if (cart is False):
                  ang = None
                cart = True
              else:
                if (cart is True):
                  ang = None
                cart = False
                if (m > l):
                  continue
              # Only compute if above threshold
              if (abs(MO[f]) > self.eps):
                if (compute):
                  if callback is not None:
                    num += 1
                    callback('Computing: {0}/{1} ...'.format(num, total))
                  # The AO contribution is either in the cache
                  # or we compute it now
                  if (not use_cache or np.isnan(cache[f,0])):
                    # Compute relative coordinates if not done yet
                    if (x0 is None):
                      x0, y0, z0 = [x, y, z] - c['xyz'][:, np.newaxis]
                      r2 = x0**2 + y0**2 + z0**2
                    # Compute angular part if not done yet
                    if (ao_ang is None):
                      ao_ang = self.ang(x0, y0, z0, l, m, cart=cart)
                    # Compute radial part if not done yet
                    if (s not in rad_l):
                      rad_l[s] = self.rad(r2, l, p[1], p[0], cache=prim_cache)
                    cch = ao_ang*rad_l[s]
                    # Save in the cache if enabled
                    if (use_cache):
                      cache[f][0:cch.size] = np.copy(cch)
                  elif (use_cache):
                    cch = cache[f][0:x.size]
                  # Add the AO contribution to the MO
                  mo += MO[f]*cch
                else:
                  total += 1
              f += 1
    if (use_cache):
      cache.flush()
    return mo

  # Compute electron density as sum of square of (natural) orbitals times occupation.
  # It can use a cache for MO evaluation and a mask to select only some orbitals.
  def dens(self, x, y, z, cache=None, mask=None, spin=False, callback=None, interrupt=False):
    dens = np.zeros_like(x)
    if (self.MO_b):
      MO_list = [j for i in zip_longest(self.MO_a, self.MO_b) for j in i]
    else:
      MO_list = self.MO
    # If there is a callback function, we take two passes,
    # one to count the orbitals and another to actually compute them
    if (callback is None):
      actions = [True]
    else:
      total = 0
      actions = [False, True]

    npoints = x.size
    if (cache is not None):
      chunk_size = cache.shape[1]
    else:
      chunk_size = npoints
    chunk_list = list(range(0, npoints, chunk_size))

    for compute in actions:
      if (compute):
        do_list = chunk_list
      else:
        do_list = [0]
      for chunk,start in enumerate(do_list):
        if ((cache is not None) and (len(do_list) > 1)):
          cache[:,0] = np.nan
        x_ = x[start:start+chunk_size]
        y_ = y[start:start+chunk_size]
        z_ = z[start:start+chunk_size]
        num = 0
        j = 0
        for i,orb in enumerate(MO_list):
          if (interrupt):
            return dens
          if (orb is None):
            continue
          f = 1.0
          if (MO_list is self.MO):
            # Natural orbitals
            ii = i
            s = 'n'
          else:
            # Add alternated alpha and beta orbitals
            ii = i//2
            if (i%2 == 0):
              s = 'a'
            else:
              s = 'b'
              if (spin):
                f = -1.0
          if ((mask is None) or mask[j]):
            occup = f*orb.get('root_occup', orb['occup'])
            if (abs(occup) > self.eps):
              if (compute):
                if callback is not None:
                  num += 1
                  if (len(do_list) > 1):
                    callback('Computing: {0}/{1} (chunk {2}/{3}) ...'.format(num, total, chunk+1, len(do_list)))
                  else:
                    callback('Computing: {0}/{1} ...'.format(num, total))
                dens[start:start+chunk_size] += occup*self.mo(ii, x_, y_, z_, s, cache, interrupt=interrupt)**2
              else:
                total += 1
          j += 1
    return dens

  # Compute the Laplacian of a field by central finite differences
  def laplacian(self, box, field):
    n = field.shape
    box[:,0] /= n[0]-1
    box[:,1] /= n[1]-1
    box[:,2] /= n[2]-1
    g = np.linalg.inv(np.dot(box.T, box))
    data = -2*field*(sum(np.diag(g)))
    for i in range(n[0]):
      if ((i == 0) or (i == n[0]-1)):
        data[i,:,:] = None
      else:
        data[i,:,:] += (field[i-1,:,:]+field[i+1,:,:])*g[0,0]
        if (abs(g[0,1]) > 0):
          for j in range(1, n[1]-1):
            data[i,j,:] += (field[i-1,j-1,:]+field[i+1,j+1,:]-field[i-1,j+1,:]-field[i+1,j-1,:])*g[0,1]/2
        if (abs(g[0,2]) > 0):
          for k in range(1, n[2]-1):
            data[i,:,k] += (field[i-1,:,k-1]+field[i+1,:,k+1]-field[i-1,:,k+1]-field[i+1,:,k-1])*g[0,2]/2
    for j in range(n[1]):
      if ((j == 0) or (j == n[1]-1)):
        data[:,j,:] = None
      else:
        data[:,j,:] += (field[:,j-1,:]+field[:,j+1,:])*g[1,1]
        if (abs(g[1,2]) > 0):
          for k in range(1, n[2]-1):
            data[:,j,k] += (field[:,j-1,k-1]+field[:,j+1,k+1]-field[:,j-1,k+1]-field[:,j+1,k-1])*g[1,2]/2
    for k in range(n[2]):
      if ((k == 0) or (k == n[2]-1)):
        data[:,:,k] = None
      else:
        data[:,:,k] += (field[:,:,k-1]+field[:,:,k+1])*g[2,2]
    return data.flatten()

  # Returns binomial coefficient as a fraction
  # Easy overflow for large arguments, but we are interested in relatively small arguments
  def _binom(self, n, k):
    mk = max(k,n-k)
    try:
      binom = Fraction(np.math.factorial(n), np.math.factorial(mk))
      binom *= Fraction(1, np.math.factorial(n-mk))
      assert (binom.denominator == 1)
    except ValueError:
      binom = Fraction(0, 1)
    return binom

  # Computes the coefficient for x^lx * y^ly * z^lz in the expansion of
  # the real solid harmonic S(l,±m) = C * r^l*(Y(l,m)±Y(l,-m))
  # Since the coefficients are square roots of rational numbers, this
  # returns the square of the coefficient as a fraction, with its sign
  #
  # See:
  # Transformation between Cartesian and pure spherical harmonic Gaussians
  # doi: 10.1002/qua.560540202
  # (note that there appears to be a error in v(4,0), the coefficient 1/4
  #  should probably be 3/4*sqrt(3/35) )
  def _c_sph(self, l, m, lx, ly, lz):
    assert (lx + ly + lz == l) and (lx >= 0) and (ly >= 0) and (lz >= 0)
    am = abs(m)
    assert (am <= l)
    j = lx + ly - am
    if (j % 2 == 0):
      j = j//2
    else:
      return Fraction(0, 1)
    c = 0
    for i in range((l-am)//2+1):
      c += self._binom(l, i) * self._binom(i, j) * Fraction(np.math.factorial(2*l-2*i), np.math.factorial(l-am-2*i)) * (-1)**i
    if (c == 0):
      return Fraction(0, 1)
    c_sph = c
    c = 0
    for k in range(j+1):
      c += self._binom(j, k) * self._binom(am, lx-2*k) * 1j**(am-lx+2*k)
    if (m >= 0):
      c = int(np.real(c))
    else:
      c = int(np.imag(c))
    if (c == 0):
      return Fraction(0, 1)
    c_sph *= c
    if (c_sph < 0):
      c_sph *= -c_sph
    else:
      c_sph *= c_sph
    if (m == 0):
      lm = 1
    else:
      lm = 2
    c = Fraction(np.math.factorial(l-am), np.math.factorial(l+am))
    c *= Fraction(lm, np.math.factorial(l))
    c *= Fraction(1, np.math.factorial(2*l))
    c_sph *= c
    return c_sph

  # Writes a new HDF5 file
  def write_hdf5(self, filename):
    with h5py.File(self.h5file, 'r') as fi, h5py.File(filename, 'w') as fo:
      fo.attrs['Pegamoid_version'] = '{0} {1}'.format(__name__, __version__)
      # Copy some data from the original file
      for a in ['NSYM', 'NBAS', 'NPRIM', 'IRREP_LABELS', 'NATOMS_ALL', 'NATOMS_UNIQUE']:
        if (a in fi.attrs):
          fo.attrs[a] = fi.attrs[a]
      for d in ['CENTER_LABELS', 'CENTER_CHARGES', 'CENTER_COORDINATES', 'BASIS_FUNCTION_IDS',
                'DESYM_CENTER_LABELS', 'DESYM_CENTER_CHARGES', 'DESYM_CENTER_COORDINATES', 'DESYM_BASIS_FUNCTION_IDS', 'DESYM_MATRIX',
                'PRIMITIVES', 'PRIMITIVE_IDS']:
        if (d in fi):
          fi.copy(d, fo)
      if (len(self.N_bas) > 1):
        sym = np.linalg.inv(self.mat)
      else:
        sym = np.eye(sum(self.N_bas))
      # Write orbital data from current orbitals
      # (could be loaded from InpOrb, selected from a root and/or have modified types)
      uhf = len(self.MO_b) > 0
      nMO = [(sum(self.N_bas[:i]), sum(self.N_bas[:i+1])) for i in range(len(self.N_bas))]
      if (uhf):
        cff = []
        for i,j in nMO:
          for k in range(i,j):
            cff.extend(np.dot(sym, self.MO_a[k].get('root_coeff', self.MO_a[k]['coeff']))[i:j])
        fo.create_dataset('MO_ALPHA_VECTORS', data=cff)
        cff = []
        for i,j in nMO:
          for k in range(i,j):
            cff.extend(np.dot(sym, self.MO_b[k].get('root_coeff', self.MO_b[k]['coeff']))[i:j])
        fo.create_dataset('MO_BETA_VECTORS', data=cff)
        fo.create_dataset('MO_ALPHA_OCCUPATIONS', data=[o['occup'] for o in self.MO_a])
        fo.create_dataset('MO_BETA_OCCUPATIONS', data=[o['occup'] for o in self.MO_b])
        fo.create_dataset('MO_ALPHA_ENERGIES', data=[o['ene'] for o in self.MO_a])
        fo.create_dataset('MO_BETA_ENERGIES', data=[o['ene'] for o in self.MO_b])
        tp = [o.get('newtype', o['type']) for o in self.MO_a]
        for i,o in enumerate(self.MO_a):
          if (tp[i] == '?'):
            tp[i] = 'I' if (o['occup'] > 0.5) else 'S'
        fo.create_dataset('MO_ALPHA_TYPEINDICES', data=np.string_(tp))
        tp = [o.get('newtype', o['type'])for o in self.MO_b]
        for i,o in enumerate(self.MO_b):
          if (tp[i] == '?'):
            tp[i] = 'I' if (o['occup'] > 0.5) else 'S'
        fo.create_dataset('MO_BETA_TYPEINDICES', data=np.string_(tp))
      else:
        cff = []
        for i,j in nMO:
          for k in range(i,j):
            cff.extend(np.dot(sym, self.MO[k].get('root_coeff', self.MO[k]['coeff']))[i:j])
        fo.create_dataset('MO_VECTORS', data=cff)
        fo.create_dataset('MO_OCCUPATIONS', data=[o.get('root_occup', o['occup']) for o in self.MO])
        fo.create_dataset('MO_ENERGIES', data=[o.get('root_ene', o['ene']) for o in self.MO])
        tp = [o.get('newtype', o['type']) for o in self.MO]
        for i,o in enumerate(self.MO):
          if (tp[i] == '?'):
            tp[i] = 'I' if (o.get('root_occup', o['occup']) > 1.0) else 'S'
        fo.create_dataset('MO_TYPEINDICES', data=np.string_(tp))
      if (self.notes is not None):
        fo.create_dataset('Pegamoid_notes', data=np.string_(self.notes))

  # Creates an InpOrb file from scratch
  def create_inporb(self, filename, MO=None):
    nMO = OrderedDict()
    for i,n in zip(self.irrep, self.N_bas):
      nMO[i] = n
    uhf = (len(self.MO_b) > 0) and (MO is not self.MO)
    if (uhf):
      alphaMO = self.MO_a
      index, error = create_index(alphaMO, self.MO_b, nMO)
    else:
      alphaMO = self.MO
      index, error = create_index(alphaMO, [], nMO)
    if (index is None):
      if (error is not None):
        raise Exception(error)
      return
    if (len(self.N_bas) > 1):
      sym = np.linalg.inv(self.mat)
    else:
      sym = np.eye(sum(self.N_bas))
    nMO = [(sum(self.N_bas[:i]), sum(self.N_bas[:i+1])) for i in range(len(self.N_bas))]
    with open(filename, 'w') as f:
      f.write('#INPORB 2.2\n')
      f.write('#INFO\n')
      f.write('* File generated by {0} from {1}\n'.format(__name__, self.file))
      f.write(wrap_list([int(uhf), len(self.N_bas), 0], 3, '{:8d}')[0])
      f.write('\n')
      f.write(wrap_list(self.N_bas, 8, '{:8d}')[0])
      f.write('\n')
      f.write(wrap_list(self.N_bas, 8, '{:8d}')[0])
      f.write('\n')
      f.write('*BC:HOST {0} PID {1} DATE {2}\n'.format(gethostname(), os.getpid(), datetime.now().ctime()))
      f.write('#ORB\n')
      for s,(i,j) in enumerate(nMO):
        for k in range(i,j):
          f.write('* ORBITAL{0:5d}{1:5d}\n'.format(s+1, k-i+1))
          cff = alphaMO[k].get('root_coeff', alphaMO[k]['coeff'])
          cff = np.dot(sym, cff)
          cff = wrap_list(cff[i:j], 5, '{:21.14E}', sep=' ')
          f.write(' ' + '\n '.join(cff) + '\n')
      if (uhf):
        f.write('#UORB\n')
        for s,(i,j) in enumerate(nMO):
          for k in range(i,j):
            f.write('* ORBITAL{0:5d}{1:5d}\n'.format(s+1, k-i+1))
            cff = self.MO_b[k].get('root_coeff', self.MO_b[k]['coeff'])
            cff = np.dot(sym, cff)
            cff = wrap_list(cff[i:j], 5, '{:21.14E}', sep=' ')
            f.write(' ' + '\n '.join(cff) + '\n')
      f.write('#OCC\n')
      f.write('* OCCUPATION NUMBERS\n')
      for i,j in nMO:
        occ = wrap_list([o.get('root_occup', o['occup']) for o in alphaMO[i:j]], 5, '{:21.14E}', sep=' ')
        f.write(' ' + '\n '.join(occ) + '\n')
      if (uhf):
        f.write('#UOCC\n')
        f.write('* Beta OCCUPATION NUMBERS\n')
        for i,j in nMO:
          occ = wrap_list([o.get('root_occup', o['occup']) for o in self.MO_b[i:j]], 5, '{:21.14E}', sep=' ')
          f.write(' ' + '\n '.join(occ) + '\n')
      f.write('#ONE\n')
      f.write('* ONE ELECTRON ENERGIES\n')
      for i,j in nMO:
        ene = wrap_list([o.get('root_ene', o['ene']) for o in alphaMO[i:j]], 10, '{:11.4E}', sep=' ')
        f.write(' ' + '\n '.join(ene) + '\n')
      if (uhf):
        f.write('#UONE\n')
        f.write('* Beta ONE ELECTRON ENERGIES\n')
        for i,j in nMO:
          ene = wrap_list([o.get('root_ene', o['ene']) for o in self.MO_b[i:j]], 10, '{:11.4E}', sep=' ')
          f.write(' ' + '\n '.join(ene) + '\n')
      f.write('#INDEX\n')
      f.write('\n'.join(index))
      f.write('\n')

if __name__ == "__main__" :
    file_name = '/home/alessio/config/Stephan/zNorbornadiene_P000-000_P017-136_P114-766.rasscf.h5'
    a = Orbitals(file_name, 'hdf5')
    single = a.mo(0,np.array([-10.0]),np.array([-10.0]),np.array([-10.0]))
    print(single)


