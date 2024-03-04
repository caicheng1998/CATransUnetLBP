import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import data_utils
import numpy as np
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing
from openbabel import pybel, openbabel

# tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

from data_utils.data import Featurizer


def pocket_density_from_mol(model, mol):
    """Predict porobability density of pockets using pybel.Molecule object
    as input"""

    if not isinstance(mol, pybel.Molecule):
        raise TypeError('mol should be a pybel.Molecule object, got %s '
                        'instead' % type(mol))
    if model.featurizer is None:
        raise ValueError('featurizer must be set to make predistions for '
                         'molecules')
    if model.scale is None:
        raise ValueError('scale must be set to make predistions')
    prot_coords, prot_features = model.featurizer.get_features(mol)
    centroid = prot_coords.mean(axis=0)
    prot_coords -= centroid
    resolution = 1. / model.scale
    x = data_utils.data.make_grid(prot_coords, prot_features,
                             max_dist=model.max_dist,
                             grid_resolution=resolution)

    density = model.predict(x)
    origin = (centroid - model.max_dist)
    step = np.array([1.0 / model.scale] * 3)
    return density, origin, step


def pocket_density_from_grid(model, pdbid):
    """Predict porobability density of pockets using 3D grid (np.ndarray)
    as input"""

    if model.data_handle is None:
        raise ValueError('data_handle must be set to make predictions '
                         'using PDBIDs')
    if model.scale is None:
        raise ValueError('scale must be set to make predistions')
    x, _ = model.data_handle.prepare_complex(pdbid)
    origin = (model.data_handle[pdbid]['centroid'][:] - model.max_dist)
    step = np.array([1.0 / model.scale] * 3)
    density = model.predict(x)
    return density, origin, step


def get_pockets_segmentation(model, density, threshold=0.5, min_size=48):
    """Predict pockets using specified threshold on the probability density.
    Filter out pockets smaller than min_size A^3
    """

    if len(density) != 1:
        raise ValueError('segmentation of more than one pocket is not'
                         ' supported')

    voxel_size = (1 / model.scale) ** 3
    # get a general shape, without distinguishing output channels
    bw = closing((density[0] > threshold).any(axis=-1))

    # remove artifacts connected to border
    cleared = clear_border(bw)

    # label regions
    label_image, num_labels = label(cleared, return_num=True)
    for i in range(1, num_labels + 1):
        pocket_idx = (label_image == i)
        pocket_size = pocket_idx.sum() * voxel_size
        if pocket_size < min_size:
            label_image[np.where(pocket_idx)] = 0
    return label_image


def predict_pocket_atoms(mol, origin, step, pockets, dist_cutoff=4.5, expand_residue=True,):
    """Predict pockets for a given molecule and get AAs forming them
    Parameters
    ----------
    mol: pybel.Molecule object
        Protein structure
    dist_cutoff: float, optional (default=4.5)
        Maximal distance between protein atom and predicted pocket
    expand_residue: bool, optional (default=True)
        Inlude whole residue if at least one atom is included in the pocket
    pocket_kwargs:
        Keyword argument passed to `get_pockets_segmentation` method

    Returns
    -------
    pocket_mols: list of pybel.Molecule objects
        Fragments of molecule corresponding to detected pockets.
    """

    from scipy.spatial.distance import cdist

    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    residue2atom = np.array([[a.idx - 1 for a in r.atoms]
                             for r in mol.residues])


    # find atoms close to pockets
    pocket_atoms = []
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        distance = cdist(coords, indices)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])
        pocket_atoms.append([int(idx) for idx in close_atoms])

    # create molecules correcponding to atom indices
    pocket_mols = []
    for pocket in pocket_atoms:
        # copy molecule
        pocket_mol = mol.clone
        atoms_to_del = (set(range(len(pocket_mol.atoms)))
                        - set(pocket))
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify()
        pocket_mols.append(pocket_mol)

    return pocket_mols


def save_pocket_mol2(model, mol, path, format, mol_name, **pocket_kwargs):
    density, origin, step = pocket_density_from_mol(model, mol)
    pockets = get_pockets_segmentation(model, density, **pocket_kwargs)
    atom_pockets = predict_pocket_atoms(mol, origin, step, pockets)
    i = 0
    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= step
        indices += origin
        mol = openbabel.OBMol()
        for idx in indices:
            a = mol.NewAtom()
            a.SetAtomicNum(1)
            a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
        p_mol = pybel.Molecule(mol)
        p_mol.write(format, path + '/' + mol_name + '_pocket' + str(i) + '.' + format, overwrite=True)
        i += 1
    for j, pocket in enumerate(atom_pockets):
        pocket.write('mol2', os.path.join(path, mol_name + 'pocket%i_residues.mol2' % j), overwrite=True)
    print(str(pockets.max()) + " pocket(s) generated, file at:" + path + '/')


