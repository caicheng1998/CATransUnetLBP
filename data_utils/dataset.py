from tensorflow.keras.utils import Sequence
from random import choice
from skimage.draw import ellipsoid
from scipy import ndimage
import numpy as np
from math import sin, cos, sqrt, pi
from itertools import combinations
import h5py


def rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=np.float)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


ROTATIONS = [rotation_matrix([1, 1, 1], 0)]

for a1 in range(3):
    for t in range(1, 4):
        axis = np.zeros(3)
        axis[a1] = 1
        theta = t * pi / 2.0
        ROTATIONS.append(rotation_matrix(axis, theta))

for (a1, a2) in combinations(range(3), 2):
    axis = np.zeros(3)
    axis[[a1, a2]] = 1.0
    theta = pi
    ROTATIONS.append(rotation_matrix(axis, theta))
    axis[a2] = -1.0
    ROTATIONS.append(rotation_matrix(axis, theta))

for t in [1, 2]:
    theta = t * 2 * pi / 3
    axis = np.ones(3)
    ROTATIONS.append(rotation_matrix(axis, theta))
    for a1 in range(3):
        axis = np.ones(3)
        axis[a1] = -1
        ROTATIONS.append(rotation_matrix(axis, theta))


class Binding_pocket_dataset(Sequence):

    def __init__(self, hdf_path, max_dist, grid_resolution, batch_size=20, ids=None, augment=False):

        self.transform = augment
        self.max_dist = max_dist
        self.grid_resolution = grid_resolution
        self.hdf_path = hdf_path
        self.data_handle = None
        self.ids = ids
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.data_handle is None:
            self.data_handle = h5py.File(self.hdf_path, 'r')
        batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        rec_grid_batch = []
        pocket_dens_batch = []
        for pdbid in batch_ids:
            if self.transform:
                rot = choice(range(24))
                tr = 5 * np.random.rand(1, 3)
            else:
                rot = 0
                tr = (0, 0, 0)
            rec_grid, pocket_dens = self.prepare_complex(pdbid, rotation=rot, translation=tr)
            rec_grid_batch.append(rec_grid)
            pocket_dens_batch.append(pocket_dens)

        return np.stack(rec_grid_batch), np.stack(pocket_dens_batch)

    def make_grid(self, coords, features, grid_resolution=1.0, max_dist=10.0):

        coords = np.asarray(coords, dtype=np.float)
        c_shape = coords.shape
        N = len(coords)
        features = np.asarray(features, dtype=np.float)
        f_shape = features.shape
        num_features = f_shape[1]
        max_dist = float(max_dist)
        grid_resolution = float(grid_resolution)

        box_size = int(np.ceil(2 * max_dist / grid_resolution + 1))

        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)

        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        grid = np.zeros((1, box_size, box_size, box_size, num_features), dtype=np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[0, x, y, z] += f

        return grid

    def rotate(self, coords, rotation):
        global ROTATIONS
        return np.dot(coords, ROTATIONS[rotation])

    def prepare_complex(self, pdbid, rotation=0, translation=(0, 0, 0), vmin=0, vmax=1):

        prot_coords = self.data_handle[pdbid]['coords'][:]
        poc_coords = self.data_handle[pdbid]['pocket_coords'][:]
        poc_features = self.data_handle[pdbid]['pocket_features'][:]
        prot_features = self.data_handle[pdbid]['features'][:]
        prot_coords = self.rotate(prot_coords, rotation)
        prot_coords += translation
        poc_coords = self.rotate(poc_coords, rotation)
        poc_coords += translation
        footprint = ellipsoid(2, 2, 2)
        footprint = footprint.reshape((1, *footprint.shape, 1))

        rec_grid = self.make_grid(prot_coords, prot_features,
                                  max_dist=self.max_dist,
                                  grid_resolution=self.grid_resolution)

        pocket_dens = self.make_grid(poc_coords, poc_features,
                                     max_dist=self.max_dist)
        margin = ndimage.maximum_filter(pocket_dens, footprint=footprint)
        pocket_dens += margin
        pocket_dens = pocket_dens.clip(vmin, vmax)

        zoom = rec_grid.shape[1] / pocket_dens.shape[1]
        pocket_dens = np.stack([ndimage.zoom(pocket_dens[0, ..., i], zoom)
                                for i in range(poc_features.shape[1])], -1)
        rec_grid = np.squeeze(rec_grid)
        return rec_grid, pocket_dens


