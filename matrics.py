import numpy as np
from openbabel import pybel
from scipy.spatial.distance import euclidean


def make_3d_grid(pocket, min_coords, max_coords, resolution):
    shifted_coords = pocket - min_coords

    grid_shape = np.ceil((max_coords) / resolution).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=bool)

    for cell in shifted_coords:
        cell_idx = np.floor(cell / resolution).astype(int)
        grid[tuple(cell_idx)] = True
    return grid


def get_IOU(pocket1, pocket2, resolution):
    min_coords = np.min(np.min(pocket1, axis=0), np.min(pocket2, axis=0))
    max_coords = np.max(np.max(pocket1, axis=0), np.max(pocket2, axis=0))

    grid1 = make_3d_grid(pocket1, min_coords, max_coords, resolution)
    grid2 = make_3d_grid(pocket2, min_coords, max_coords, resolution)

    intersection_grid = np.logical_and(grid1, grid2)
    union_grid = np.logical_or(grid1, grid2)

    intersection = np.sum(intersection_grid)
    union = np.sum(union_grid)

    return intersection / union


def get_coordinates(mol_file):
    molecule = next(pybel.readfile(mol_file.split('.')[-1], mol_file))
    mol_coords = [atom.coords for atom in molecule.atoms]
    return np.array(mol_coords)


def get_DCC(pkt1, pkt2):
    pocket1_coords = get_coordinates(pkt1)
    pocket2_coords = get_coordinates(pkt2)
    dcc = euclidean(pocket1_coords.mean(axis=0), pocket2_coords.mean(axis=0))
    return dcc


def get_DVO(pkt1, pkt2, resolution=2):
    pocket1_coords = get_coordinates(pkt1)
    pocket2_coords = get_coordinates(pkt2)
    return get_IOU(pocket1_coords, pocket2_coords, resolution)

