import argparse
from predict_fuctions import save_pocket_mol2, predict_pocket_atoms
from CATransUnet import CATransUnet
from data_utils.data import Featurizer
import time
from openbabel import pybel
import os

import warnings
warnings.filterwarnings("ignore", category=Warning)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model_path = 'model_dir/model.tf'


def input_path(path):
    """Check if input exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' % path)
    return path


def output_path(path):
    path = os.path.abspath(path)
    return path


def myprint(s):
    with open('/modelsummary.txt', 'a') as f:
        f.write(s + '\n')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', required=True, type=input_path, nargs='+',
                        help='paths to protein structures')
    parser.add_argument('--model', '-m', type=input_path, default=model_path,
                        help='path to the .hdf file with trained model.')
    parser.add_argument('--output', '-o', type=output_path,
                        help='name for the output directory. If not specified, '
                             '"pockets_<YYYY>-<MM>-<DD>" will be used')
    parser.add_argument('--format', '-f', default='mol2',
                        help='input format; can be any format for 3D structures'
                             ' supported by Open Babel')
    parser.add_argument('--max_dist', type=float, default=35,
                        help='max_dist parameter used for training set')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scale parameter used for training set')

    return parser.parse_args()


def main():

    args = parse_args()

    if args.output is None:
        args.output = 'pockets_' + time.strftime('%Y-%m-%d')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.access(args.output, os.W_OK):
        raise IOError('Cannot create files inside %s (check your permissions).' % args.output)

    model = CATransUnet.load_model(model_path, scale=args.scale, max_dist=args.max_dist,
                                   featurizer=Featurizer(save_molecule_codes=False))
    mol_path = args.input[0]
    mol_file = mol_path.split("/")[-1]
    mol_name = mol_file.split(".")[0]
    mol_format = mol_file.split(".")[-1]

    mol = next(pybel.readfile(mol_format, mol_path))

    save_pocket_mol2(model, mol, args.output, "mol2", mol_name, threshold=0.5, min_size=48)


if __name__ == '__main__':
    main()
