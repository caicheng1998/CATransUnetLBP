import argparse
import os
import sys
import functools

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import time
import pandas as pd
from CATransUnet import CATransUnet, dice_loss, dice
from data_utils.dataset import Binding_pocket_dataset

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

def input_path(path):
    """Check if input exists."""

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('%s does not exist.' % path)
    return path


def output_path(path):
    path = os.path.abspath(path)
    return path


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', '-i', required=True, type=input_path,
                        help='path to the .hdf file with prepared data. ')
    parser.add_argument('--test_hdf', '-t', type=input_path,
                        help='path to the .hdf file with prepared data. ')
    parser.add_argument('--model', '-m', type=input_path,
                        help='path to the .hdf file with pretrained model. '
                             'If not specified, a new model will be trained from scratch.')
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--steps_per_epoch', default=850, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_ids', type=input_path,
                        help='text file with IDs to use for training (each in separate line). '
                             'If not specified, all proteins in the database '
                             '(except those listed with --test_ids) will be used. '
                             'Note that --test_ids has higher priority (i.e. if '
                             'ID is specified with both --train_ids and '
                             '--test_ids, it will be in the test set)')
    parser.add_argument('--test_ids', type=input_path,
                        help='text file with IDs to use for testing (each in separate line). '
                             'If not specified, all proteins will be used for training. '
                             'This option has higher priority than --train_ids (i.e. if '
                             'ID is specified with both --train_ids and '
                             '--test_ids, it will be in the test set)')
    parser.add_argument('--load', '-l', action='store_true',
                        help='whether to load all data into memory')
    parser.add_argument('--output', '-o', type=output_path,
                        help='name for the output directory. If not specified, '
                             '"output_<YYYY>-<MM>-<DD>" will be used')
    parser.add_argument('--verbose', '-v', default=2, type=int,
                        help='verbosity level for keras')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        args.output = 'output_' + time.strftime('%Y-%m-%d')
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.access(args.output, os.W_OK):
        raise IOError('Cannot create files inside %s (check your permissions).' % args.output)

    if args.train_ids:
        with open(args.train_ids) as f:
            train_ids = list(filter(None, f.read().split('\n')))
    else:
        train_ids = None

    if args.test_ids:
        with open(args.test_ids) as f:
            test_ids = list(filter(None, f.read().split('\n')))
    else:
        test_ids = None

    train_dataset = Binding_pocket_dataset(args.input, max_dist=35, grid_resolution=2,
                                           ids=train_ids, augment=True, batch_size=args.batch_size)
    test_dataset = None
    if args.test_hdf is not None and test_ids is not None:
        test_dataset = Binding_pocket_dataset(args.test_hdf, max_dist=35, grid_resolution=2,
                                              ids=test_ids, augment=False, batch_size=args.batch_size)

    if args.model:
        model = CATransUnet.load_model(args.model)
    else:
        lr = 1e-4
        model = CATransUnet()
        model.compile(optimizer=Adam(lr=lr), loss=dice_loss,
                      metrics=[dice])


    callbacks = [ModelCheckpoint(os.path.join(args.output, 'checkpoint'),
                                 save_best_only=False)]

    if test_ids:
        num_val_steps = 150
        callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                               patience=3, min_lr=1e-10)
        callback_early_stop = EarlyStopping(monitor='val_loss', patience=10,
                                            restore_best_weights=True)
        callbacks.append(callback_reduce_lr)
        callbacks.append(callback_early_stop)

        callbacks.append(ModelCheckpoint(os.path.join(args.output, 'best_weights'),
                                         save_best_only=True))

    else:
        num_val_steps = None

    workers = 4
    use_multiprocessing = True
    model.fit(train_dataset, steps_per_epoch=args.steps_per_epoch,
              epochs=args.epochs, verbose=args.verbose, callbacks=callbacks,
              validation_data=test_dataset, validation_steps=num_val_steps,
              workers=workers, use_multiprocessing=use_multiprocessing)

    history = pd.DataFrame(model.history.history)
    history.to_csv(os.path.join(args.output, 'history.csv'))
    model.save(os.path.join(args.output, 'model.hdf'), save_format="tf")


if __name__ == '__main__':
    main()
