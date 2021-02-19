"""
Handle arguments for train/test scripts.

The MIT License (MIT)
Originally created at 5/25/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Ahmed (@gmail.com)
"""

import argparse
import json
import pprint
import os.path as osp
from datetime import datetime
from argparse import ArgumentParser
from ..utils import str2bool, create_dir


def parse_arguments(notebook_options=None):
    """Parse the arguments for the training (or test) execution of a ReferIt3D net.
    :param notebook_options: (list) e.g., ['--max-distractors', '100'] to give/parse arguments from inside a jupyter notebook.
    :return:
    """
    parser = argparse.ArgumentParser(description='ReferIt3D Nets + Ablations')

    #
    # Non-optional arguments
    #
    parser.add_argument('-scannet-file', type=str, required=True, help='pkl file containing the data of Scannet'
                                                                       ' as generated by running XXX')
    parser.add_argument('-referit3D-file', type=str, required=True)

    #
    # I/O file-related arguments
    #
    parser.add_argument('--log-dir', type=str, help='where to save training-progress, model, etc')
    parser.add_argument('--resume-path', type=str, help='model-path to resume')
    parser.add_argument('--config-file', type=str, default=None, help='config file')

    #
    # Dataset-oriented arguments
    #
    parser.add_argument('--max-distractors', type=int, default=51,
                        help='Maximum number of distracting objects to be drawn from a scan.')
    parser.add_argument('--max-seq-len', type=int, default=24,
                        help='utterances with more tokens than this they will be ignored.')
    parser.add_argument('--points-per-object', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')
    parser.add_argument('--unit-sphere-norm', type=str2bool, default=False,
                        help="Normalize each point-cloud to be in a unit sphere.")
    parser.add_argument('--mentions-target-class-only', type=str2bool, default=True,
                        help='If True, drop references that do not explicitly mention the target-class.')
    parser.add_argument('--min-word-freq', type=int, default=3)
    parser.add_argument('--max-test-objects', type=int, default=88)

    #
    # Training arguments
    #
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'])
    parser.add_argument('--max-train-epochs', type=int, default=100, help='number of training epochs. [default: 100]')
    parser.add_argument('--n-workers', type=int, default=-1,
                        help='number of data loading workers [default: -1 is all cores available -1.]')
    parser.add_argument('--random-seed', type=int, default=2020,
                        help='Control pseudo-randomness (net-wise, point-cloud sampling etc.) fostering reproducibility.')
    parser.add_argument('--init-lr', type=float, default=0.0005, help='learning rate for training.')
    parser.add_argument('--patience', type=int, default=10, help='if test-acc does not improve for patience consecutive'
                                                                 'epoch, stop training.')

    #
    # Model arguments
    #
    parser.add_argument('--model', type=str, default='referIt3DNet', choices=['referIt3DNet',
                                                                              'directObj2Lang',
                                                                              'referIt3DNetAttentive'])
    parser.add_argument('--object-latent-dim', type=int, default=128)
    parser.add_argument('--language-latent-dim', type=int, default=128)
    parser.add_argument('--word-embedding-dim', type=int, default=64)
    parser.add_argument('--graph-out-dim', type=int, default=128)
    parser.add_argument('--dgcnn-intermediate-feat-dim', nargs='+', type=int, default=[128, 128, 128, 128])

    parser.add_argument('--object-encoder', type=str, default='pnet_pp', choices=['pnet_pp', 'pnet'])
    parser.add_argument('--language-fusion', type=str, default='both', choices=['before', 'after', 'both'])
    parser.add_argument('--word-dropout', type=float, default=0.1)
    parser.add_argument('--knn', type=int, default=7, help='For DGCNN number of neighbors')
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')

    #
    # Misc arguments
    #
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device. [default: 0]')
    parser.add_argument('--n-gpus', type=int, default=1, help='number gpu devices. [default: 1]')
    parser.add_argument('--dist-mgr', type=str, help='distributed training manager module path',
                        default='referit3d.models.dist_mgr.local.LocalDistMgr')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size per gpu. [default: 32]')
    parser.add_argument('--save-args', type=str2bool, default=True, help='save arguments in a json.txt')
    parser.add_argument('--experiment-tag', type=str, default=None, help='will be used to name a subdir '
                                                                         'for log-dir if given')
    parser.add_argument('--cluster-pid', type=str, default=None)

    #
    # "Joint" (Sr3d+Nr3D) training
    #
    parser.add_argument('--augment-with-sr3d', type=str, default=None,
                        help='csv with sr3d data to augment training data'
                             'of args.referit3D-file')
    parser.add_argument('--vocab-file', type=str, default=None, help='optional, .pkl file for vocabulary (useful when '
                                                                     'working with multiple dataset and single model.')
    parser.add_argument('--fine-tune', type=str2bool, default=False,
                        help='use if you train with dataset x and then you '
                             'continue training with another dataset')
    parser.add_argument('--s-vs-n-weight', type=float, default=None, help='importance weight of sr3d vs nr3d '
                                                                          'examples [use less than 1]')

    # Parse args
    if notebook_options is not None:
        args = parser.parse_args(notebook_options)
    else:
        args = parser.parse_args()

    if not args.resume_path and not args.log_dir:
        raise ValueError


def prepare_log_files(args):
    if args.config_file is not None:
        with open(args.config_file, 'r') as fin:
            configs_dict = json.load(fin)
            apply_configs(args, configs_dict)

    # Create logging related folders and arguments
    if args.log_dir:
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        if args.experiment_tag:
            args.log_dir = osp.join(args.log_dir, args.experiment_tag, timestamp)
        else:
            args.log_dir = osp.join(args.log_dir, timestamp)

        args.checkpoint_dir = create_dir(osp.join(args.log_dir, 'checkpoints'))
        args.tensorboard_dir = create_dir(osp.join(args.log_dir, 'tb_logs'))

    if args.resume_path and not args.log_dir:  # resume and continue training in previous log-dir.
        checkpoint_dir = osp.split(args.resume_path)[0]  # '/xxx/yyy/log_dir/checkpoints/model.pth'
        args.checkpoint_dir = checkpoint_dir
        args.log_dir = osp.split(checkpoint_dir)[0]
        args.tensorboard_dir = osp.join(args.log_dir, 'tb_logs')

    # Print them nicely
    args_string = pprint.pformat(vars(args))
    print(args_string)

    if args.save_args:
        out = osp.join(args.log_dir, 'config.json.txt')
        with open(out, 'w') as f_out:
            json.dump(vars(args), f_out, indent=4, sort_keys=True)

    return args


def read_saved_args(config_file, override_args=None, verbose=True):
    """
    :param config_file:
    :param override_args: dict e.g., {'gpu': '0'}
    :param verbose:
    :return:
    """
    parser = ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, 'r') as f_in:
        args.__dict__ = json.load(f_in)

    if override_args is not None:
        for key, val in override_args.items():
            args.__setattr__(key, val)

    if verbose:
        args_string = pprint.pformat(vars(args))
        print(args_string)

    return args


def apply_configs(args, config_dict):
    for k, v in config_dict.items():
        setattr(args, k, v)
