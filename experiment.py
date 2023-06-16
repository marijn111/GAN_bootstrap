"""
experiment.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 1. TimeGAN model
from timegan import TimeGAN
# TODO add alternatives approaches.
# 2. Data loading
from data_loading import data_loading


import argparse
import os
import numpy as np
import warnings
import json

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(args):
    """Main function for experiments.

    Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - model: model used for this experiment
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation

    Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
    """
    print(f'Starting training loop')

    # Data loading
    data_path = os.path.join(args.data_path, args.dgp_type, f'{args.dgp_type}_10k_{args.run_id}.csv')
    data, min_val, max_val = data_loading(data_path, seq_len=args.seq_len)

    print(args.data_name + ' dataset is ready.')

    # Synthetic data generation
    # Set network parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations_embedding'] = args.iteration_embedding
    parameters['iterations_supervised'] = args.iteration_supervised
    parameters['iterations_joint'] = args.iteration_joint
    parameters['batch_size'] = args.batch_size
    parameters['output_path'] = args.output_path
    parameters['run_id'] = args.run_id
    parameters['dgp_type'] = args.dgp_type

    timegan = TimeGAN(data, parameters, min_value=min_val, max_value=max_val)
    losses, times = timegan.train()

    print(f'Finish Synthetic Data Generation for loop')

    return data


if __name__ == '__main__':

    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        choices=['timegan'],
        default='timegan',
        type=str)
    parser.add_argument(
        '--data_name',
        choices=['dgp'],
        default='dgp',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=24,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru'],
        default='gru',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=24,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration_embedding',
        help='Training iterations for embedding phase (should be optimized)',
        default=10000,
        type=int)
    parser.add_argument(
        '--iteration_supervised',
        help='Training iterations for supervised phase (should be optimized)',
        default=10000,
        type=int)
    parser.add_argument(
        '--iteration_joint',
        help='Training iterations for joint phase (should be optimized)',
        default=10000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)
    parser.add_argument(
        '--data_path',
        help='path to the data',
        type=str,
    )
    parser.add_argument(
        '--output_path',
        help='path to the output',
        type=str,
    )
    parser.add_argument(
        '--dgp_type',
        help='type of dgp',
        choices=['arma', 'ar', 'ma', 'farima', 'garch'],
        type=str,
        default='arma'
    )
    parser.add_argument(
        '--run_id',
        help='run id',
        type=int,
        default=0
    )

    args = parser.parse_args()
    ori_data = main(args)
