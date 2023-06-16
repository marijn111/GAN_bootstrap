import os
import numpy as np
import pickle
import argparse


def main(args):
    dgp_types = ['ar', 'arma', 'ma', 'farima', 'garch']
    monte_carlo_sims = args.monte_carlo_sims

    for dgp in dgp_types:
        pop_mean_presence_99 = 0
        pop_mean_presence_90 = 0
        empirical_length_99 = []
        empirical_length_90 = []

        for i in range(1, monte_carlo_sims+1):
            with open(os.path.join(args.data_path, dgp, f'final_output_run_{i}.pkl'), 'rb') as f:
                output = pickle.load(f)
            qt_99 = np.percentile(output['presence'], [1, 99])
            qt_90 = np.percentile(output['presence'], [10, 90])

            if qt_99[0] < 0 < qt_99[1]:
                pop_mean_presence_99 += 1

            if qt_90[0] < 0 < qt_90[1]:
                pop_mean_presence_90 += 1

            empirical_length_99.append(abs(qt_99[0] - qt_99[1]))
            empirical_length_90.append(abs(qt_90[0] - qt_90[1]))

        print(f'{dgp}:')
        print(f'Pop mean presence 99%: {pop_mean_presence_99/monte_carlo_sims}')
        print(f'Pop mean presence 90%: {pop_mean_presence_90/monte_carlo_sims}')
        print(f'Empirical length 99%: {np.mean(empirical_length_99)}')
        print(f'Empirical length 90%: {np.mean(empirical_length_90)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_path',
        help='path to the data',
        type=str,
    )
    parser.add_argument(
        '--monte_carlo_sims',
        help='number of simulation runs',
        type=int,
        default=800
    )

    args = parser.parse_args()
    main(args)
