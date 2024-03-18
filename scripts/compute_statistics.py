import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats

from modules.io import plot_init


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_name',
                   help='Path of the output name.')
    
    p.add_argument('--in_results', nargs='+', required=True,
                   help='List of all results directories.')
    
    p.add_argument('--measures', nargs='+', required=True,
                   help='List of all measures to analyse.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    min_nb_voxels = args.min_nb_voxels

    print(args.in_results)

    results = []
    for i, result in enumerate(args.in_results):
        print("Loading: ", result)
        results.append(np.load(result))

    nb_bins = len(results[0]['Angle_min'])
    nb_results = len(results)

    for measure in args.measures:
        print(measure)
        to_analyse = np.zeros((nb_results, nb_bins))
        for i, result in enumerate(results):
            to_analyse[i] = result[measure]

        coeff_var = scipy.stats.variation(to_analyse, axis=0, nan_policy='omit')
        print(np.mean(coeff_var))


if __name__ == "__main__":
    main()
