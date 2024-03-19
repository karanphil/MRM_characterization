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
    
    p.add_argument('--names', nargs='+', required=True, action='append',
                   help='List of names.')

    p.add_argument('--variation', action='store_true')

    p.add_argument('--correlation', action='store_true')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    min_nb_voxels = args.min_nb_voxels
    names = args.names[0]

    print(args.in_results)
    print(names)

    results = []
    for i, result in enumerate(args.in_results):
        if str(Path(result).parent) in names:
            print("Loading: ", result)
            results.append(np.load(result))

    print(results)
    nb_bins = len(results[0]['Angle_min'])
    nb_results = len(results)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    plot_init()

    for measure in args.measures:
        print(measure)
        to_analyse = np.zeros((nb_results, nb_bins))
        for i, result in enumerate(results):
            to_analyse[i] = result[measure]

        if args.variation:
            coeff_var = scipy.stats.variation(to_analyse, axis=0,
                                              nan_policy='omit')
            print(np.mean(coeff_var))

        if args.correlation:
            # to_analyse = np.ma.masked_values(to_analyse, np.nan)
            print(to_analyse)
            corr = np.ma.corrcoef(to_analyse, allow_masked=True)
            print(corr)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(corr, cmap=cm.navia, norm=norm)
            fig.colorbar(cax)
            ax.set_xticklabels(names)
            ax.set_yticklabels(names)
            plt.show()


if __name__ == "__main__":
    main()
