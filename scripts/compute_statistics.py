import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
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
    
    p.add_argument('--names', nargs='+',
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
    names = []
    if args.names:
        names = args.names
        print(names)

    print(args.in_results)

    results = []
    for i, result in enumerate(args.in_results):
        if args.names:
            if str(Path(result).parent) in names:
                print("Loading: ", result)
                results.append(np.load(result))
        else:
            print("Loading: ", result)
            results.append(np.load(result))
            names.append(str(Path(result).parent))

    print(results)
    nb_bins = len(results[0]['Angle_min'])
    nb_results = len(results)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    plot_init(font_size=8, dims=(10, 10))

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
            dataset = pd.DataFrame(data=to_analyse.T)
            corr = dataset.corr()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(corr, cmap=cm.navia_r, norm=norm) # abs? Que faire des coeff négatifs?
            fig.colorbar(cax)
            ax.set_xticks(np.arange(0, nb_results, 1))
            ax.set_yticks(np.arange(0, nb_results, 1))
            ax.set_xticklabels(names, rotation=90)
            ax.set_yticklabels(names)
            plt.show()
            # plt.savefig("toto1.png", dpi=500,)


if __name__ == "__main__":
    main()
