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
    
    p.add_argument('--whole_wm', default=[],
                   help='Path to the whole WM characterization.')

    p.add_argument('--variation', action='store_true')

    p.add_argument('--is_bundles', action='store_true')

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

    if args.whole_wm:
        print("Loading: ", args.whole_wm)
        whole_wm = np.load(args.whole_wm)
        results.append(whole_wm)
        names.append("WM")

    nb_bins = len(results[0]['Angle_min'])
    nb_results = len(results)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    plot_init(font_size=8, dims=(10, 10))

    for measure in args.measures:
        print(measure)
        to_analyse = np.zeros((nb_results, nb_bins))
        for i, result in enumerate(results):
            to_analyse[i] = result[measure]
            to_analyse[i, result['Nb_voxels'] < min_nb_voxels] = np.nan

        if args.variation:
            coeff_var = scipy.stats.variation(to_analyse, axis=0,
                                              nan_policy='omit')
            print(np.nanmean(coeff_var))

        if args.is_bundles:
            dataset = pd.DataFrame(data=to_analyse.T)
            corr = dataset.corr()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(np.square(corr), cmap=cm.navia_r, norm=norm)
            fig.colorbar(cax)
            ax.set_xticks(np.arange(0, nb_results, 1))
            ax.set_yticks(np.arange(0, nb_results, 1))
            ax.set_xticklabels(names, rotation=90)
            ax.set_yticklabels(names)
            plt.show()
            # plt.savefig("toto1.png", dpi=500,)

            variation_matrix = np.zeros((nb_results, nb_results))
            pair_array = np.zeros((2, nb_bins))
            bundles_idx = np.flip(np.arange(1, nb_results, 1))
            for b_idx in bundles_idx:
                for next_b_idx in range(b_idx):
                    pair_array[0] = to_analyse[b_idx]
                    pair_array[1] = to_analyse[next_b_idx]
                    variation_matrix[b_idx, next_b_idx] = np.nanmean(scipy.stats.variation(pair_array, axis=0, nan_policy='omit'))
                    variation_matrix[next_b_idx, b_idx] = variation_matrix[b_idx, next_b_idx]

            norm = mpl.colors.Normalize(vmin=0, vmax=np.max(variation_matrix) * 100)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(variation_matrix * 100, cmap=cm.navia, norm=norm)
            fig.colorbar(cax)
            ax.set_xticks(np.arange(0, nb_results, 1))
            ax.set_yticks(np.arange(0, nb_results, 1))
            ax.set_xticklabels(names, rotation=90)
            ax.set_yticklabels(names)
            plt.show()
            # plt.savefig("toto1.png", dpi=500,)



if __name__ == "__main__":
    main()
