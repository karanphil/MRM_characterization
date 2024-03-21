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
    p.add_argument('out_dir',
                   help='Path of the output directory.')
    
    p.add_argument('--in_stats', nargs='+', required=True,
                   help='List of all stats files.')
    
    p.add_argument('--stat_name', required=True)

    p.add_argument('--measures_name', required=True)

    p.add_argument('--mean', action='store_true')

    p.add_argument('--multi', action='store_true')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    results = []
    for i, result in enumerate(args.in_results):
        print("Loading: ", result)
        results.append(np.load(result))

    nb_results = len(results)

    plot_init(font_size=8, dims=(10, 10))

    for measure in args.measures:
        print(measure)
        to_analyse = np.zeros((nb_results, nb_bins))
        for i, result in enumerate(results):
            to_analyse[i] = result[measure]
            to_analyse[i, result['Nb_voxels'] < min_nb_voxels] = np.nan

        if args.is_bundles:
            dataset = pd.DataFrame(data=to_analyse.T)
            corr = dataset.corr()
            out_path = out_dir / (measure + '_' + args.suffix + '_correlation.txt')
            np.savetxt(out_path, corr)
            # norm = mpl.colors.Normalize(vmin=np.min(corr), vmax=1)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # cax = ax.matshow(corr, cmap=cm.navia_r, norm=norm)
            # fig.colorbar(cax)
            # ax.set_xticks(np.arange(0, nb_results, 1))
            # ax.set_yticks(np.arange(0, nb_results, 1))
            # ax.set_xticklabels(names, rotation=90)
            # ax.set_yticklabels(names)
            # plt.show()
            # # plt.savefig("toto1.png", dpi=500,)

            variation_matrix = np.zeros((nb_results, nb_results))
            pair_array = np.zeros((2, nb_bins))
            bundles_idx = np.flip(np.arange(1, nb_results, 1))
            for b_idx in bundles_idx:
                for next_b_idx in range(b_idx):
                    pair_array[0] = to_analyse[b_idx]
                    pair_array[1] = to_analyse[next_b_idx]
                    variation_matrix[b_idx, next_b_idx] = np.nanmean(scipy.stats.variation(pair_array, axis=0))
                    variation_matrix[next_b_idx, b_idx] = variation_matrix[b_idx, next_b_idx]

            out_path = out_dir / (measure + '_' + args.suffix + '_variation.txt')
            np.savetxt(out_path, variation_matrix * 100)

            # norm = mpl.colors.Normalize(vmin=0, vmax=np.ceil(np.max(variation_matrix) * 100))
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # cax = ax.matshow(variation_matrix * 100, cmap=cm.navia, norm=norm)
            # fig.colorbar(cax)
            # ax.set_xticks(np.arange(0, nb_results, 1))
            # ax.set_yticks(np.arange(0, nb_results, 1))
            # ax.set_xticklabels(names, rotation=90)
            # ax.set_yticklabels(names)
            # plt.show()
            # # plt.savefig("toto1.png", dpi=500,)


if __name__ == "__main__":
    main()
