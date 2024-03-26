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
    
    p.add_argument('--names', nargs='+',
                   help='List of names.')

    p.add_argument('--mean', action='store_true') # Must be done one measure at the time.

    p.add_argument('--split', action='store_true')

    p.add_argument('--fused', action='store_true')

    p.add_argument('--suffix', default='', type=str)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    stats = []
    sub_names = []
    measure_names = []
    for i, stat in enumerate(args.in_stats):
        print("Loading: ", stat)
        stats.append(np.loadtxt(stat))
        sub_names.append(str(Path(stat).parent.parent))
        measure_names.append(str(Path(stat).name).split('_')[0])

    measure_name = str(Path(args.in_stats[0]).name).split('_')[0]
    stat_name = str(Path(args.in_stats[0]).name).split('_')[-1].split('.')[0]
    if stat_name == 'correlation':
        cmap = cm.navia_r
        cmap_label = "Correlation coefficient"
    elif stat_name == 'variation':
        cmap = cm.navia
        cmap_label = "Variation coefficient"

    mean_stats = np.nanmean(np.asarray(stats), axis=0)
    norm = mpl.colors.Normalize(vmin=np.min([np.min(mean_stats), np.min(stats)]),
                                vmax=np.max([np.max(mean_stats), np.max(stats)]))

    if args.mean:
        plot_init(font_size=10, dims=(10, 10))
        norm = mpl.colors.Normalize(vmin=np.min(mean_stats), vmax=np.max(mean_stats))
        nb_rows = mean_stats.shape[0]
        fig = plt.figure()
        fig, ax = plt.subplots(1, 1, layout='constrained')
        cax = ax.matshow(mean_stats, cmap=cmap, norm=norm)
        fig.colorbar(cax, location='right', label=cmap_label,  fraction=0.05, pad=0.04)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.set_xticks(np.arange(0, nb_rows, 1))
        ax.set_yticks(np.arange(0, nb_rows, 1))
        if args.names:
            ax.set_xticklabels(args.names, rotation=90)
            ax.set_yticklabels(args.names)
        # plt.show()
        out_path = out_dir / ('{}_{}_mean_{}.png').format(measure_name, args.suffix, stat_name)
        plt.savefig(out_path, dpi=500)

        out_path = out_dir / ('{}_{}_mean_{}.txt').format(measure_name, args.suffix, stat_name)
        np.savetxt(out_path, mean_stats)

    if args.split:
        plot_init(font_size=10, dims=(10, 10))
        for i, stat in enumerate(stats):
            nb_rows = stat.shape[0]
            fig, ax = plt.subplots(1, 1, layout='constrained')
            cax = ax.matshow(stat, cmap=cmap, norm=norm)
            fig.colorbar(cax, location='right', label=cmap_label, fraction=0.05, pad=0.04)
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            ax.set_xticks(np.arange(0, nb_rows, 1))
            ax.set_yticks(np.arange(0, nb_rows, 1))
            if args.names:
                ax.set_xticklabels(args.names, rotation=90)
                ax.set_yticklabels(args.names)
            # plt.show()
            out_path = out_dir / ('{}_{}_{}_{}.png').format(measure_name, args.suffix,
                                                            sub_names[i], stat_name)
            plt.savefig(out_path, dpi=500)


    if args.fused:
        plot_init(font_size=10, dims=(10, 4))
        fig, ax = plt.subplots(1, len(stats), layout='constrained')
        for i, stat in enumerate(stats):
            nb_rows = stat.shape[0]
            cax = ax[i].matshow(stat, cmap=cmap, norm=norm)
            ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            ax[i].set_xticks(np.arange(0, nb_rows, 1))
            ax[i].set_yticks(np.arange(0, nb_rows, 1))
            ax[i].set_title(measure_names[i])
            if args.names:
                ax[i].set_xticklabels(args.names, rotation=90)
                if i == 0:
                    ax[i].set_yticklabels(args.names)
                else:
                    ax[i].set_yticklabels('')
        fig.colorbar(cax, ax=ax[-1], location='right', label=cmap_label, fraction=0.05, pad=0.04)
        # plt.show()
        out_path = out_dir / ('all_measures_fused_{}_{}.png').format(args.suffix, stat_name)
        plt.savefig(out_path, dpi=500)


if __name__ == "__main__":
    main()
