import argparse
from cmcrameri import cm
from matplotlib.ticker import FormatStrFormatter
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
    
    # p.add_argument('--names', nargs='+',
    #                help='List of names.')

    p.add_argument('--mean', action='store_true') # Must be done one measure at the time.

    p.add_argument('--split', action='store_true')

    p.add_argument('--fused', action='store_true')

    p.add_argument('--single_line', action='store_true')

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

    mean_stats = np.nanmean(np.asarray(stats), axis=0)

    with open(args.in_stats[0]) as f:
        names = f.readline().strip('\n').strip('#').split(' ')[1:-1]
    if "" in names:
        names.remove("")
    if '' in names:
        names.remove('')
    measure_name = str(Path(args.in_stats[0]).name).split('_')[0]
    stat_name = str(Path(args.in_stats[0]).name).split('_')[-1].split('.')[0]
    if stat_name == 'correlation':
        # cmap = cm.navia_r
        # cmap = cm.roma # or cm.vik_r for diverging gradients (correlation)
        cmap = cm.vik_r
        # cmap = cm.bam
        # cmap_label = "Pearson correlation coefficient"
        # cmap_label = "Within-subject mean PCC"
        # cmap_label = "Between-subject mean PCC"
        cmap_label = "PCC"
        norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    elif stat_name == 'variation':
        cmap = cm.navia
        cmap_label = "Coefficient of variation (%)"
        norm = mpl.colors.Normalize(vmin=0,
                                    vmax=np.nanmax([np.nanmax(mean_stats), np.nanmax(stats)]))

    # cmap = cm.roma # or cm.vik_r for diverging gradients (correlation)
    # norm = mpl.colors.Normalize(vmin=np.nanmin([np.nanmin(mean_stats), np.nanmin(stats)]),
    #                             vmax=np.nanmax([np.nanmax(mean_stats), np.nanmax(stats)]))

    if args.mean:
        # plot_init(font_size=10, dims=(10, 10))
        # norm = mpl.colors.Normalize(vmin=np.nanmin(mean_stats), vmax=np.nanmax(mean_stats))
        # nb_rows = mean_stats.shape[0]
        # fig = plt.figure()
        # fig, ax = plt.subplots(1, 1, layout='constrained')
        # cax = ax.matshow(mean_stats, cmap=cmap, norm=norm)
        # fig.colorbar(cax, location='right', label=cmap_label,  fraction=0.05, pad=0.04)
        # ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        # ax.set_xticks(np.arange(0, nb_rows, 1))
        # ax.set_yticks(np.arange(0, nb_rows, 1))
        # ax.set_xticklabels(names, rotation=90)
        # ax.set_yticklabels(names)
        # # plt.show()
        # out_path = out_dir / ('{}_{}_mean_{}.png').format(measure_name, args.suffix, stat_name)
        # plt.savefig(out_path, dpi=500)
        with open(args.in_stats[0]) as f:
            header = f.readline().strip('\n').strip('#')

        out_path = out_dir / ('{}_{}_mean_{}.txt').format(measure_name, args.suffix, stat_name)
        np.savetxt(out_path, mean_stats, header=header)

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
            ax.set_xticklabels(names, rotation=90)
            ax.set_yticklabels(names)
            # plt.show()
            out_path = out_dir / ('{}_{}_{}_{}.png').format(measure_name, args.suffix,
                                                            sub_names[i], stat_name)
            plt.savefig(out_path, dpi=500)

    if args.fused:
        nb_rows = stats[0].shape[0]
        font_size = 6 if nb_rows > 15 else 10
        width = 16 if nb_rows > 15 else 10
        plot_init(font_size=10, dims=(width, 4))
        fig, ax = plt.subplots(1, len(stats), layout='constrained')
        for i, stat in enumerate(stats):
            cax = ax[i].matshow(stat, cmap=cmap, norm=norm)
            ax[i].tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            ax[i].set_xticks(np.arange(0, nb_rows, 1))
            ax[i].set_yticks(np.arange(0, nb_rows, 1))
            ax[i].set_title(measure_names[i], fontsize=12)
            ax[i].set_xticklabels(names, rotation=90, fontsize=font_size)
            if i == 0:
                ax[i].set_yticklabels(names, fontsize=font_size)
            else:
                ax[i].set_yticklabels('')
        if stat_name == "correlation":
            pad = 0.04
        elif stat_name == "variation" and "all" in args.suffix:
            pad = 0.035
        elif stat_name == "variation" and "few" in args.suffix:
            pad = 0.06
        clb = fig.colorbar(cax, ax=ax[-1], location='right',
                     fraction=0.05, pad=pad, format=FormatStrFormatter('%.1f'))
        clb.ax.set_ylabel(cmap_label, fontsize=12) # remove if variation
        # plt.show()
        out_path = out_dir / ('all_measures_fused_{}_{}.png').format(args.suffix, stat_name)
        plt.savefig(out_path, dpi=500)

    if args.single_line:
        for i, stat in enumerate(stats):
            stat = stat.reshape((1, stat.shape[0]))
            nb_rows = stat.shape[1]
            font_size = 6 if nb_rows > 15 else 10
            width = 16 if nb_rows > 15 else 10
            plot_init(font_size=10, dims=(10, 2))
            fig, ax = plt.subplots(1, 1, layout='constrained')
            cax = ax.matshow(stat, cmap=cmap, norm=norm)
            # fig.colorbar(cax, location='right', label=cmap_label, fraction=0.05, pad=0.04)
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
            ax.set_xticks(np.arange(0, nb_rows, 1))
            ax.set_yticks([])
            ax.set_xticklabels(names, rotation=90)
            #ax.set_yticklabels("MT")
            # ax.set_ylabel("WM", rotation=90)
            if measure_name == "MT":
                ax.set_title("MTR vs MTsat")
            if measure_name == "ihMT":
                ax.set_title("ihMTR vs ihMTsat")
            # plt.show()
            out_path = out_dir / ('{}_{}_{}.png').format(measure_name, args.suffix, stat_name)
            plt.savefig(out_path, dpi=500)


if __name__ == "__main__":
    main()
