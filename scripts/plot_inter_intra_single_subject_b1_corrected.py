import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import splrep, BSpline

from modules.io import plot_init


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_name',
                   help='Path of the output name.')
    
    p.add_argument('--inter_results', nargs='+',
                   help='List of all inter-subject results directories.')
    p.add_argument('--intra_results', nargs='+',
                   help='List of all intra-subject results directories.')
    p.add_argument('--single_results', nargs='+',
                   help='List of the single-subject results directories.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    min_nb_voxels = args.min_nb_voxels

    # Inter-subject results
    inter_subs = []
    inter_subs_names = []
    max_count_inter = 0
    for result in args.inter_results:
        inter_subs.append(np.load(result))
        inter_subs_names.append(Path(result).parent.name)
        if np.max(np.load(result)['Nb_voxels']) < 2000:
            max_count_inter = np.max([max_count_inter, np.max(np.load(result)['Nb_voxels'])])

    # Intra-subject results
    intra_subs = []
    intra_subs_names = []
    max_count_intra = 0
    for result in args.intra_results:
        intra_subs.append(np.load(result))
        intra_subs_names.append(Path(result).parent.name)
        max_count_intra = np.max([max_count_intra, np.max(np.load(result)['Nb_voxels'])])

    # Single-subject results
    single_subs = []
    single_subs_names = []
    max_count_single = 0
    for result in args.single_results:
        single_subs.append(np.load(result))
        single_subs_names.append(Path(result).parent.name)
        max_count_single = np.max([max_count_single, np.max(np.load(result)['Nb_voxels'])])

    mid_bins = (single_subs[0]['Angle_min'] + single_subs[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    norm_inter = mpl.colors.Normalize(vmin=0, vmax=max_count_inter)
    norm_intra = mpl.colors.Normalize(vmin=0, vmax=max_count_intra)
    norm_single = mpl.colors.Normalize(vmin=0, vmax=max_count_single)

    plot_init(dims=(8, 10), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    fig, ax = plt.subplots(6, 2,
                           gridspec_kw={"width_ratios":[1, 1]},
                           layout='constrained')

    # ------------------------------Inter-subject------------------------------
    min_mtr, min_mtsat, min_ihmtr, min_ihmtsat = (10000, 10000, 10000, 10000)
    max_mtr, max_mtsat, max_ihmtr, max_ihmtsat = (0, 0, 0, 0)

    for i, inter_sub in enumerate(inter_subs):
        nb_voxels = inter_sub['Nb_voxels']
        is_measures = nb_voxels >= min_nb_voxels
        is_not_measures = np.invert(is_measures)

        min_mtr = np.min((np.nanmin(inter_sub['MTR'][is_measures]), min_mtr))
        max_mtr = np.max((np.nanmax(inter_sub['MTR'][is_measures]), max_mtr))
        min_ihmtr = np.min((np.nanmin(inter_sub['ihMTR'][is_measures]), min_ihmtr))
        max_ihmtr = np.max((np.nanmax(inter_sub['ihMTR'][is_measures]), max_ihmtr))
        min_mtsat = np.min((np.nanmin(inter_sub['MTsat'][is_measures]), min_mtsat))
        max_mtsat = np.max((np.nanmax(inter_sub['MTsat'][is_measures]), max_mtsat))
        min_ihmtsat = np.min((np.nanmin(inter_sub['ihMTsat'][is_measures]), min_ihmtsat))
        max_ihmtsat = np.max((np.nanmax(inter_sub['ihMTsat'][is_measures]), max_ihmtsat))

        # With is_measures
        colorbar = ax[0, 0].scatter(mid_bins[is_measures],
                                    inter_sub['MTR'][is_measures],
                                    c=nb_voxels[is_measures],
                                    cmap='Greys', norm=norm_inter,
                                    label=inter_subs_names[i], linewidths=1,
                                    edgecolors=cm.naviaS(i + 10), marker="o")
        ax[0, 1].scatter(mid_bins[is_measures],
                         inter_sub['ihMTR'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_inter,
                         label=inter_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[1, 0].scatter(mid_bins[is_measures],
                         inter_sub['MTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_inter,
                         label=inter_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[1, 1].scatter(mid_bins[is_measures],
                         inter_sub['ihMTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_inter,
                         label=inter_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        # With is_not_measures
        ax[0, 0].scatter(mid_bins[is_not_measures],
                         inter_sub['MTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_inter,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[0, 1].scatter(mid_bins[is_not_measures],
                         inter_sub['ihMTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_inter,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[1, 0].scatter(mid_bins[is_not_measures],
                         inter_sub['MTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_inter,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[1, 1].scatter(mid_bins[is_not_measures],
                         inter_sub['ihMTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_inter,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")

    ax[0, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[1, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[0, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[1, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[0, 0].set_yticks([np.round(min_mtr, decimals=1),
                         np.round(np.mean((min_mtr, max_mtr)), decimals=1),
                         np.round(max_mtr, decimals=1)])
    ax[1, 0].set_yticks([np.round(min_mtsat, decimals=1),
                         np.round(np.mean((min_mtsat, max_mtsat)), decimals=1),
                         np.round(max_mtsat, decimals=1)])
    ax[0, 1].set_yticks([np.round(min_ihmtr, decimals=1),
                         np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1),
                         np.round(max_ihmtr, decimals=1)])
    ax[1, 1].set_yticks([np.round(min_ihmtsat, decimals=1),
                         np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1),
                         np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[:2, 1], location='right', label="Voxel count")

    # ------------------------------Intra-subject------------------------------
    min_mtr, min_mtsat, min_ihmtr, min_ihmtsat = (10000, 10000, 10000, 10000)
    max_mtr, max_mtsat, max_ihmtr, max_ihmtsat = (0, 0, 0, 0)

    # labels = np.array(["Session 1", "Session 2", "Session 3", "Session 4", "Session 5"])

    for i, intra_sub in enumerate(intra_subs):
        nb_voxels = intra_sub['Nb_voxels']
        is_measures = nb_voxels >= min_nb_voxels
        is_not_measures = np.invert(is_measures)

        min_mtr = np.min((np.nanmin(intra_sub['MTR'][is_measures]), min_mtr))
        max_mtr = np.max((np.nanmax(intra_sub['MTR'][is_measures]), max_mtr))
        min_ihmtr = np.min((np.nanmin(intra_sub['ihMTR'][is_measures]), min_ihmtr))
        max_ihmtr = np.max((np.nanmax(intra_sub['ihMTR'][is_measures]), max_ihmtr))
        min_mtsat = np.min((np.nanmin(intra_sub['MTsat'][is_measures]), min_mtsat))
        max_mtsat = np.max((np.nanmax(intra_sub['MTsat'][is_measures]), max_mtsat))
        min_ihmtsat = np.min((np.nanmin(intra_sub['ihMTsat'][is_measures]), min_ihmtsat))
        max_ihmtsat = np.max((np.nanmax(intra_sub['ihMTsat'][is_measures]), max_ihmtsat))

        # With is_measures
        colorbar = ax[2, 0].scatter(mid_bins[is_measures],
                                    intra_sub['MTR'][is_measures],
                                    c=nb_voxels[is_measures],
                                    cmap='Greys', norm=norm_intra,
                                    label=intra_subs_names[i], linewidths=1,
                                    edgecolors=cm.naviaS(i + 10), marker="o")
        ax[2, 1].scatter(mid_bins[is_measures],
                         intra_sub['ihMTR'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_intra,
                         label=intra_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[3, 0].scatter(mid_bins[is_measures],
                         intra_sub['MTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_intra,
                         label=intra_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[3, 1].scatter(mid_bins[is_measures],
                         intra_sub['ihMTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_intra,
                         label=intra_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        # With is_not_measures
        ax[2, 0].scatter(mid_bins[is_not_measures],
                         intra_sub['MTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_intra,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[2, 1].scatter(mid_bins[is_not_measures],
                         intra_sub['ihMTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_intra,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[3, 0].scatter(mid_bins[is_not_measures],
                         intra_sub['MTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_intra,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")
        ax[3, 1].scatter(mid_bins[is_not_measures],
                         intra_sub['ihMTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_intra,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 10), marker="o")

    ax[2, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[3, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[2, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[3, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[2, 0].set_yticks([np.round(min_mtr, decimals=1),
                         np.round(np.mean((min_mtr, max_mtr)), decimals=1),
                         np.round(max_mtr, decimals=1)])
    ax[3, 0].set_yticks([np.round(min_mtsat, decimals=1),
                         np.round(np.mean((min_mtsat, max_mtsat)), decimals=1),
                         np.round(max_mtsat, decimals=1)])
    ax[2, 1].set_yticks([np.round(min_ihmtr, decimals=1),
                         np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1),
                         np.round(max_ihmtr, decimals=1)])
    ax[3, 1].set_yticks([np.round(min_ihmtsat, decimals=1),
                         np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1),
                         np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[2:4, 1], location='right', label="Voxel count")

    # if labels is not None:
    #     ax[2, 1].legend(loc=1, prop={'size': 8})

    # ------------------------------Single-subject-----------------------------
    min_mtr, min_mtsat, min_ihmtr, min_ihmtsat = (10000, 10000, 10000, 10000)
    max_mtr, max_mtsat, max_ihmtr, max_ihmtsat = (0, 0, 0, 0)

    # labels = np.array(["Session 1", "Session 2", "Session 3", "Session 4", "Session 5"])

    for i, single_sub in enumerate(single_subs):
        nb_voxels = single_sub['Nb_voxels']
        is_measures = nb_voxels >= min_nb_voxels
        is_not_measures = np.invert(is_measures)

        min_mtr = np.min((np.nanmin(single_sub['MTR'][is_measures]), min_mtr))
        max_mtr = np.max((np.nanmax(single_sub['MTR'][is_measures]), max_mtr))
        min_ihmtr = np.min((np.nanmin(single_sub['ihMTR'][is_measures]), min_ihmtr))
        max_ihmtr = np.max((np.nanmax(single_sub['ihMTR'][is_measures]), max_ihmtr))
        min_mtsat = np.min((np.nanmin(single_sub['MTsat'][is_measures]), min_mtsat))
        max_mtsat = np.max((np.nanmax(single_sub['MTsat'][is_measures]), max_mtsat))
        min_ihmtsat = np.min((np.nanmin(single_sub['ihMTsat'][is_measures]), min_ihmtsat))
        max_ihmtsat = np.max((np.nanmax(single_sub['ihMTsat'][is_measures]), max_ihmtsat))

        is_not_nan = nb_voxels > 0
        weights = np.sqrt(nb_voxels) / np.max(nb_voxels)
        mtr_fit = splrep(mid_bins[is_not_nan], single_sub['MTR'][is_not_nan], w=weights, s=0.0005)
        mtsat_fit = splrep(mid_bins[is_not_nan], single_sub['MTsat'][is_not_nan], w=weights, s=0.00005)
        ihmtr_fit = splrep(mid_bins[is_not_nan], single_sub['ihMTR'][is_not_nan], w=weights, s=0.0005)
        ihmtsat_fit = splrep(mid_bins[is_not_nan], single_sub['ihMTsat'][is_not_nan], w=weights, s=0.000005)

        # With is_measures
        colorbar = ax[4, 0].scatter(mid_bins[is_measures],
                                    single_sub['MTR'][is_measures],
                                    c=nb_voxels[is_measures],
                                    cmap='Greys', norm=norm_single,
                                    label=single_subs_names[i], linewidths=1,
                                    edgecolors=cm.naviaS(i + 12), marker="o")
        ax[4, 0].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--",
                      color=cm.naviaS(i + 12))
        ax[4, 1].scatter(mid_bins[is_measures],
                         single_sub['ihMTR'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_single,
                         label=single_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[4, 1].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--",
                      color=cm.naviaS(i + 12))
        ax[5, 0].scatter(mid_bins[is_measures],
                         single_sub['MTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_single,
                         label=single_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[5, 0].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--",
                      color=cm.naviaS(i + 12))
        ax[5, 1].scatter(mid_bins[is_measures],
                         single_sub['ihMTsat'][is_measures],
                         c=nb_voxels[is_measures],
                         cmap='Greys', norm=norm_single,
                         label=single_subs_names[i], linewidths=1,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[5, 1].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--",
                      color=cm.naviaS(i + 12))
        # With is_not_measures
        ax[4, 0].scatter(mid_bins[is_not_measures],
                         single_sub['MTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_single,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[4, 1].scatter(mid_bins[is_not_measures],
                         single_sub['ihMTR'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_single,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[5, 0].scatter(mid_bins[is_not_measures],
                         single_sub['MTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_single,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 12), marker="o")
        ax[5, 1].scatter(mid_bins[is_not_measures],
                         single_sub['ihMTsat'][is_not_measures],
                         c=nb_voxels[is_not_measures], cmap='Greys',
                         norm=norm_single,
                         linewidths=1, alpha=0.5,
                         edgecolors=cm.naviaS(i + 12), marker="o")

    ax[4, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[5, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[4, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[5, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[4, 0].set_yticks([np.round(min_mtr, decimals=1),
                         np.round(np.mean((min_mtr, max_mtr)), decimals=1),
                         np.round(max_mtr, decimals=1)])
    ax[5, 0].set_yticks([np.round(min_mtsat, decimals=1),
                         np.round(np.mean((min_mtsat, max_mtsat)), decimals=1),
                         np.round(max_mtsat, decimals=1)])
    ax[4, 1].set_yticks([np.round(min_ihmtr, decimals=1),
                         np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1),
                         np.round(max_ihmtr, decimals=1)])
    ax[5, 1].set_yticks([np.round(min_ihmtsat, decimals=1),
                         np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1),
                         np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[4:6, 1], location='right', label="Voxel count")

    ax[0, 0].text(0.025, 0.88, "a) Between-subject", transform=ax[0, 0].transAxes, 
                  size=10, weight='bold')
    ax[2, 0].text(0.025, 0.88, "b) Within-subject", transform=ax[2, 0].transAxes, 
                  size=10, weight='bold')
    ax[4, 0].text(0.025, 0.88, "c) Single-timepoint", transform=ax[4, 0].transAxes, 
                  size=10, weight='bold')
    
    ax[0, 0].text(0.675, 0.025, "CVb=1.81%", transform=ax[0, 0].transAxes, size=10)
    ax[0, 1].text(0.675, 0.88, "CVb=5.68%", transform=ax[0, 1].transAxes, size=10)
    ax[1, 0].text(0.675, 0.025, "CVb=4.91%", transform=ax[1, 0].transAxes, size=10)
    ax[1, 1].text(0.675, 0.88, "CVb=6.44%", transform=ax[1, 1].transAxes, size=10)

    ax[2, 0].text(0.67, 0.025, "CVw=0.93%", transform=ax[2, 0].transAxes, size=10)
    ax[2, 1].text(0.67, 0.88, "CVw=4.18%", transform=ax[2, 1].transAxes, size=10)
    ax[3, 0].text(0.67, 0.025, "CVw=1.81%", transform=ax[3, 0].transAxes, size=10)
    ax[3, 1].text(0.67, 0.88, "CVw=4.10%", transform=ax[3, 1].transAxes, size=10)

    ax[5, 0].set_xlabel(r'$\theta_a$')
    ax[5, 1].set_xlabel(r'$\theta_a$')
    ax[0, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[0, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[2, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[2, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[3, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[3, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[4, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[4, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[5, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[5, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    plt.setp(ax, xlim=(0, 90))
    ax[0, 0].set_ylabel("MTR")
    ax[0, 1].set_ylabel("ihMTR")
    ax[1, 0].set_ylabel("MTsat")
    ax[1, 1].set_ylabel("ihMTsat")
    ax[2, 0].set_ylabel("MTR")
    ax[2, 1].set_ylabel("ihMTR")
    ax[3, 0].set_ylabel("MTsat")
    ax[3, 1].set_ylabel("ihMTsat")
    ax[4, 0].set_ylabel("MTR")
    ax[4, 1].set_ylabel("ihMTR")
    ax[5, 0].set_ylabel("MTsat")
    ax[5, 1].set_ylabel("ihMTsat")

    fig.get_layout_engine().set(h_pad=0, hspace=0.05, wspace=0.05)

    line = plt.Line2D([0.024,0.94],[0.672, 0.672], transform=fig.transFigure, color="black", linestyle=(0, (5, 5)))
    fig.add_artist(line)
    line = plt.Line2D([0.024,0.94],[0.346, 0.346], transform=fig.transFigure, color="black", linestyle=(0, (5, 5)))
    fig.add_artist(line)

    # fig.tight_layout()
    # plt.show()
    plt.savefig(args.out_name, dpi=500)
    # plt.close()


if __name__ == "__main__":
    main()
