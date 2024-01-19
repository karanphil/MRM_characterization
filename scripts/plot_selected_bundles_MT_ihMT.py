import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import splrep, BSpline

from modules.io import plot_init

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--results', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   required=True,
                   help='List of names for the characterized bundles.')

    p.add_argument('--whole_WM', default=[],
                   help='Path to the whole WM characterization.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_results = len(args.results[0])

    out_folder = Path(args.out_folder)
    min_nb_voxels = args.min_nb_voxels

    bundles_names = args.bundles_names[0]

    results = []
    extracted_bundles = []
    max_count = 0
    tmp = 0
    for i, result in enumerate(args.results[0]):
        if str(Path(result).parent) in bundles_names:
            print("Loading: ", result)
            results.append(np.load(result))
            extracted_bundles.append(str(Path(result).parent))
            curr_max_count = np.max(results[tmp]['Nb_voxels'])
            if curr_max_count > max_count:
                max_count = curr_max_count
            tmp += 1

    if args.whole_WM:
        print("Loading: ", args.whole_WM)
        whole_wm = np.load(args.whole_WM)
        whole_mid_bins = (whole_wm['Angle_min'] + whole_wm['Angle_max']) / 2.

    # if "MCP" in bundles_names:
    #     bundles_names.remove("MCP")
    #     bundles_names.append("MCP")

    bundles_names.append("WM")

    nb_bundles = len(bundles_names)
    # nb_rows = int(np.ceil(nb_bundles / 2))
    nb_rows = nb_bundles

    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    s_mtr = np.ones(15) * 0.0005
    s_mtr[5] = 0.001

    s_mtsat = np.ones(15) * 0.0005
    s_mtsat[1] = 0.00001
    s_mtsat[5] = 0.0001

    s_ihmtr = np.ones(15) * 0.0005
    s_ihmtr[1] = 0.0001
    s_ihmtr[5] = 0.001

    s_ihmtsat = np.ones(15) * 0.0005
    s_ihmtsat[1] = 0.000005
    s_ihmtsat[2] = 0.00001
    s_ihmtsat[7] = 0.000005

    # out_path = out_folder / str("all_bundles_original_1f_LABELS.png")
    # out_path = out_folder / str("all_bundles_original_1f.png")
    out_path1 = out_folder / str("few_bundles_original_1f.png")
    plot_init(dims=(10, 8), font_size=10)
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.titlesize'] = 12
    fig, ax = plt.subplots(nb_rows, 4, layout='constrained')
    for i in range(nb_bundles):
        col = 0
        row = i
        if bundles_names[i] in extracted_bundles:
            bundle_idx = extracted_bundles.index(bundles_names[i])
            result = results[bundle_idx]
            is_measures = result['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            colorbar = ax[row, col].scatter(mid_bins[is_measures],
                                            result['MTR'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col].scatter(mid_bins[is_not_measures],
                                 result['MTR'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_measures],
                                            result['MTsat'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_not_measures],
                                 result['MTsat'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 2].scatter(mid_bins[is_measures],
                                            result['ihMTR'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 2].scatter(mid_bins[is_not_measures],
                                 result['ihMTR'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 3].scatter(mid_bins[is_measures],
                                            result['ihMTsat'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(5), linewidths=1)
            ax[row, col + 3].scatter(mid_bins[is_not_measures],
                                 result['ihMTsat'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(5), linewidths=1)

            is_not_nan = result['Nb_voxels'] > 0
            weights = np.sqrt(result['Nb_voxels'][is_not_nan]) / np.max(result['Nb_voxels'][is_not_nan])
            mtr_fit = splrep(mid_bins[is_not_nan], result['MTR'][is_not_nan], w=weights, s=s_mtr[i])
            ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(2))
            mtsat_fit = splrep(mid_bins[is_not_nan], result['MTsat'][is_not_nan], w=weights, s=s_mtsat[i])
            ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(3))
            ihmtr_fit = splrep(mid_bins[is_not_nan], result['ihMTR'][is_not_nan], w=weights, s=s_ihmtr[i])
            ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(4))
            ihmtsat_fit = splrep(mid_bins[is_not_nan], result['ihMTsat'][is_not_nan], w=weights, s=s_ihmtsat[i])
            ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(5))

            ax[row, col].set_ylim(0.975 * np.nanmin(result['MTR']),
                                  1.025 * np.nanmax(result['MTR']))
            ax[row, col].set_yticks([np.round(np.nanmin(result['MTR']), decimals=1),
                                     np.round(np.nanmax(result['MTR']), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.nanmin(result['MTsat']),
                                      1.025 * np.nanmax(result['MTsat']))
            ax[row, col + 1].set_yticks([np.round(np.nanmin(result['MTsat']), decimals=1),
                                         np.round(np.nanmax(result['MTsat']), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)
            # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
            ax[row, col + 2].set_ylim(0.975 * np.nanmin(result['ihMTR']),
                                      1.025 * np.nanmax(result['ihMTR']))
            ax[row, col + 2].set_yticks([np.round(np.nanmin(result['ihMTR']), decimals=1),
                                         np.round(np.nanmax(result['ihMTR']), decimals=1)])
            ax[row, col + 2].set_xlim(0, 90)
            ax[row, col + 3].set_ylim(0.975 * np.nanmin(result['ihMTsat']),
                                      1.025 * np.nanmax(result['ihMTsat']))
            ax[row, col + 3].set_yticks([np.round(np.nanmin(result['ihMTsat']), decimals=1),
                                         np.round(np.nanmax(result['ihMTsat']), decimals=1)])
            ax[row, col + 3].set_xlim(0, 90)

            bundle_idx += 1
        else:
            is_measures = whole_wm['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            ax[row, col].scatter(whole_mid_bins[is_measures],
                                whole_wm['MTR'][is_measures],
                                c=whole_wm['Nb_voxels'][is_measures],
                                cmap='Greys', norm=norm,
                                edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(2), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_measures],
                                            whole_wm['MTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(3), linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTR'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(4), linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors=cm.naviaS(5), linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors=cm.naviaS(5), linewidths=1)

            is_not_nan = whole_wm['Nb_voxels'] > 0
            weights = np.sqrt(whole_wm['Nb_voxels'][is_not_nan]) / np.max(whole_wm['Nb_voxels'][is_not_nan])
            mtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTR'][is_not_nan], w=weights, s=0.00001)
            ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(2))
            mtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTsat'][is_not_nan], w=weights, s=0.000001)
            ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(3))
            ihmtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTR'][is_not_nan], w=weights, s=0.00005)
            ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(4))
            ihmtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTsat'][is_not_nan], w=weights, s=0.0000005)
            ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(5))

            ax[row, col].set_ylim(0.975 * np.nanmin(whole_wm['MTR']),
                                  1.025 * np.nanmax(whole_wm['MTR']))
            ax[row, col].set_yticks([np.round(np.nanmin(whole_wm['MTR']), decimals=1),
                                     np.round(np.nanmax(whole_wm['MTR']), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.nanmin(whole_wm['MTsat']),
                                      1.025 * np.nanmax(whole_wm['MTsat']))
            ax[row, col + 1].set_yticks([np.round(np.nanmin(whole_wm['MTsat']), decimals=1),
                                         np.round(np.nanmax(whole_wm['MTsat']), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)
            # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
            ax[row, col + 2].set_ylim(0.975 * np.nanmin(whole_wm['ihMTR']),
                                      1.025 * np.nanmax(whole_wm['ihMTR']))
            ax[row, col + 2].set_yticks([np.round(np.nanmin(whole_wm['ihMTR']), decimals=1),
                                         np.round(np.nanmax(whole_wm['ihMTR']), decimals=1)])
            ax[row, col + 2].set_xlim(0, 90)
            ax[row, col + 3].set_ylim(0.975 * np.nanmin(whole_wm['ihMTsat']),
                                      1.025 * np.nanmax(whole_wm['ihMTsat']))
            ax[row, col + 3].set_yticks([np.round(np.nanmin(whole_wm['ihMTsat']), decimals=1),
                                         np.round(np.nanmax(whole_wm['ihMTsat']), decimals=1)])
            ax[row, col + 3].set_xlim(0, 90)

        # if col == 0:
        #     ax[row, col + 1].legend(handles=[colorbar], labels=[bundles_names[i]],
        #                         loc='center left', bbox_to_anchor=(1.0, 0.5),
        #                         markerscale=0, handletextpad=-2.0, handlelength=2)
        # if col == 0:
        #     ax[row, col].legend(handles=[colorbar], labels=[bundles_names[i]],
        #                         loc='center left', bbox_to_anchor=(-0.6, 0.5),
        #                         markerscale=0, handletextpad=-2.0, handlelength=2)
        if row != nb_rows - 1:
            ax[row, col].get_xaxis().set_ticks([])
            ax[row, col + 1].get_xaxis().set_ticks([])
            ax[row, col + 2].get_xaxis().set_ticks([])
            ax[row, col + 3].get_xaxis().set_ticks([])

        if row == 0:
            ax[row, col].title.set_text('MTR')
            ax[row, col + 1].title.set_text('MTsat')
            ax[row, col + 2].title.set_text('ihMTR')
            ax[row, col + 3].title.set_text('ihMTsat')

        ax[row, 0].set_ylabel(bundles_names[i], labelpad=10)
        # if row == (nb_rows - 1) / 2 and col == 0:
        #     ax[row, col].set_ylabel('MTR', color="C0")
        #     ax[row, col + 1].set_ylabel('ihMTR', color="C2")
        #     ax[row, col].yaxis.set_label_coords(-0.2, 0.5)
        # if row == (nb_rows - 1) / 2 and col == 0:
        #     axt.set_ylabel('MTsat', color="C1")
        #     axt2.set_ylabel('ihMTsat', color="C4")
    fig.colorbar(colorbar, ax=ax[:, -1], location='right',
                 label="Voxel count", aspect=100)
    ax[nb_rows - 1, 0].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 0].set_xlim(0, 90)
    ax[nb_rows - 1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 1].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 1].set_xlim(0, 90)
    ax[nb_rows - 1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 2].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 2].set_xlim(0, 90)
    ax[nb_rows - 1, 2].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 3].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 3].set_xlim(0, 90)
    ax[nb_rows - 1, 3].set_xticks([0, 15, 30, 45, 60, 75, 90])
    # if nb_bundles % 2 != 0:
    #     ax[nb_rows - 1, 1].set_yticks([])
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    # plt.show()
    plt.savefig(out_path1, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
