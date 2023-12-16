import argparse
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
    p.add_argument('--results_tilted', nargs='+', default=[],
                   action='append', required=True,
                   help='List of characterization results.')
    p.add_argument('--bundles_names', nargs='+', default=[], action='append',
                   help='List of names for the characterized bundles.')

    p.add_argument('--whole_WM', default=[],
                   help='Path to the whole WM characterization.')
    p.add_argument('--whole_WM_tilted', default=[],
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

    results = []
    results_tilted = []
    extracted_bundles = []
    max_count = 0
    for i, result in enumerate(args.results[0]):
        print("Loading: ", result)
        results.append(np.load(result))
        extracted_bundles.append(str(Path(result).parent.name))
        curr_max_count = np.max(results[i]['Nb_voxels'])
        if curr_max_count > max_count:
            max_count = curr_max_count

    for i, result in enumerate(args.results_tilted[0]):
        print("Loading: ", result)
        results_tilted.append(np.load(result))
        curr_max_count = np.max(results_tilted[i]['Nb_voxels'])
        if curr_max_count > max_count:
            max_count = curr_max_count

    if args.whole_WM:
        print("Loading: ", args.whole_WM)
        whole_wm = np.load(args.whole_WM)
        whole_mid_bins = (whole_wm['Angle_min'] + whole_wm['Angle_max']) / 2.

    if args.whole_WM_tilted:
        print("Loading: ", args.whole_WM_tilted)
        whole_wm_tilted = np.load(args.whole_WM_tilted)

    if args.bundles_names != []:
        bundles_names = args.bundles_names[0]
    else:
        bundles_names = np.copy(extracted_bundles)
        bundles_names = list(bundles_names)

    if "MCP" in bundles_names:
        bundles_names.remove("MCP")
        bundles_names.append("MCP")

    bundles_names.append("WM")

    nb_bundles = len(bundles_names)
    nb_rows = int(np.ceil(nb_bundles / 2))

    mid_bins = (results[0]['Angle_min'] + results[0]['Angle_max']) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    # out_path = out_folder / str("all_bundles_original_1f_LABELS.png")
    # out_path = out_folder / str("all_bundles_original_1f.png")
    out_path = out_folder / str("all_bundles_original_1f_TOTO.png")
    plot_init(dims=(15, 10), font_size=10)
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 3
    plt.rcParams['axes.titlesize'] = 10
    fig, ax = plt.subplots(nb_rows, 8, layout='constrained')
    for i in range(nb_bundles):
        col = i % 2
        if col == 1:
            col = 4
        row = i // 2
        if bundles_names[i] in extracted_bundles:
            bundle_idx = extracted_bundles.index(bundles_names[i])
            result = results[bundle_idx]
            result_tilted = results_tilted[bundle_idx]
            is_measures = result['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            is_not_nan = result['Nb_voxels'] > 0
            norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
            colorbar = ax[row, col].scatter(mid_bins[is_measures],
                                            result['MTR'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col].scatter(mid_bins[is_not_measures],
                                 result['MTR'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_measures],
                                            result['MTsat'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 1].scatter(mid_bins[is_not_measures],
                                 result['MTsat'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 2].scatter(mid_bins[is_measures],
                                            result['ihMTR'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 2].scatter(mid_bins[is_not_measures],
                                 result['ihMTR'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 3].scatter(mid_bins[is_measures],
                                            result['ihMTsat'][is_measures],
                                            c=result['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 3].scatter(mid_bins[is_not_measures],
                                 result['ihMTsat'][is_not_measures],
                                 c=result['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            
            mean_mtr = np.sum(result['MTR'][is_not_nan] * result['Nb_voxels'][is_not_nan] / np.sum(result['Nb_voxels'][is_not_nan]))
            ax[row, col].axhline(y=mean_mtr, color="C0", linewidth=1, alpha=0.5, linestyle="--")
            mean_mtsat = np.sum(result['MTsat'][is_not_nan] * result['Nb_voxels'][is_not_nan] / np.sum(result['Nb_voxels'][is_not_nan]))
            ax[row, col + 1].axhline(y=mean_mtsat, color="C0", linewidth=1, alpha=0.5, linestyle="--")
            mean_ihmtr = np.sum(result['ihMTR'][is_not_nan] * result['Nb_voxels'][is_not_nan] / np.sum(result['Nb_voxels'][is_not_nan]))
            ax[row, col + 2].axhline(y=mean_ihmtr, color="C0", linewidth=1, alpha=0.5, linestyle="--")
            mean_ihmtsat = np.sum(result['ihMTsat'][is_not_nan] * result['Nb_voxels'][is_not_nan] / np.sum(result['Nb_voxels'][is_not_nan]))
            ax[row, col + 3].axhline(y=mean_ihmtsat, color="C0", linewidth=1, alpha=0.5, linestyle="--")

            # !!!!!!!!!!!!!!! Ajust the s values!!!!!!!!!!!!!!!!!!!
            weights = np.sqrt(result['Nb_voxels'][is_not_nan]) / np.max(result['Nb_voxels'][is_not_nan])
            # mtr_fit = splrep(mid_bins[is_not_nan], result['MTR'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C0")
            # mtsat_fit = splrep(mid_bins[is_not_nan], result['MTsat'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C0")
            # ihmtr_fit = splrep(mid_bins[is_not_nan], result['ihMTR'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C0")
            # ihmtsat_fit = splrep(mid_bins[is_not_nan], result['ihMTsat'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C0")

            ax[row, col].set_ylim(0.975 * np.min((np.nanmin(result['MTR']), np.nanmin(result_tilted['MTR']))),
                                  1.025 * np.max((np.nanmax(result['MTR']), np.nanmax(result_tilted['MTR']))))
            ax[row, col].set_yticks([np.round(np.min((np.nanmin(result['MTR']), np.nanmin(result_tilted['MTR']))), decimals=1),
                                     np.round(np.max((np.nanmax(result['MTR']), np.nanmax(result_tilted['MTR']))), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.min((np.nanmin(result['MTsat']), np.nanmin(result_tilted['MTsat']))),
                                      1.025 * np.max((np.nanmax(result['MTsat']), np.nanmax(result_tilted['MTsat']))))
            ax[row, col + 1].set_yticks([np.round(np.min((np.nanmin(result['MTsat']), np.nanmin(result_tilted['MTsat']))), decimals=1),
                                         np.round(np.max((np.nanmax(result['MTsat']), np.nanmax(result_tilted['MTsat']))), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)
            # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
            ax[row, col + 2].set_ylim(0.975 * np.min((np.nanmin(result['ihMTR']), np.nanmin(result_tilted['ihMTR']))),
                                      1.025 * np.max((np.nanmax(result['ihMTR']), np.nanmax(result_tilted['ihMTR']))))
            ax[row, col + 2].set_yticks([np.round(np.min((np.nanmin(result['ihMTR']), np.nanmin(result_tilted['ihMTR']))), decimals=1),
                                         np.round(np.max((np.nanmax(result['ihMTR']), np.nanmax(result_tilted['ihMTR']))), decimals=1)])
            ax[row, col + 2].set_xlim(0, 90)
            ax[row, col + 3].set_ylim(0.975 * np.min((np.nanmin(result['ihMTsat']), np.nanmin(result_tilted['ihMTsat']))),
                                      1.025 * np.max((np.nanmax(result['ihMTsat']), np.nanmax(result_tilted['ihMTsat']))))
            ax[row, col + 3].set_yticks([np.round(np.min((np.nanmin(result['ihMTsat']), np.nanmin(result_tilted['ihMTsat']))), decimals=1),
                                         np.round(np.max((np.nanmax(result['ihMTsat']), np.nanmax(result_tilted['ihMTsat']))), decimals=1)])
            ax[row, col + 3].set_xlim(0, 90)

            is_measures = result_tilted['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            is_not_nan = result_tilted['Nb_voxels'] > 0
            axt = ax[row, col]
            axt.scatter(mid_bins[is_measures],
                        result_tilted['MTR'][is_measures],
                        c=result_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt.scatter(mid_bins[is_not_measures],
                        result_tilted['MTR'][is_not_measures],
                        c=result_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Si je remets les fits, je dois produire un weights tilted!!!!!!!!!!!!!!!!!!
            # mtr_fit = splrep(mid_bins[is_not_nan], result_tilted['MTR'][is_not_nan], w=weights, s=0.0005)
            # axt.plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C1")

            # axt.set_yticks([np.round(np.nanmin(result_tilted['MTR']), decimals=1),
            #                 np.round(np.nanmax(result_tilted['MTR']), decimals=1)])
            
            axt2 = ax[row, col + 1]
            axt2.scatter(mid_bins[is_measures],
                        result_tilted['MTsat'][is_measures],
                        c=result_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt2.scatter(mid_bins[is_not_measures],
                        result_tilted['MTsat'][is_not_measures],
                        c=result_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # mtsat_fit = splrep(mid_bins[is_not_nan], result_tilted['MTsat'][is_not_nan], w=weights, s=0.0005)
            # axt2.plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C1")

            # axt2.set_yticks([np.round(np.nanmin(result_tilted['MTsat']), decimals=1),
            #                 np.round(np.nanmax(result_tilted['MTsat']), decimals=1)])
            
            axt3 = ax[row, col + 2]
            axt3.scatter(mid_bins[is_measures],
                        result_tilted['ihMTR'][is_measures],
                        c=result_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt3.scatter(mid_bins[is_not_measures],
                        result_tilted['ihMTR'][is_not_measures],
                        c=result_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # ihmtr_fit = splrep(mid_bins[is_not_nan], result_tilted['ihMTR'][is_not_nan], w=weights, s=0.0005)
            # axt3.plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C1")

            # axt3.set_yticks([np.round(np.nanmin(result_tilted['ihMTR']), decimals=1),
            #                 np.round(np.nanmax(result_tilted['ihMTR']), decimals=1)])
            
            axt4 = ax[row, col + 3]
            axt4.scatter(mid_bins[is_measures],
                        result_tilted['ihMTsat'][is_measures],
                        c=result_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt4.scatter(mid_bins[is_not_measures],
                        result_tilted['ihMTsat'][is_not_measures],
                        c=result_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # ihmtsat_fit = splrep(mid_bins[is_not_nan], result_tilted['ihMTsat'][is_not_nan], w=weights, s=0.0005)
            # axt4.plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C1")

            # axt4.set_yticks([np.round(np.nanmin(result_tilted['ihMTsat']), decimals=1),
            #                 np.round(np.nanmax(result_tilted['ihMTsat']), decimals=1)])

            # ax[row, col].set_zorder(1)
            # ax[row, col].patch.set_visible(False)

            # ax[row, col + 1].set_zorder(1)
            # ax[row, col + 1].patch.set_visible(False)

            mean_mtr = np.sum(result_tilted['MTR'][is_not_nan] * result_tilted['Nb_voxels'][is_not_nan] / np.sum(result_tilted['Nb_voxels'][is_not_nan]))
            axt.axhline(y=mean_mtr, color="C1", linewidth=1, alpha=0.5, linestyle="--")
            mean_mtsat = np.sum(result_tilted['MTsat'][is_not_nan] * result_tilted['Nb_voxels'][is_not_nan] / np.sum(result_tilted['Nb_voxels'][is_not_nan]))
            axt2.axhline(y=mean_mtsat, color="C1", linewidth=1, alpha=0.5, linestyle="--")
            mean_ihmtr = np.sum(result_tilted['ihMTR'][is_not_nan] * result_tilted['Nb_voxels'][is_not_nan] / np.sum(result_tilted['Nb_voxels'][is_not_nan]))
            axt3.axhline(y=mean_ihmtr, color="C1", linewidth=1, alpha=0.5, linestyle="--")
            mean_ihmtsat = np.sum(result_tilted['ihMTsat'][is_not_nan] * result_tilted['Nb_voxels'][is_not_nan] / np.sum(result_tilted['Nb_voxels'][is_not_nan]))
            axt4.axhline(y=mean_ihmtsat, color="C1", linewidth=1, alpha=0.5, linestyle="--")

            bundle_idx += 1
        else:
            is_measures = whole_wm['Nb_voxels'] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            ax[row, col].scatter(whole_mid_bins[is_measures],
                                whole_wm['MTR'][is_measures],
                                c=whole_wm['Nb_voxels'][is_measures],
                                cmap='Greys', norm=norm,
                                edgecolors="C0", linewidths=1)
            ax[row, col].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_measures],
                                            whole_wm['MTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 1].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['MTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTR'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 2].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTR'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_measures],
                                            whole_wm['ihMTsat'][is_measures],
                                            c=whole_wm['Nb_voxels'][is_measures],
                                            cmap='Greys', norm=norm,
                                            edgecolors="C0", linewidths=1)
            ax[row, col + 3].scatter(whole_mid_bins[is_not_measures],
                                 whole_wm['ihMTsat'][is_not_measures],
                                 c=whole_wm['Nb_voxels'][is_not_measures],
                                 cmap='Greys', norm=norm, alpha=0.5,
                                 edgecolors="C0", linewidths=1)

            is_not_nan = whole_wm['Nb_voxels'] > 0
            weights = np.sqrt(whole_wm['Nb_voxels'][is_not_nan]) / np.max(whole_wm['Nb_voxels'][is_not_nan])
            # mtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTR'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C0")
            # mtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['MTsat'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 1].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C0")
            # ihmtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTR'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 2].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C0")
            # ihmtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm['ihMTsat'][is_not_nan], w=weights, s=0.0005)
            # ax[row, col + 3].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C0")

            ax[row, col].set_ylim(0.975 * np.min((np.nanmin(whole_wm['MTR']), np.nanmin(whole_wm_tilted['MTR']))),
                                  1.025 * np.max((np.nanmax(whole_wm['MTR']), np.nanmax(whole_wm_tilted['MTR']))))
            ax[row, col].set_yticks([np.round(np.min((np.nanmin(whole_wm['MTR']), np.nanmin(whole_wm_tilted['MTR']))), decimals=1),
                                     np.round(np.max((np.nanmax(whole_wm['MTR']), np.nanmax(whole_wm_tilted['MTR']))), decimals=1)])
            ax[row, col].set_xlim(0, 90)
            # ax[row, col].tick_params(axis='y', labelcolor="C0")
            ax[row, col + 1].set_ylim(0.975 * np.min((np.nanmin(whole_wm['MTsat']), np.nanmin(whole_wm_tilted['MTsat']))),
                                      1.025 * np.max((np.nanmax(whole_wm['MTsat']), np.nanmax(whole_wm_tilted['MTsat']))))
            ax[row, col + 1].set_yticks([np.round(np.min((np.nanmin(whole_wm['MTsat']), np.nanmin(whole_wm_tilted['MTsat']))), decimals=1),
                                        np.round(np.max((np.nanmax(whole_wm['MTsat']), np.nanmax(whole_wm_tilted['MTsat']))), decimals=1)])
            ax[row, col + 1].set_xlim(0, 90)
            # ax[row, col + 1].tick_params(axis='y', labelcolor="C2")
            ax[row, col + 2].set_ylim(0.975 * np.min((np.nanmin(whole_wm['ihMTR']), np.nanmin(whole_wm_tilted['ihMTR']))),
                                      1.025 * np.max((np.nanmax(whole_wm['ihMTR']), np.nanmax(whole_wm_tilted['ihMTR']))))
            ax[row, col + 2].set_yticks([np.round(np.min((np.nanmin(whole_wm['ihMTR']), np.nanmin(whole_wm_tilted['ihMTR']))), decimals=1),
                                        np.round(np.max((np.nanmax(whole_wm['ihMTR']), np.nanmax(whole_wm_tilted['ihMTR']))), decimals=1)])
            ax[row, col + 2].set_xlim(0, 90)
            ax[row, col + 3].set_ylim(0.975 * np.min((np.nanmin(whole_wm['ihMTsat']), np.nanmin(whole_wm_tilted['ihMTsat']))),
                                      1.025 * np.max((np.nanmax(whole_wm['ihMTsat']), np.nanmax(whole_wm_tilted['ihMTsat']))))
            ax[row, col + 3].set_yticks([np.round(np.min((np.nanmin(whole_wm['ihMTsat']), np.nanmin(whole_wm_tilted['ihMTsat']))), decimals=1),
                                        np.round(np.max((np.nanmax(whole_wm['ihMTsat']), np.nanmax(whole_wm_tilted['ihMTsat']))), decimals=1)])
            ax[row, col + 3].set_xlim(0, 90)

            axt = ax[row, col]
            axt.scatter(whole_mid_bins[is_measures],
                        whole_wm_tilted['MTR'][is_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt.scatter(whole_mid_bins[is_not_measures],
                        whole_wm_tilted['MTR'][is_not_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # mtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm_tilted['MTR'][is_not_nan], w=weights, s=0.0005)
            # axt.plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C1")

            # axt.set_yticks([np.round(np.nanmin(whole_wm_tilted['MTR']), decimals=1),
            #                 np.round(np.nanmax(whole_wm_tilted['MTR']), decimals=1)])
            
            axt2 = ax[row, col + 1]
            axt2.scatter(whole_mid_bins[is_measures],
                        whole_wm_tilted['MTsat'][is_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt2.scatter(whole_mid_bins[is_not_measures],
                        whole_wm_tilted['MTsat'][is_not_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # mtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm_tilted['MTsat'][is_not_nan], w=weights, s=0.0005)
            # axt2.plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C1")

            # axt2.set_yticks([np.round(np.nanmin(whole_wm_tilted['MTsat']), decimals=1),
            #                 np.round(np.nanmax(whole_wm_tilted['MTsat']), decimals=1)])
            
            axt3 = ax[row, col + 2]
            axt3.scatter(whole_mid_bins[is_measures],
                        whole_wm_tilted['ihMTR'][is_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt3.scatter(whole_mid_bins[is_not_measures],
                        whole_wm_tilted['ihMTR'][is_not_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # ihmtr_fit = splrep(whole_mid_bins[is_not_nan], whole_wm_tilted['ihMTR'][is_not_nan], w=weights, s=0.0005)
            # axt3.plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C1")

            # axt3.set_yticks([np.round(np.nanmin(whole_wm_tilted['ihMTR']), decimals=1),
            #                 np.round(np.nanmax(whole_wm_tilted['ihMTR']), decimals=1)])
            
            axt4 = ax[row, col + 3]
            axt4.scatter(whole_mid_bins[is_measures],
                        whole_wm_tilted['ihMTsat'][is_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_measures],
                        cmap='Greys', norm=norm,
                        edgecolors="C1", linewidths=1)
            axt4.scatter(whole_mid_bins[is_not_measures],
                        whole_wm_tilted['ihMTsat'][is_not_measures],
                        c=whole_wm_tilted['Nb_voxels'][is_not_measures],
                        cmap='Greys', norm=norm, alpha=0.5,
                        edgecolors="C1", linewidths=1)
            # ihmtsat_fit = splrep(whole_mid_bins[is_not_nan], whole_wm_tilted['ihMTsat'][is_not_nan], w=weights, s=0.0005)
            # axt4.plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C1")

            # axt4.set_yticks([np.round(np.nanmin(whole_wm_tilted['ihMTsat']), decimals=1),
            #                 np.round(np.nanmax(whole_wm_tilted['ihMTsat']), decimals=1)])

            # ax[row, col].set_zorder(1)
            # ax[row, col].patch.set_visible(False)

            # ax[row, col + 1].set_zorder(1)
            # ax[row, col + 1].patch.set_visible(False)
        if col == 0:
            ax[row, col + 3].legend(handles=[colorbar], labels=[bundles_names[i]],
                                loc='center left', bbox_to_anchor=(1.0, 0.5),
                                markerscale=0, handletextpad=-2.0, handlelength=2)
        if col == 4:
            ax[row, col].legend(handles=[colorbar], labels=[bundles_names[i]],
                                loc='center left', bbox_to_anchor=(-0.8, 0.5),
                                markerscale=0, handletextpad=-2.0, handlelength=2)
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
    ax[nb_rows - 1, 4].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 4].set_xlim(0, 90)
    ax[nb_rows - 1, 4].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 5].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 5].set_xlim(0, 90)
    ax[nb_rows - 1, 5].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 6].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 6].set_xlim(0, 90)
    ax[nb_rows - 1, 6].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[nb_rows - 1, 7].set_xlabel(r'$\theta_a$')
    ax[nb_rows - 1, 7].set_xlim(0, 90)
    ax[nb_rows - 1, 7].set_xticks([0, 15, 30, 45, 60, 75, 90])
    # if nb_bundles % 2 != 0:
    #     ax[nb_rows - 1, 1].set_yticks([])
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    # plt.show()
    plt.savefig(out_path, dpi=500, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
