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
    
    p.add_argument('--in_results', nargs='+',
                   help='List of all results directories.')
    p.add_argument('--sub_id', type=int, default=26,
                   help='ID of the selected subject.')
    p.add_argument('--ses_id', type=int, default=3,
                   help='ID of the selected session.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = args.in_results

    file_name = "/new_characterization/1f_original_results.npz"
    ses_id = args.ses_id - 1
    min_nb_voxels = args.min_nb_voxels

    # extract subjects
    subjects = []
    for i, result in enumerate(results):
        subjects.append(result.split("_")[0])
    subjects = np.unique(np.asarray(subjects))

    dummy = results[0]
    dims = np.load(dummy + file_name)["MTR"].shape
    mtr_sub = np.zeros(((len(subjects),) + dims))
    mtsat_sub = np.zeros(((len(subjects),) + dims))
    ihmtr_sub = np.zeros(((len(subjects),) + dims))
    ihmtsat_sub = np.zeros(((len(subjects),) + dims))
    nb_voxels_sub = np.zeros(((len(subjects),) + dims))
    mtr_ses = np.zeros(((5,) + dims))
    mtsat_ses = np.zeros(((5,) + dims))
    ihmtr_ses = np.zeros(((5,) + dims))
    ihmtsat_ses = np.zeros(((5,) + dims))
    nb_voxels_ses = np.zeros(((5,) + dims))
    labels_sub = np.zeros((len(subjects)), dtype=object)

    for i, subject in enumerate(subjects):
        print(subject)
        sessions = list(Path('.').glob(subject + "*"))
        subject_id = int(subject.split('-')[1])
        for j, session in enumerate(sessions[:5]):
            print(session)
            tmp_result = np.load(str(session) + file_name)
            mtr_sub[i] += tmp_result['MTR']
            mtsat_sub[i] += tmp_result['MTsat']
            ihmtr_sub[i] += tmp_result['ihMTR']
            ihmtsat_sub[i] += tmp_result['ihMTsat']
            nb_voxels_sub[i] += tmp_result['Nb_voxels']
            if args.sub_id == subject_id:
                mtr_ses[j] = tmp_result['MTR']
                mtsat_ses[j] = tmp_result['MTsat']
                ihmtr_ses[j] = tmp_result['ihMTR']
                ihmtsat_ses[j] = tmp_result['ihMTsat']
                nb_voxels_ses[j] = tmp_result['Nb_voxels']
        mtr_sub[i] /= 5
        mtsat_sub[i] /= 5
        ihmtr_sub[i] /= 5
        ihmtsat_sub[i] /= 5
        nb_voxels_sub[i] /= 5
        labels_sub[i] = str(i + 1)

    mid_bins = (tmp_result['Angle_min'] + tmp_result['Angle_max']) / 2.
    
    max_count_sub = np.max(nb_voxels_sub)
    max_count_ses = np.max(nb_voxels_ses)
    max_count = np.max(nb_voxels_ses[ses_id])
    norm_sub = mpl.colors.Normalize(vmin=0, vmax=max_count_sub)
    norm_ses = mpl.colors.Normalize(vmin=0, vmax=max_count_ses)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    highres_bins = np.arange(0, 90 + 1, 0.5)

    plot_init(dims=(8, 10), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    fig, ax = plt.subplots(6, 2,
                           gridspec_kw={"width_ratios":[1, 1]},
                           layout='constrained')

    labels = None

    min_mtr = 10000
    max_mtr = 0
    min_mtsat = 10000
    max_mtsat = 0
    min_ihmtr = 10000
    max_ihmtr = 0
    min_ihmtsat = 10000
    max_ihmtsat = 0
    for i in range(mtr_sub.shape[0] - 1):
        is_measures = nb_voxels_sub[i + 1] >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        min_mtr = np.min((np.nanmin(mtr_sub[i + 1, is_measures]), min_mtr))
        max_mtr = np.max((np.nanmax(mtr_sub[i + 1, is_measures]), max_mtr))
        min_mtsat = np.min((np.nanmin(mtsat_sub[i + 1, is_measures]), min_mtsat))
        max_mtsat = np.max((np.nanmax(mtsat_sub[i + 1, is_measures]), max_mtsat))
        min_ihmtr = np.min((np.nanmin(ihmtr_sub[i + 1, is_measures]), min_ihmtr))
        max_ihmtr = np.max((np.nanmax(ihmtr_sub[i + 1, is_measures]), max_ihmtr))
        min_ihmtsat = np.min((np.nanmin(ihmtsat_sub[i + 1, is_measures]), min_ihmtsat))
        max_ihmtsat = np.max((np.nanmax(ihmtsat_sub[i + 1, is_measures]), max_ihmtsat))
        if labels is not None:
            colorbar = ax[0, 0].scatter(mid_bins[is_measures], mtr_sub[i + 1, is_measures],
                                    c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                                    label=labels[i + 1], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
            ax[0, 1].scatter(mid_bins[is_measures], ihmtr_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            label=labels[i + 1], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[1, 0].scatter(mid_bins[is_measures], mtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            label=labels[i + 1], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[1, 1].scatter(mid_bins[is_measures], ihmtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            label=labels[i + 1], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
        else:
            colorbar = ax[0, 0].scatter(mid_bins[is_measures], mtr_sub[i + 1, is_measures],
                                    c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                                    linewidths=1,
                                    edgecolors=cm.naviaS(i), marker="o")
            ax[0, 1].scatter(mid_bins[is_measures], ihmtr_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors=cm.naviaS(i), marker="o")
            ax[1, 0].scatter(mid_bins[is_measures], mtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors=cm.naviaS(i), marker="o")
            ax[1, 1].scatter(mid_bins[is_measures], ihmtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors=cm.naviaS(i), marker="o")
            ax[0, 0].scatter(mid_bins[is_not_measures], mtr_sub[i + 1, is_not_measures],
                                    c=nb_voxels_sub[i + 1, is_not_measures], cmap='Greys', norm=norm_sub,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i), marker="o")
            ax[0, 1].scatter(mid_bins[is_not_measures], ihmtr_sub[i + 1, is_not_measures],
                            c=nb_voxels_sub[i + 1, is_not_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i), marker="o")
            ax[1, 0].scatter(mid_bins[is_not_measures], mtsat_sub[i + 1, is_not_measures],
                            c=nb_voxels_sub[i + 1, is_not_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i), marker="o")
            ax[1, 1].scatter(mid_bins[is_not_measures], ihmtsat_sub[i + 1, is_not_measures],
                            c=nb_voxels_sub[i + 1, is_not_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i), marker="o")

    ax[0, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[1, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[0, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[1, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[0, 0].set_yticks([np.round(min_mtr, decimals=1), np.round(np.mean((min_mtr, max_mtr)), decimals=1), np.round(max_mtr, decimals=1)])
    ax[1, 0].set_yticks([np.round(min_mtsat, decimals=1), np.round(np.mean((min_mtsat, max_mtsat)), decimals=1), np.round(max_mtsat, decimals=1)])
    ax[0, 1].set_yticks([np.round(min_ihmtr, decimals=1), np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1), np.round(max_ihmtr, decimals=1)])
    ax[1, 1].set_yticks([np.round(min_ihmtsat, decimals=1), np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1), np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[:2, 1], location='right', label="Voxel count")

    labels = np.array(["Session 1", "Session 2", "Session 3", "Session 4", "Session 5"])

    min_mtr = 10000
    max_mtr = 0
    min_mtsat = 10000
    max_mtsat = 0
    min_ihmtr = 10000
    max_ihmtr = 0
    min_ihmtsat = 10000
    max_ihmtsat = 0
    for i in range(mtr_ses.shape[0]):
        is_measures = nb_voxels_ses[i] >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        min_mtr = np.min((np.nanmin(mtr_ses[i, is_measures]), min_mtr))
        max_mtr = np.max((np.nanmax(mtr_ses[i, is_measures]), max_mtr))
        min_mtsat = np.min((np.nanmin(mtsat_ses[i, is_measures]), min_mtsat))
        max_mtsat = np.max((np.nanmax(mtsat_ses[i, is_measures]), max_mtsat))
        min_ihmtr = np.min((np.nanmin(ihmtr_ses[i, is_measures]), min_ihmtr))
        max_ihmtr = np.max((np.nanmax(ihmtr_ses[i, is_measures]), max_ihmtr))
        min_ihmtsat = np.min((np.nanmin(ihmtsat_ses[i, is_measures]), min_ihmtsat))
        max_ihmtsat = np.max((np.nanmax(ihmtsat_ses[i, is_measures]), max_ihmtsat))
        if labels is not None:
            colorbar = ax[2, 0].scatter(mid_bins[is_measures], mtr_ses[i, is_measures],
                                    c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                                    label=labels[i], linewidths=1,
                                    edgecolors=cm.naviaS(i+10), marker="o")
            ax[2, 1].scatter(mid_bins[is_measures], ihmtr_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors=cm.naviaS(i+10), marker="o")
            ax[3, 0].scatter(mid_bins[is_measures], mtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors=cm.naviaS(i+10), marker="o")
            ax[3, 1].scatter(mid_bins[is_measures], ihmtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors=cm.naviaS(i+10), marker="o")
            ax[2, 0].scatter(mid_bins[is_not_measures], mtr_ses[i, is_not_measures],
                                    c=nb_voxels_ses[i, is_not_measures], cmap='Greys', norm=norm_ses,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+10), marker="o")
            ax[2, 1].scatter(mid_bins[is_not_measures], ihmtr_ses[i, is_not_measures],
                            c=nb_voxels_ses[i, is_not_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i+10), marker="o")
            ax[3, 0].scatter(mid_bins[is_not_measures], mtsat_ses[i, is_not_measures],
                            c=nb_voxels_ses[i, is_not_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i+10), marker="o")
            ax[3, 1].scatter(mid_bins[is_not_measures], ihmtsat_ses[i, is_not_measures],
                            c=nb_voxels_ses[i, is_not_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1, alpha=0.5,
                            edgecolors=cm.naviaS(i+10), marker="o")
        else:
            colorbar = ax[2, 0].scatter(mid_bins[is_measures], mtr_ses[i, is_measures],
                                    c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                                    linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
            ax[2, 1].scatter(mid_bins[is_measures], ihmtr_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[3, 0].scatter(mid_bins[is_measures], mtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[3, 1].scatter(mid_bins[is_measures], ihmtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")

    ax[2, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[3, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[2, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[3, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[2, 0].set_yticks([np.round(min_mtr, decimals=1), np.round(np.mean((min_mtr, max_mtr)), decimals=1), np.round(max_mtr, decimals=1)])
    ax[3, 0].set_yticks([np.round(min_mtsat, decimals=1), np.round(np.mean((min_mtsat, max_mtsat)), decimals=1), np.round(max_mtsat, decimals=1)])
    ax[2, 1].set_yticks([np.round(min_ihmtr, decimals=1), np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1), np.round(max_ihmtr, decimals=1)])
    ax[3, 1].set_yticks([np.round(min_ihmtsat, decimals=1), np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1), np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[2:4, 1], location='right', label="Voxel count")

    if labels is not None:
        ax[2, 1].legend(loc=1, prop={'size': 8})

    labels = None
    is_measures = nb_voxels_ses[ses_id] >= min_nb_voxels
    is_not_measures = np.invert(is_measures)

    is_not_nan = nb_voxels_ses[ses_id] > 0

    weights = np.sqrt(nb_voxels_ses[ses_id]) / np.max(nb_voxels_ses[ses_id])

    mtr_fit = splrep(mid_bins[is_not_nan], mtr_ses[ses_id, is_not_nan], w=weights, s=0.0005)
    mtsat_fit = splrep(mid_bins[is_not_nan], mtsat_ses[ses_id, is_not_nan], w=weights, s=0.00005)
    ihmtr_fit = splrep(mid_bins[is_not_nan], ihmtr_ses[ses_id, is_not_nan], w=weights, s=0.0005)
    ihmtsat_fit = splrep(mid_bins[is_not_nan], ihmtsat_ses[ses_id, is_not_nan], w=weights, s=0.000005)

    if labels is not None:
        colorbar = ax[4, 0].scatter(mid_bins[is_measures], mtr_ses[ses_id, is_measures],
                                c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                                label=labels[ses_id], linewidths=1,
                                edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[4, 1].scatter(mid_bins[is_measures], ihmtr_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 0].scatter(mid_bins[is_measures], mtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 1].scatter(mid_bins[is_measures], ihmtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
    else:
        colorbar = ax[4, 0].scatter(mid_bins[is_measures], mtr_ses[ses_id, is_measures],
                                c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                                linewidths=1,
                                edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[4, 0].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color=cm.naviaS(ses_id+10))
        ax[4, 1].scatter(mid_bins[is_measures], ihmtr_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[4, 1].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color=cm.naviaS(ses_id+10))
        ax[5, 0].scatter(mid_bins[is_measures], mtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 0].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color=cm.naviaS(ses_id+10))
        ax[5, 1].scatter(mid_bins[is_measures], ihmtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 1].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color=cm.naviaS(ses_id+10))
        ax[4, 0].scatter(mid_bins[is_not_measures], mtr_ses[ses_id, is_not_measures],
                                c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                                linewidths=1, alpha=0.5,
                                edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[4, 1].scatter(mid_bins[is_not_measures], ihmtr_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 0].scatter(mid_bins[is_not_measures], mtsat_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")
        ax[5, 1].scatter(mid_bins[is_not_measures], ihmtsat_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors=cm.naviaS(ses_id+10), marker="o")

    min_mtr = np.nanmin(mtr_ses[ses_id, is_measures])
    max_mtr = np.nanmax(mtr_ses[ses_id, is_measures])
    min_mtsat = np.nanmin(mtsat_ses[ses_id, is_measures])
    max_mtsat = np.nanmax(mtsat_ses[ses_id, is_measures])
    min_ihmtr = np.nanmin(ihmtr_ses[ses_id, is_measures])
    max_ihmtr = np.nanmax(ihmtr_ses[ses_id, is_measures])
    min_ihmtsat = np.nanmin(ihmtsat_ses[ses_id, is_measures])
    max_ihmtsat = np.nanmax(ihmtsat_ses[ses_id, is_measures])

    ax[4, 0].set_ylim(0.975 * min_mtr, 1.025 * max_mtr)
    ax[5, 0].set_ylim(0.975 * min_mtsat, 1.025 * max_mtsat)
    ax[4, 1].set_ylim(0.975 * min_ihmtr, 1.025 * max_ihmtr)
    ax[5, 1].set_ylim(0.975 * min_ihmtsat, 1.025 * max_ihmtsat)

    ax[4, 0].set_yticks([np.round(min_mtr, decimals=1), np.round(np.mean((min_mtr, max_mtr)), decimals=1), np.round(max_mtr, decimals=1)])
    ax[5, 0].set_yticks([np.round(min_mtsat, decimals=1), np.round(np.mean((min_mtsat, max_mtsat)), decimals=1), np.round(max_mtsat, decimals=1)])
    ax[4, 1].set_yticks([np.round(min_ihmtr, decimals=1), np.round(np.mean((min_ihmtr, max_ihmtr)), decimals=1), np.round(max_ihmtr, decimals=1)])
    ax[5, 1].set_yticks([np.round(min_ihmtsat, decimals=1), np.round(np.mean((min_ihmtsat, max_ihmtsat)), decimals=1), np.round(max_ihmtsat, decimals=1)])

    fig.colorbar(colorbar, ax=ax[4:6, 1], location='right', label="Voxel count")

    ax[0, 0].text(0.02, 0.88, "a) Inter-subject", transform=ax[0, 0].transAxes, 
                  size=10)#, weight='bold')
    ax[2, 0].text(0.02, 0.88, "b) Intra-subject", transform=ax[2, 0].transAxes, 
                  size=10)#, weight='bold')
    ax[4, 0].text(0.02, 0.88, "c) Single-subject", transform=ax[4, 0].transAxes, 
                  size=10)#, weight='bold')

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
    # fig.tight_layout()
    # plt.show()
    plt.savefig(args.out_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
