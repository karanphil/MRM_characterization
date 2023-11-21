import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.interpolate import splrep, BSpline

from modules.io import plot_init

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_name',
                   help='Path of the output name.')
    
    p.add_argument('--in_results', nargs='+',
                   help='List of all results.')
    p.add_argument('--sub_id', type=int,
                   help='ID of the selected subject.')
    p.add_argument('--ses_id', type=int,
                   help='ID of the selected session.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    results = args.in_results

    file_name = "/results_txt/WB_single_fiber_mt{0}_ihmt{0}_results_0.5_fa_thr_1_bin_width.txt"
    ses_id = args.ses_id - 1
    min_nb_voxels = args.min_nb_voxels

    # extract subjects
    subjects = []
    for i, result in enumerate(results):
        subjects.append(result.split("_")[0])
    subjects = np.unique(np.asarray(subjects))

    dummy = results[0]
    dims = np.loadtxt(dummy + file_name.format("r"), skiprows=1).shape
    ratio_sub = np.zeros(((len(subjects),) + dims))
    satur_sub = np.zeros(((len(subjects),) + dims))
    ratio_ses = np.zeros(((5,) + dims))
    satur_ses = np.zeros(((5,) + dims))
    labels_sub = np.zeros((len(subjects)), dtype=object)

    for i, subject in enumerate(subjects):
        print(subject)
        sessions = list(Path('.').glob(subject + "*"))
        for j, session in enumerate(sessions[:5]):
            print(session)
            ratio_sub[i] += np.loadtxt(str(session) + file_name.format("r"), skiprows=1)
            satur_sub[i] += np.loadtxt(str(session) + file_name.format("sat"), skiprows=1)
            if i + 3 == args.sub_id:
                ratio_ses[j] = np.loadtxt(str(session) + file_name.format("r"), skiprows=1)
                satur_ses[j] = np.loadtxt(str(session) + file_name.format("sat"), skiprows=1)
        ratio_sub[i] /= 5
        satur_sub[i] /= 5
        labels_sub[i] = str(i + 1)

    bins = np.zeros((ratio_sub[:, :, 0].shape[1] + 1))
    bins[:ratio_sub[:, :, 0].shape[1]] = ratio_sub[0, :, 0]
    bins[-1] = ratio_sub[0, -1, 1]

    mtr_sub = ratio_sub[:, :, 2]
    ihmtr_sub = ratio_sub[:, :, 3]
    nb_voxels_sub = ratio_sub[:, :, 4]
    mtsat_sub = satur_sub[:, :, 2]
    ihmtsat_sub = satur_sub[:, :, 3]

    mtr_ses = ratio_ses[:, :, 2]
    ihmtr_ses = ratio_ses[:, :, 3]
    nb_voxels_ses = ratio_ses[:, :, 4]
    mtsat_ses = satur_ses[:, :, 2]
    ihmtsat_ses = satur_ses[:, :, 3]
    
    max_count_sub = np.max(nb_voxels_sub)
    max_count_ses = np.max(nb_voxels_ses)
    max_count = np.max(nb_voxels_ses[ses_id])
    norm_sub = mpl.colors.Normalize(vmin=0, vmax=max_count_sub)
    norm_ses = mpl.colors.Normalize(vmin=0, vmax=max_count_ses)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    mid_bins = (bins[:-1] + bins[1:]) / 2.
    highres_bins = np.arange(0, 90 + 1, 0.5)

    plot_init(dims=(8, 10), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    fig, ax = plt.subplots(6, 2,
                           gridspec_kw={"width_ratios":[1, 1]},
                           layout='constrained')

    labels = None

    for i in range(mtr_sub.shape[0] - 1):
        is_measures = nb_voxels_sub[i + 1] >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
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
                                    edgecolors="C" + str(i), marker="o")
            ax[0, 1].scatter(mid_bins[is_measures], ihmtr_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[1, 0].scatter(mid_bins[is_measures], mtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[1, 1].scatter(mid_bins[is_measures], ihmtsat_sub[i + 1, is_measures],
                            c=nb_voxels_sub[i + 1, is_measures], cmap='Greys', norm=norm_sub,
                            linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            
    fig.colorbar(colorbar, ax=ax[:2, 1], location='right', label="Voxel count")

    labels = np.array(["Session 1", "Session 2", "Session 3", "Session 4", "Session 5"])

    for i in range(mtr_ses.shape[0]):
        is_measures = nb_voxels_ses[i] >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        if labels is not None:
            colorbar = ax[2, 0].scatter(mid_bins[is_measures], mtr_ses[i, is_measures],
                                    c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                                    label=labels[i], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
            ax[2, 1].scatter(mid_bins[is_measures], ihmtr_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[3, 0].scatter(mid_bins[is_measures], mtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
            ax[3, 1].scatter(mid_bins[is_measures], ihmtsat_ses[i, is_measures],
                            c=nb_voxels_ses[i, is_measures], cmap='Greys', norm=norm_ses,
                            label=labels[i], linewidths=1,
                            edgecolors="C" + str(i), marker="o")
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
            
    fig.colorbar(colorbar, ax=ax[2:4, 1], location='right', label="Voxel count")
    if labels is not None:
        ax[2, 1].legend(loc=1, prop={'size': 8})

    labels = None
    is_measures = nb_voxels_ses[ses_id] >= min_nb_voxels
    is_not_measures = np.invert(is_measures)

    is_not_nan = nb_voxels_ses[ses_id] > 0

    weights = np.sqrt(nb_voxels_ses[ses_id])

    mtr_fit = splrep(mid_bins[is_not_nan], mtr_ses[ses_id, is_not_nan], s=0.5)
    mtsat_fit = splrep(mid_bins[is_not_nan], mtsat_ses[ses_id, is_not_nan], s=0.5)
    ihmtr_fit = splrep(mid_bins[is_not_nan], ihmtr_ses[ses_id, is_not_nan], s=0.5)
    ihmtsat_fit = splrep(mid_bins[is_not_nan], ihmtsat_ses[ses_id, is_not_nan], s=0.005)

    if labels is not None:
        colorbar = ax[4, 0].scatter(mid_bins[is_measures], mtr_ses[ses_id, is_measures],
                                c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                                label=labels[ses_id], linewidths=1,
                                edgecolors="C0", marker="o")
        ax[4, 1].scatter(mid_bins[is_measures], ihmtr_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors="C0", marker="o")
        ax[5, 0].scatter(mid_bins[is_measures], mtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors="C0", marker="o")
        ax[5, 1].scatter(mid_bins[is_measures], ihmtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        label=labels[ses_id], linewidths=1,
                        edgecolors="C0", marker="o")
    else:
        colorbar = ax[4, 0].scatter(mid_bins[is_measures], mtr_ses[ses_id, is_measures],
                                c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                                linewidths=1,
                                edgecolors="C0", marker="o")
        ax[4, 0].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C0")
        ax[4, 1].scatter(mid_bins[is_measures], ihmtr_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors="C0", marker="o")
        ax[4, 1].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C0")
        ax[5, 0].scatter(mid_bins[is_measures], mtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors="C0", marker="o")
        ax[5, 0].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C0")
        ax[5, 1].scatter(mid_bins[is_measures], ihmtsat_ses[ses_id, is_measures],
                        c=nb_voxels_ses[ses_id, is_measures], cmap='Greys', norm=norm,
                        linewidths=1,
                        edgecolors="C0", marker="o")
        ax[5, 1].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C0")
        ax[4, 0].scatter(mid_bins[is_not_measures], mtr_ses[ses_id, is_not_measures],
                                c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                                linewidths=1, alpha=0.5,
                                edgecolors="C0", marker="o")
        ax[4, 1].scatter(mid_bins[is_not_measures], ihmtr_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors="C0", marker="o")
        ax[5, 0].scatter(mid_bins[is_not_measures], mtsat_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors="C0", marker="o")
        ax[5, 1].scatter(mid_bins[is_not_measures], ihmtsat_ses[ses_id, is_not_measures],
                        c=nb_voxels_ses[ses_id, is_not_measures], cmap='Greys', norm=norm,
                        linewidths=1, alpha=0.5,
                        edgecolors="C0", marker="o")
        
    fig.colorbar(colorbar, ax=ax[4:6, 1], location='right', label="Voxel count")

    ax[5, 0].set_xlabel(r'$\theta_a$')
    ax[5, 1].set_xlabel(r'$\theta_a$')
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
    plt.show()
    #plt.savefig(args.out_name, dpi=300)
    #plt.close()


if __name__ == "__main__":
    main()
