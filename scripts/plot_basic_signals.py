import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (extract_measures, plot_init)
from modules.orientation_dependence import compute_single_fiber_means

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_e1',
                   help='Path of the DTI peaks. The peaks are expected to be '
                        'given as unit directions.')
    p.add_argument('in_fa',
                   help='Path of the FA.')
    p.add_argument('in_nufo',
                   help='Path to the NuFO.')
    p.add_argument('in_wm_mask',
                   help='Path of the WM mask.')
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--measures', nargs='+', default=[],
                   action='append', required=True,
                   help='List of measures to characterize.')

    p.add_argument('--in_roi',
                   help='Path to the ROI for where to analyze.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=1, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load the data
    fa_img = nib.load(args.in_fa)
    wm_mask_img = nib.load(args.in_wm_mask)
    e1_img = nib.load(args.in_e1)
    nufo_img = nib.load(args.in_nufo)

    nufo = nufo_img.get_fdata()
    e1 = e1_img.get_fdata()
    fa = fa_img.get_fdata()
    wm_mask = wm_mask_img.get_fdata()

    affine = fa_img.affine

    min_nb_voxels = args.min_nb_voxels

    if args.in_roi:
        roi_img = nib.load(args.in_roi)
        roi = roi_img.get_fdata()
    else:
        roi = None

    measures, _ = extract_measures(args.measures,
                                                fa.shape)


    #----------------------- Single-fiber section -----------------------------
    # Pre-loop just for calculating norm

    plot_init(dims=(10, 5), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize']=2
    fig, ax = plt.subplots(2, 2,
                            gridspec_kw={"width_ratios":[1, 1]},
                            layout='constrained')
    bins, measure_means, nb_voxels =\
        compute_single_fiber_means(e1, fa,
                                wm_mask,
                                affine,
                                measures,
                                nufo=nufo,
                                mask=roi,
                                bin_width=1,
                                fa_thr=0.5)

    is_measures = nb_voxels >= min_nb_voxels
    mid_bins = (bins[:-1] + bins[1:]) / 2

    Sp = measure_means[..., 0]
    Sn = measure_means[..., 1]
    Spn = measure_means[..., 2]
    Snp = measure_means[..., 3]
    S0 = measure_means[..., 4]
    T1 = measure_means[..., 5] * 3

    mean_SpSn = (Sp + Sn) / 2
    mean_SpnSnp = (Spn + Snp) / 2
    mean_SpSn_S0 = mean_SpSn / S0
    mean_SpnSnp_S0 = mean_SpnSnp / S0
    mtr = 1 - mean_SpSn_S0
    ihmtr = mean_SpSn_S0 - mean_SpnSnp_S0

    ax[0, 0].scatter(mid_bins[is_measures], mean_SpSn[is_measures],
                                label=r"(S$_+$+S$_-)/2$", linewidths=1,
                                color=cm.naviaS(2), marker="o")
    ax[0, 0].scatter(mid_bins[is_measures], Sp[is_measures],
                                #label=r"S$_+$", linewidths=1,
                                color=cm.naviaS(2), marker=">", alpha=0.5,)
    ax[0, 0].scatter(mid_bins[is_measures], Sn[is_measures],
                                #label=r"S$_+$", linewidths=1,
                                color=cm.naviaS(2), marker="<", alpha=0.5)
    ax[0, 0].scatter(mid_bins[is_measures], mean_SpnSnp[is_measures],
                                label=r"(S$_{+-}$+S$_{-+})/2$", linewidths=1,
                                color=cm.naviaS(3), marker="o")
    ax[0, 0].scatter(mid_bins[is_measures], Spn[is_measures],
                                #label=r"S$_{+-}$", linewidths=1,
                                color=cm.naviaS(3), marker=">", alpha=0.5)
    ax[0, 0].scatter(mid_bins[is_measures], Snp[is_measures],
                                #label=r"S$_{-+}$", linewidths=1,
                                color=cm.naviaS(3), marker="<", alpha=0.5)
    ax[0, 0].scatter(mid_bins[is_measures], S0[is_measures],
                                label=r"S$_0$", linewidths=1,
                                color=cm.naviaS(4), marker="o")
    # ax[0, 0].scatter(mid_bins[is_measures], T1[is_measures],
    #                             label=r"T$_1$", linewidths=1,
    #                             color=cm.naviaS(4), marker="o")
    ax[0, 1].scatter(mid_bins[is_measures], mean_SpSn_S0[is_measures],
                                label=r"(S$_+$+S$_-)/2$S$_0$", linewidths=1,
                                color=cm.naviaS(5), marker="o")
    ax[0, 1].scatter(mid_bins[is_measures], mean_SpnSnp_S0[is_measures],
                                label=r"(S$_{+-}$+S$_{-+})/2$S$_0$", linewidths=1,
                                color=cm.naviaS(8), marker="o")
    ax[1, 0].scatter(mid_bins[is_measures], mtr[is_measures],
                                label=r"1 - (S$_+$+S$_-)/2$S$_0$", linewidths=1,
                                color=cm.naviaS(5), marker="o")
    ax[1, 1].scatter(mid_bins[is_measures], ihmtr[is_measures],
                                label=r"(S$_+$+S$_-)/2$S$_0$-(S$_{+-}$+S$_{-+})/2$S$_0$", linewidths=1,
                                color=cm.naviaS(8), marker="o")

    # ax[0, 0].set_ylim(0.975 * np.nanmin(mtr), 1.025 * np.nanmax(mtr))
    # ax[0, 1].set_ylim(0.975 * np.nanmin(ihmtr), 1.025 * np.nanmax(ihmtr))
    # ax[1, 0].set_ylim(0.975 * np.nanmin(mtsat), 1.025 * np.nanmax(mtsat))
    # ax[1, 1].set_ylim(0.975 * np.nanmin(ihmtsat), 1.025 * np.nanmax(ihmtsat))

    ax[0, 0].set_ylabel('Basic signals')
    ax[0, 1].set_ylabel('Normalized signals')
    ax[1, 0].set_ylabel("MTR")
    ax[1, 1].set_ylabel("ihMTR")

    ax[0, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[0, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax[1, 0].set_xlabel(r'$\theta_a$')
    ax[1, 1].set_xlabel(r'$\theta_a$')
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_xlim(0, 90)
    ax[0, 0].legend(loc=5, bbox_to_anchor=(1,0.6))
    ax[0, 1].legend(loc=6)
    ax[1, 0].legend(loc=2)
    ax[1, 1].legend(loc=1)
    # ax[0, 1].get_legend().set_title(r"FA threshold")

    # fig.tight_layout()
    # plt.show()
    plt.savefig("basic_signals_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
