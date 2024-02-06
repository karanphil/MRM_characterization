import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import (extract_measures, plot_init)
from modules.orientation_dependence import compute_single_fiber_means


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
    p.add_argument('--measures_names', nargs='+', default=[], action='append',
                   help='List of names for the measures to characterize.')

    p.add_argument('--in_roi',
                   help='Path to the ROI for where to analyze.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_frac_thr', default=0.1,
                   help='Value of the minimal fraction threshold for '
                        'selecting peaks to correct [%(default)s].')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    nb_measures = len(args.measures[0])
    if args.measures_names != [] and\
        (len(args.measures_names[0]) != nb_measures):
        parser.error('When using --measures_names, you need to specify ' +
                     'the same number of measures as given in --measures.')

    out_folder = Path(args.out_folder)

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

    measures, measures_name = extract_measures(args.measures,
                                                fa.shape,
                                                args.measures_names)


    #----------------------- Single-fiber section -----------------------------
    # Pre-loop just for calculating norm
    max_count = 0
    for i, fa_thr in enumerate([0.5, 0.6, 0.7]):
        bins, measure_means, nb_voxels =\
            compute_single_fiber_means(e1, fa,
                                    wm_mask,
                                    affine,
                                    measures,
                                    nufo=nufo,
                                    mask=roi,
                                    bin_width=1,
                                    fa_thr=fa_thr)
        if np.max(nb_voxels) > max_count:
            max_count = np.max(nb_voxels)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)

    plot_init(dims=(10, 5), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    fig, ax = plt.subplots(2, 2,
                            gridspec_kw={"width_ratios":[1, 1]},
                            layout='constrained')
    for i, fa_thr in enumerate([0.5, 0.6, 0.7]):
        bins, measure_means, nb_voxels =\
            compute_single_fiber_means(e1, fa,
                                    wm_mask,
                                    affine,
                                    measures,
                                    nufo=nufo,
                                    mask=roi,
                                    bin_width=1,
                                    fa_thr=fa_thr)

        is_measures = nb_voxels >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        is_not_nan = nb_voxels > 0
        mid_bins = (bins[:-1] + bins[1:]) / 2

        mtr = measure_means[..., 0]
        ihmtr = measure_means[..., 1]
        mtsat = measure_means[..., 2]
        ihmtsat = measure_means[..., 3]

        colorbar = ax[0, 0].scatter(mid_bins[is_measures], mtr[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=fa_thr, linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 1].scatter(mid_bins[is_measures], ihmtr[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=fa_thr, linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 0].scatter(mid_bins[is_measures], mtsat[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=fa_thr, linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 1].scatter(mid_bins[is_measures], ihmtsat[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=fa_thr, linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 0].scatter(mid_bins[is_not_measures], mtr[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 1].scatter(mid_bins[is_not_measures], ihmtr[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 0].scatter(mid_bins[is_not_measures], mtsat[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 1].scatter(mid_bins[is_not_measures], ihmtsat[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")

        ax[0, 0].set_ylim(0.975 * np.nanmin(mtr), 1.025 * np.nanmax(mtr))
        ax[0, 1].set_ylim(0.975 * np.nanmin(ihmtr), 1.025 * np.nanmax(ihmtr))
        ax[1, 0].set_ylim(0.975 * np.nanmin(mtsat), 1.025 * np.nanmax(mtsat))
        ax[1, 1].set_ylim(0.975 * np.nanmin(ihmtsat), 1.025 * np.nanmax(ihmtsat))

        ax[0, 0].set_ylabel("MTR")
        ax[0, 1].set_ylabel("ihMTR")
        ax[1, 0].set_ylabel("MTsat")
        ax[1, 1].set_ylabel("ihMTsat")

        ax[0, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[0, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 0].set_xlabel(r'$\theta_a$')
        ax[1, 1].set_xlabel(r'$\theta_a$')
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].set_xlim(0, 90)
        ax[0, 1].legend(loc=1)
        ax[0, 1].get_legend().set_title(r"FA threshold")

    fig.colorbar(colorbar, ax=ax[:, 1], location='right', label="Voxel count")
        # fig.tight_layout()
    # plt.show()
    plt.savefig("fa_thr_plot.png", dpi=300)
    plt.close()

    max_count = 0
    for bin_width in [1, 3, 5, 10]:
        bins, measure_means, nb_voxels =\
            compute_single_fiber_means(e1, fa,
                                    wm_mask,
                                    affine,
                                    measures,
                                    nufo=nufo,
                                    mask=roi,
                                    bin_width=bin_width,
                                    fa_thr=0.5)
        if np.max(nb_voxels) > max_count:
            max_count = np.max(nb_voxels)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    
    plot_init(dims=(10, 5), font_size=10)
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['lines.linewidth'] = 0.5
    fig, ax = plt.subplots(2, 2,
                            gridspec_kw={"width_ratios":[1, 1]},
                            layout='constrained')
    for i, bin_width in enumerate([1, 3, 5, 10]):
        bins, measure_means, nb_voxels =\
            compute_single_fiber_means(e1, fa,
                                    wm_mask,
                                    affine,
                                    measures,
                                    nufo=nufo,
                                    mask=roi,
                                    bin_width=bin_width,
                                    fa_thr=0.5)
        is_measures = nb_voxels >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        is_not_nan = nb_voxels > 0
        mid_bins = (bins[:-1] + bins[1:]) / 2

        mtr = measure_means[..., 0]
        ihmtr = measure_means[..., 1]
        mtsat = measure_means[..., 2]
        ihmtsat = measure_means[..., 3]

        colorbar = ax[0, 0].scatter(mid_bins[is_measures], mtr[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=str(bin_width) + r"$^\circ$ bins", linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 1].scatter(mid_bins[is_measures], ihmtr[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=str(bin_width) + r"$^\circ$ bins", linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 0].scatter(mid_bins[is_measures], mtsat[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=str(bin_width) + r"$^\circ$ bins", linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 1].scatter(mid_bins[is_measures], ihmtsat[is_measures],
                                    c=nb_voxels[is_measures], cmap='Greys', norm=norm,
                                    label=str(bin_width) + r"$^\circ$ bins", linewidths=1,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 0].scatter(mid_bins[is_not_measures], mtr[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[0, 1].scatter(mid_bins[is_not_measures], ihmtr[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 0].scatter(mid_bins[is_not_measures], mtsat[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")
        ax[1, 1].scatter(mid_bins[is_not_measures], ihmtsat[is_not_measures],
                                    c=nb_voxels[is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors=cm.naviaS(i+2), marker="o")

        ax[0, 0].set_ylim(0.975 * np.nanmin(mtr), 1.025 * np.nanmax(mtr))
        ax[0, 1].set_ylim(0.975 * np.nanmin(ihmtr), 1.025 * np.nanmax(ihmtr))
        ax[1, 0].set_ylim(0.975 * np.nanmin(mtsat), 1.025 * np.nanmax(mtsat))
        ax[1, 1].set_ylim(0.975 * np.nanmin(ihmtsat), 1.025 * np.nanmax(ihmtsat))

        ax[0, 0].set_ylabel("MTR")
        ax[0, 1].set_ylabel("ihMTR")
        ax[1, 0].set_ylabel("MTsat")
        ax[1, 1].set_ylabel("ihMTsat")

        ax[0, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[0, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 0].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 1].set_xticks([0, 15, 30, 45, 60, 75, 90])
        ax[1, 0].set_xlabel(r'$\theta_a$')
        ax[1, 1].set_xlabel(r'$\theta_a$')
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].set_xlim(0, 90)
        ax[0, 1].legend(loc=1)
        # ax[0, 1].get_legend().set_title(r"FA threshold")

    fig.colorbar(colorbar, ax=ax[:, 1], location='right', label="Voxel count")
        # fig.tight_layout()
    # plt.show()
    plt.savefig("bin_width_plot.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
