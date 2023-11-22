import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.interpolate import splrep, BSpline

from modules.io import plot_init
from modules.orientation_dependence import analyse_delta_m_max

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_prefix',
                   help='Output prefix and path.')
    
    p.add_argument('in_2f_results',
                   help='Results file for the crossing fibers analysis.')
    
    p.add_argument('--in_3f_results',
                   help='Results file for the crossing fibers analysis.')
    
    p.add_argument('--delta_max', action="store_true",
                   help='Plot delta_max or not.')

    g = p.add_argument_group(title='Characterization parameters')
    g.add_argument('--min_nb_voxels', default=30, type=int,
                   help='Value of the minimal number of voxels per bin '
                        '[%(default)s].')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    min_nb_voxels = args.min_nb_voxels
    results = np.load(args.in_2f_results)
    f = 0

    mtr = results['MTR']
    mtsat = results['MTsat']
    ihmtr = results['ihMTR']
    ihmtsat = results['ihMTsat']
    nb_voxels = results['Nb_voxels']

    bins = np.concatenate((results['Angle_min'], [results['Angle_max'][-1]]))
    mid_bins = (results['Angle_min'] + results['Angle_max']) / 2.

    views = np.array([[45, -135], [45, 45], [30, -45], [10, -90], [10, 0]])

    is_measures = nb_voxels >= min_nb_voxels
    is_not_measures = np.invert(is_measures)

    mtr[is_not_measures] = None
    ihmtr[is_not_measures] = None
    mtsat[is_not_measures] = None
    ihmtsat[is_not_measures] = None

    mtr_norm = mpl.colors.Normalize(vmin=np.nanmin(mtr[f]), vmax=np.nanmax(mtr[f]))
    ihmtr_norm = mpl.colors.Normalize(vmin=np.nanmin(ihmtr[f]), vmax=np.nanmax(ihmtr[f]))
    mtsat_norm = mpl.colors.Normalize(vmin=np.nanmin(mtsat[f]), vmax=np.nanmax(mtsat[f]))
    ihmtsat_norm = mpl.colors.Normalize(vmin=np.nanmin(ihmtsat[f]), vmax=np.nanmax(ihmtsat[f]))

    # -----------------------------3D plot-------------------------------------

    plot_init(dims=(10, 10), font_size=10)
    plt.rcParams['axes.labelsize'] = 12
    fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    plt.subplots_adjust(left=0, bottom=0.04, right=1, top=1, wspace=0.00, hspace=0.00)
    X, Y = np.meshgrid(mid_bins, mid_bins)
    ax[0, 0].plot_surface(X, Y, mtr[f], cmap=cm.navia, norm=mtr_norm)
    ax[0, 0].set_zlabel(' MTR ', labelpad=10)
    ax[0, 0].view_init(views[0, 0], views[0, 1])
    # ax[0, 0].set_xticklabels(ax[0, 0].get_xticklabels(), rotation=90)
    ax[0, 1].plot_surface(X, Y, ihmtr[f], cmap=cm.navia, norm=ihmtr_norm)
    ax[0, 1].set_zlabel('ihMTR', labelpad=10)
    ax[0, 1].view_init(views[1, 0], views[1, 1])
    ax[1, 0].plot_surface(X, Y, mtsat[f], cmap=cm.navia, norm=mtsat_norm)
    ax[1, 0].set_zlabel('MTsat', labelpad=10)
    ax[1, 0].view_init(views[0, 0], views[0, 1])
    ax[1, 1].plot_surface(X, Y, ihmtsat[f], cmap=cm.navia, norm=ihmtsat_norm)
    ax[1, 1].set_zlabel('ihMTsat', labelpad=10)
    ax[1, 1].view_init(views[1, 0], views[1, 1])

    fig.tight_layout()
    # fig.get_layout_engine().set(h_pad=0, hspace=0, w_pad=0, wspace=0)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_xlabel(r'$\theta_{a1}$', labelpad=8)
            ax[i, j].set_ylabel(r'$\theta_{a2}$', labelpad=8)
            ax[i, j].set_box_aspect(aspect=None, zoom=0.85)

    # plt.subplot_tool()
    # plt.show()
    # for v, view in enumerate(views[:]):
    #     out_path = out_folder / str(str(nametype) + "_" + str(names[i]) + "_3D_view_" + str(v) + "_2f.png")
    #     ax.view_init(view[0], view[1])
    plt.savefig(args.out_prefix + "3D_2f.png", dpi=300)
    plt.close()

    # -----------------------------2D plot-------------------------------------

    mtr = results['MTR']
    mtsat = results['MTsat']
    ihmtr = results['ihMTR']
    ihmtsat = results['ihMTsat']
    nb_voxels = results['Nb_voxels']

    mtr_diag = np.diagonal(mtr, axis1=1, axis2=2)
    ihmtr_diag = np.diagonal(ihmtr, axis1=1, axis2=2)
    mtsat_diag = np.diagonal(mtsat, axis1=1, axis2=2)
    ihmtsat_diag = np.diagonal(ihmtsat, axis1=1, axis2=2)
    nb_voxels_diag = np.diagonal(nb_voxels, axis1=1, axis2=2)

    max_count = np.max(nb_voxels_diag)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    highres_bins = np.arange(0, 90 + 1, 0.5)

    labels = np.array(["[0.5, 0.6[", "[0.6, 0.7[", "[0.7, 0.8[", "[0.8, 0.9["])

    plot_init(dims=(12, 5), font_size=10)
    plt.rcParams['lines.linewidth'] = 0.5

    fig, ax = plt.subplots(2, 2,
                           gridspec_kw={"width_ratios":[1, 1]},
                           layout='constrained')
    for i in range(mtr_diag.shape[0]):
        is_measures = nb_voxels_diag[i] >= min_nb_voxels
        is_not_measures = np.invert(is_measures)
        is_not_nan = nb_voxels_diag[i] > 0
        colorbar = ax[0, 0].scatter(mid_bins[is_measures], mtr_diag[i, is_measures],
                                    c=nb_voxels_diag[i, is_measures], cmap='Greys', norm=norm,
                                    label=labels[i], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
        ax[0, 1].scatter(mid_bins[is_measures], ihmtr_diag[i, is_measures],
                                    c=nb_voxels_diag[i, is_measures], cmap='Greys', norm=norm,
                                    label=labels[i], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
        ax[1, 0].scatter(mid_bins[is_measures], mtsat_diag[i, is_measures],
                                    c=nb_voxels_diag[i, is_measures], cmap='Greys', norm=norm,
                                    label=labels[i], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
        ax[1, 1].scatter(mid_bins[is_measures], ihmtsat_diag[i, is_measures],
                                    c=nb_voxels_diag[i, is_measures], cmap='Greys', norm=norm,
                                    label=labels[i], linewidths=1,
                                    edgecolors="C" + str(i), marker="o")
        ax[0, 0].scatter(mid_bins[is_not_measures], mtr_diag[i, is_not_measures],
                                    c=nb_voxels_diag[i, is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors="C" + str(i), marker="o")
        ax[0, 1].scatter(mid_bins[is_not_measures], ihmtr_diag[i, is_not_measures],
                                    c=nb_voxels_diag[i, is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors="C" + str(i), marker="o")
        ax[1, 0].scatter(mid_bins[is_not_measures], mtsat_diag[i, is_not_measures],
                                    c=nb_voxels_diag[i, is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors="C" + str(i), marker="o")
        ax[1, 1].scatter(mid_bins[is_not_measures], ihmtsat_diag[i, is_not_measures],
                                    c=nb_voxels_diag[i, is_not_measures], cmap='Greys', norm=norm,
                                    linewidths=1, alpha=0.5,
                                    edgecolors="C" + str(i), marker="o")
        
        weights = np.sqrt(nb_voxels_diag[i, is_not_nan]) / np.max(nb_voxels_diag[i, is_not_nan])

        mtr_fit = splrep(mid_bins[is_not_nan], mtr_diag[i, is_not_nan], w=weights, s=0.0002)
        mtsat_fit = splrep(mid_bins[is_not_nan], mtsat_diag[i, is_not_nan], w=weights, s=0.001)
        ihmtr_fit = splrep(mid_bins[is_not_nan], ihmtr_diag[i, is_not_nan], w=weights, s=0.0001)
        ihmtsat_fit = splrep(mid_bins[is_not_nan], ihmtsat_diag[i, is_not_nan], w=weights, s=0.000001)

        ax[0, 0].plot(highres_bins, BSpline(*mtr_fit)(highres_bins), "--", color="C" + str(i))
        ax[0, 1].plot(highres_bins, BSpline(*ihmtr_fit)(highres_bins), "--", color="C" + str(i))
        ax[1, 0].plot(highres_bins, BSpline(*mtsat_fit)(highres_bins), "--", color="C" + str(i))
        ax[1, 1].plot(highres_bins, BSpline(*ihmtsat_fit)(highres_bins), "--", color="C" + str(i))

    ax[0, 0].set_ylim(0.985 * np.nanmin(mtr_diag), 1.015 * np.nanmax(mtr_diag))
    ax[0, 1].set_ylim(0.985 * np.nanmin(ihmtr_diag), 1.015 * np.nanmax(ihmtr_diag))
    ax[1, 0].set_ylim(0.985 * np.nanmin(mtsat_diag), 1.015 * np.nanmax(mtsat_diag))
    ax[1, 1].set_ylim(0.985 * np.nanmin(ihmtsat_diag), 1.015 * np.nanmax(ihmtsat_diag))

    ax[0, 0].set_ylabel("MTR")
    ax[0, 1].set_ylabel("ihMTR")
    ax[1, 0].set_ylabel("MTsat")
    ax[1, 1].set_ylabel("ihMTsat")

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_xlabel(r'$\theta_a$')
            ax[i, j].set_xlim(0, 90)
    ax[0, 1].legend(loc=1)
    ax[0, 1].get_legend().set_title(r"Peak$_1$ fraction")

    if args.delta_max is not None:
        sf_delta_m_max = 2.52904729
        slope, origin, delta_max, frac_thrs_mid, min_bins, max_bins =\
            analyse_delta_m_max(bins, mtr_diag.reshape((mtr_diag.shape) + (1,)),
                                sf_delta_m_max, nb_voxels_diag)

        # this is an inset axes over the main axes
        highres_frac = np.arange(0, 1.01, 0.01)
        # ax = inset_axes(ax1,
        #                 bbox_to_anchor=[0.2, 0.2, 0.2, 0.2],
        #                 width="50%", # width = 40% of parent_bbox
        #                 height=1.0) # height : 1 inch
        #                 # loc=2)
        ax2 = plt.axes([0.22, 0.88, 0.1, 0.1])
        for i in range(len(frac_thrs_mid) - 1):
            ax2.scatter(frac_thrs_mid[i], delta_max[i], color="C" + str(i),
                        linewidths=1)
        ax2.scatter(frac_thrs_mid[-1], delta_max[-1], color="black",
                    linewidths=1)
        ax2.plot(highres_frac,
                slope * highres_frac + origin,
                "--",
                color="grey")
        ax2.set_xlabel(r'Peak$_1$ fraction', labelpad=-2)
        xlim=[0, 1.03]
        ax2.set_xlim(xlim[0], xlim[1])
        ax2.set_ylabel( r'$\delta m_{max}$', labelpad=0)

    fig.colorbar(colorbar, ax=ax[:, 1], location='right', label="Voxel count")
    # fig.tight_layout()
    # plt.show()
    plt.savefig(args.out_prefix + "2D_2f.png", dpi=300)
    plt.close()

# -------------------------------2D 3f plot---------------------------------
    if args.in_3f_results is not None:
        results_3f = np.load(args.in_3f_results)
        mid_bins_3f = (results_3f['Angle_min'] + results_3f['Angle_max']) / 2.
        mtr_3f = results_3f['MTR']
        mtsat_3f = results_3f['MTsat']
        ihmtr_3f = results_3f['ihMTR']
        ihmtsat_3f = results_3f['ihMTsat']
        nb_voxels_3f = results_3f['Nb_voxels']

        max_count = np.max(nb_voxels_3f)
        norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
        highres_bins = np.arange(0, 90 + 1, 0.5)

        labels = np.array(["[0.3, 0.4[", "[0.4, 0.5[", "[0.5, 0.6[", "[0.6, 0.7[", "[0.7, 0.8[", "[0.8, 0.9["])

        plot_init(dims=(12, 5), font_size=10)

        fig, ax = plt.subplots(2, 2,
                            gridspec_kw={"width_ratios":[1, 1]},
                            layout='constrained')
        for i in range(mtr_3f.shape[0]):
            is_measures = nb_voxels_3f[i] >= min_nb_voxels
            is_not_measures = np.invert(is_measures)
            is_not_nan = nb_voxels_3f[i] > 0
            colorbar = ax[0, 0].scatter(mid_bins_3f[is_measures], mtr_3f[i, is_measures],
                                        c=nb_voxels_3f[i, is_measures], cmap='Greys', norm=norm,
                                        label=labels[i], linewidths=1,
                                        edgecolors="C" + str(i), marker="o")
            ax[0, 1].scatter(mid_bins_3f[is_measures], ihmtr_3f[i, is_measures],
                                        c=nb_voxels_3f[i, is_measures], cmap='Greys', norm=norm,
                                        label=labels[i], linewidths=1,
                                        edgecolors="C" + str(i), marker="o")
            ax[1, 0].scatter(mid_bins_3f[is_measures], mtsat_3f[i, is_measures],
                                        c=nb_voxels_3f[i, is_measures], cmap='Greys', norm=norm,
                                        label=labels[i], linewidths=1,
                                        edgecolors="C" + str(i), marker="o")
            ax[1, 1].scatter(mid_bins_3f[is_measures], ihmtsat_3f[i, is_measures],
                                        c=nb_voxels_3f[i, is_measures], cmap='Greys', norm=norm,
                                        label=labels[i], linewidths=1,
                                        edgecolors="C" + str(i), marker="o")
            ax[0, 0].scatter(mid_bins_3f[is_not_measures], mtr_3f[i, is_not_measures],
                                        c=nb_voxels_3f[i, is_not_measures], cmap='Greys', norm=norm,
                                        linewidths=1, alpha=0.5,
                                        edgecolors="C" + str(i), marker="o")
            ax[0, 1].scatter(mid_bins_3f[is_not_measures], ihmtr_3f[i, is_not_measures],
                                        c=nb_voxels_3f[i, is_not_measures], cmap='Greys', norm=norm,
                                        linewidths=1, alpha=0.5,
                                        edgecolors="C" + str(i), marker="o")
            ax[1, 0].scatter(mid_bins_3f[is_not_measures], mtsat_3f[i, is_not_measures],
                                        c=nb_voxels_3f[i, is_not_measures], cmap='Greys', norm=norm,
                                        linewidths=1, alpha=0.5,
                                        edgecolors="C" + str(i), marker="o")
            ax[1, 1].scatter(mid_bins_3f[is_not_measures], ihmtsat_3f[i, is_not_measures],
                                        c=nb_voxels_3f[i, is_not_measures], cmap='Greys', norm=norm,
                                        linewidths=1, alpha=0.5,
                                        edgecolors="C" + str(i), marker="o")

        ax[0, 0].set_ylim(0.985 * np.nanmin(mtr_3f), 1.015 * np.nanmax(mtr_3f))
        ax[0, 1].set_ylim(0.985 * np.nanmin(ihmtr_3f), 1.015 * np.nanmax(ihmtr_3f))
        ax[1, 0].set_ylim(0.985 * np.nanmin(mtsat_3f), 1.015 * np.nanmax(mtsat_3f))
        ax[1, 1].set_ylim(0.985 * np.nanmin(ihmtsat_3f), 1.015 * np.nanmax(ihmtsat_3f))

        ax[0, 0].set_ylabel("MTR")
        ax[0, 1].set_ylabel("ihMTR")
        ax[1, 0].set_ylabel("MTsat")
        ax[1, 1].set_ylabel("ihMTsat")

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].set_xlabel(r'$\theta_a$')
                ax[i, j].set_xlim(0, 90)
        ax[0, 1].legend(loc=2)
        ax[0, 1].get_legend().set_title(r"Peak$_1$ fraction")

        fig.colorbar(colorbar, ax=ax[:, 1], location='right', label="Voxel count")
        # fig.tight_layout()
        # plt.show()
        plt.savefig(args.out_prefix + "2D_3f.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
