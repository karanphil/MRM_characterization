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
    p.add_argument('out_name',
                   help='Path of the output name.')
    
    p.add_argument('in_results',
                   help='Results file for the crossing fibers analysis.')

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
    results = np.load(args.in_results)
    f = 0

    mtr = results['MTR']
    mtsat = results['MTsat']
    ihmtr = results['ihMTR']
    ihmtsat = results['ihMTsat']
    nb_voxels = results['Nb_voxels']

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

    plot_init(dims=(10, 10), font_size=10)
    plt.rcParams['axes.labelsize'] = 12
    fig, ax = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
    plt.subplots_adjust(wspace=0.00, hspace=0.00)
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

    # plt.show()
    # for v, view in enumerate(views[:]):
    #     out_path = out_folder / str(str(nametype) + "_" + str(names[i]) + "_3D_view_" + str(v) + "_2f.png")
    #     ax.view_init(view[0], view[1])
    plt.savefig(args.out_name, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
