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

    mtr = results['MTR']
    mtsat = results['MTsat']
    ihmtr = results['ihMTR']
    ihmtsat = results['ihMTsat']
    nb_voxels = results['Nb_voxels']

    mid_bins = (results['Angle_min'] + results['Angle_max']) / 2.
    
    max_count = np.max(nb_voxels)
    norm = mpl.colors.Normalize(vmin=0, vmax=max_count)
    highres_bins = np.arange(0, 90 + 1, 0.5)

    plot_init()
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig, ax = plt.subplots(2, 2, projection='3d', layout='constrained')
    X, Y = np.meshgrid(mid_bins, mid_bins)
    ax[0, 0].plot_surface(X, Y, mtr, cmap=cm.navia)
    ax[0, 0].set_zlabel('MTR')
    ax[0, 1].plot_surface(X, Y, ihmtr, cmap=cm.navia)
    ax[0, 1].set_zlabel('ihMTR')
    ax[1, 0].plot_surface(X, Y, mtsat, cmap=cm.navia)
    ax[1, 0].set_zlabel('MTsat')
    ax[1, 1].plot_surface(X, Y, ihmtsat, cmap=cm.navia)
    ax[1, 1].set_zlabel('ihMTsat')

    fig.tight_layout()
    views = np.array([[30, -135], [30, 45], [30, -45], [10, -90], [10, 0]])
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_xlabel(r'$\theta_{a1}$')
            ax[i, j].set_ylabel(r'$\theta_{a2}$')
            ax[i, j].view_init(views[0, 0], views[0, 1])
    plt.show()
    # for v, view in enumerate(views[:]):
    #     out_path = out_folder / str(str(nametype) + "_" + str(names[i]) + "_3D_view_" + str(v) + "_2f.png")
    #     ax.view_init(view[0], view[1])
    #     plt.savefig(out_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
