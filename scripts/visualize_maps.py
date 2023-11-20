import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import plot_init, extract_measures

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_folder',
                   help='Path of the output folder for txt, png, masks and '
                        'measures.')
    
    p.add_argument('--maps', nargs='+', default=[],
                   action='append', required=True,
                   help='List of images to visualize.')
    
    p.add_argument('--reference', default=[], required=True,
                   help='Reference image.')
    
    p.add_argument('--slices', nargs=3, default=[],
                   action='append', required=True,
                   help='List indices for where to slice the images.')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ref = nib.load(args.reference).get_fdata()

    data_shape = ref.shape

    maps, _ = extract_measures(args.maps, data_shape)
    maps = np.ma.masked_where(maps == 0, maps)

    x_index = int(args.slices[0][0]) # 53, 54 or 55 (55)
    y_index = int(args.slices[0][1]) # 92, 93, or 94 (93)
    z_index = int(args.slices[0][2]) # 53 (53)

    vmax = np.array([90, 90, 90, 1, 1])

    plot_init(dims=(10, 15), font_size=20)

    fig, ax = plt.subplots(maps.shape[-1], 4,
                           gridspec_kw={"width_ratios":[1.2, 1.5, 1.0, 0.05]},
                           layout='constrained')

    for i in range(maps.shape[-1]):
        x_image = np.flip(np.rot90(ref[x_index, :, :]), axis=1)
        y_image = np.rot90(ref[:, y_index, :])
        z_image = np.rot90(ref[:, :, z_index])

        colorbar = ax[i , 0].imshow(y_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        ax[i, 1].imshow(x_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        ax[i, 2].imshow(z_image, cmap="gray", vmin=0, vmax=1, interpolation='none')

        x_mask = np.flip(np.rot90(maps[..., i][x_index, :, :]), axis=1)
        y_mask = np.rot90(maps[..., i][:, y_index, :])
        z_mask = np.rot90(maps[..., i][:, :, z_index])

        colorbar = ax[i, 0].imshow(y_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')
        ax[i, 1].imshow(x_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')
        ax[i, 2].imshow(z_mask, cmap=cm.navia, vmin=0, vmax=vmax[i], interpolation='none')

        fig.colorbar(colorbar, cax=ax[i, 3])

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1] - 1):
            ax[i, j].set_axis_off()
            ax[i, j].autoscale(False)

    fig.get_layout_engine().set(h_pad=0, hspace=0)
    # fig.tight_layout()
    # plt.show()
    plt.savefig("toto2.png", dpi=300, transparent=True)



if __name__ == "__main__":
    main()
