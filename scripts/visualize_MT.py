import argparse
from cmcrameri import cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path

from modules.io import plot_init, extract_measures

from scilpy.io.utils import (add_overwrite_arg)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('out_image',
                   help='Path of the output image.')
    
    p.add_argument('--images', nargs='+', default=[],
                   action='append', required=True,
                   help='List of images to visualize.')
    
    p.add_argument('--slices', nargs=3, default=[50, 60, 30],
                   action='append',
                   help='List indices for where to slice the images.')

    p.add_argument('--mask',
                   help='Path to the brain mask')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    data_shape = nib.load(args.images[0][0]).get_fdata().shape

    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(np.uint8)
    else:
        mask = np.ones((data_shape))

    images, _ = extract_measures(args.images, data_shape)

    x_index = int(args.slices[0]) # 77, 76, 75
    y_index = int(args.slices[1]) # 91, 92, 93 (93)
    z_index = int(args.slices[2]) # 59

    vmax = np.array([4.5, 4.5, 4.5, 4.5])
    vmin = np.array([0, 0, 0, 0])

    plot_init(dims=(10, 10), font_size=20) # (10, 15)

    COLOR = 'black'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR

    fig, ax = plt.subplots(images.shape[-1], 4,
                           gridspec_kw={"width_ratios":[1.2, 1.5, 1.0, 0.05]},
                           layout='constrained')

    for i in range(images.shape[-1]):
        images[..., i] = np.where(mask == 0, np.nan, images[..., i])
        if i == 3:
            images[..., i] = np.clip(images[..., i], 0 , 1.4)
        x_image = np.flip(np.rot90(images[..., i][x_index, :, :]), axis=1)
        y_image = np.rot90(images[..., i][:, y_index, :])
        z_image = np.rot90(images[..., i][:, :, z_index])

        colorbar = ax[i, 0].imshow(y_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')
        ax[i, 1].imshow(x_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')
        ax[i, 2].imshow(z_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')

        cb = fig.colorbar(colorbar, cax=ax[i, 3])
        # cb.outline.set_color('white')

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1] - 1):
            ax[i, j].set_axis_off()
            ax[i, j].autoscale(False)

    fig.get_layout_engine().set(h_pad=0, hspace=0)
    # fig.tight_layout()
    plt.show()
    # plt.savefig(args.out_image, dpi=300, transparent=True)



if __name__ == "__main__":
    main()
