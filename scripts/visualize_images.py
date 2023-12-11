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
    p.add_argument('out_image',
                   help='Path of the output image.')
    
    p.add_argument('--images', nargs='+', default=[],
                   action='append', required=True,
                   help='List of images to visualize.')
    
    p.add_argument('--reference', default=[], required=True,
                   help='Reference image.')
    
    p.add_argument('--slices', nargs=3, default=[77, 90, 59],
                   action='append',
                   help='List indices for where to slice the images.')
    
    p.add_argument('--map',
                   help='Path to the map.')
    
    p.add_argument('--wm_mask',
                   help='Path to the WM mask')
    
    p.add_argument('--mask',
                   help='Path to the brain mask')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    ref = nib.load(args.reference).get_fdata()

    map = nib.load(args.map).get_fdata().astype(np.uint8)
    map = np.ma.masked_where(map == 0, map)

    data_shape = ref.shape

    if args.wm_mask:
        wm_mask = nib.load(args.wm_mask).get_fdata()
        wm_mask = (wm_mask >= 0.9)
    else:
        wm_mask = np.ones((data_shape))

    if args.mask:
        mask = nib.load(args.mask).get_fdata().astype(np.uint8)
    else:
        mask = np.ones((data_shape))

    images, _ = extract_measures(args.images, data_shape)

    x_index = int(args.slices[0]) # 77, 76, 75
    y_index = int(args.slices[1]) # 91, 92, 93 (93)
    z_index = int(args.slices[2]) # 59

    vmax = np.array([26, 4, 12, 1.3])
    vmin = np.array([5, 0, 0, 0])

    plot_init(dims=(10, 15), font_size=20) # (10, 15)

    COLOR = 'white'
    mpl.rcParams['text.color'] = COLOR
    mpl.rcParams['axes.labelcolor'] = COLOR
    mpl.rcParams['xtick.color'] = COLOR
    mpl.rcParams['ytick.color'] = COLOR

    fig, ax = plt.subplots(images.shape[-1] + 1, 4,
                           gridspec_kw={"width_ratios":[1.2, 1.5, 1.0, 0.05]},
                           layout='constrained')
    
    x_image = np.flip(np.rot90(ref[...][x_index, :, :]), axis=1) # 136
    y_image = np.rot90(ref[...][:, y_index, :]) # 176
    z_image = np.rot90(ref[...][:, :, z_index]) # 126

    x_mask = np.flip(np.rot90(map[x_index, :, :]), axis=1)
    y_mask = np.rot90(map[:, y_index, :])
    z_mask = np.rot90(map[:, :, z_index])
    
    ax[0, 0].imshow(y_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
    ax[0, 1].imshow(x_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
    ax[0, 2].imshow(z_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
    
    colorbar = ax[0, 0].imshow(y_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')
    ax[0, 1].imshow(x_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')
    ax[0, 2].imshow(z_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')

    cb = fig.colorbar(colorbar, cax=ax[0, 3], ax=ax[0, 3])
    cb.outline.set_color('white')
    cb.set_ticks([0, 30, 60, 90])

    for i in range(images.shape[-1]):
        images[..., i] = np.where(mask == 0, np.nan, images[..., i])
        images[..., i] = np.where(wm_mask == 0, np.nan, images[..., i])
        if i == 3:
            images[..., i] = np.clip(images[..., i], 0 , 1.4)
        x_image = np.flip(np.rot90(images[..., i][x_index, :, :]), axis=1)
        y_image = np.rot90(images[..., i][:, y_index, :])
        z_image = np.rot90(images[..., i][:, :, z_index])

        colorbar = ax[i + 1, 0].imshow(y_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')
        ax[i + 1, 1].imshow(x_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')
        ax[i + 1, 2].imshow(z_image, cmap=cm.navia, vmin=vmin[i], vmax=vmax[i], interpolation='none')

        cb = fig.colorbar(colorbar, cax=ax[i + 1, 3])
        cb.outline.set_color('white')

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1] - 1):
            ax[i, j].set_axis_off()
            ax[i, j].autoscale(False)
    
    # ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03])
    # slider = Slider(ax_slider, 'Slide->', 0.0, 136.0, valinit=93)

    def update(val):
        y_index = int(val)
        x_image = np.flip(np.rot90(ref[...][x_index, :, :]), axis=1)
        y_image = np.rot90(ref[...][:, y_index, :])
        z_image = np.rot90(ref[...][:, :, z_index])

        x_mask = np.flip(np.rot90(mask[x_index, :, :]), axis=1)
        y_mask = np.rot90(mask[:, y_index, :])
        z_mask = np.rot90(mask[:, :, z_index])
        
        ax[0, 0].imshow(y_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        ax[0, 1].imshow(x_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        ax[0, 2].imshow(z_image, cmap="gray", vmin=0, vmax=1, interpolation='none')
        
        colorbar = ax[0, 0].imshow(y_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')
        ax[0, 1].imshow(x_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')
        ax[0, 2].imshow(z_mask, cmap=cm.navia, vmin=0, vmax=90, interpolation='none')

        fig.colorbar(colorbar, cax=ax[0, 3], ax=ax[0, 3],
                    label=r'$\theta_a$')

        for i in range(images.shape[-1]):
            x_image = np.flip(np.rot90(images[..., i][x_index, :, :]), axis=1)
            y_image = np.rot90(images[..., i][:, y_index, :])
            z_image = np.rot90(images[..., i][:, :, z_index])

            colorbar = ax[i + 1, 0].imshow(y_image, cmap="gray", vmin=vmin[i], vmax=vmax[i], interpolation='none')
            ax[i + 1, 1].imshow(x_image, cmap="gray", vmin=vmin[i], vmax=vmax[i], interpolation='none')
            ax[i + 1, 2].imshow(z_image, cmap="gray", vmin=vmin[i], vmax=vmax[i], interpolation='none')

            fig.colorbar(colorbar, cax=ax[i + 1, 3])

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1] - 1):
                ax[i, j].set_axis_off()
                ax[i, j].autoscale(False)
        fig.canvas.draw_idle()
        print(val)

    # slider.on_changed(update)
    fig.get_layout_engine().set(h_pad=0, hspace=0)
    # fig.tight_layout()
    # plt.show()
    plt.savefig(args.out_image, dpi=300, transparent=True)



if __name__ == "__main__":
    main()
