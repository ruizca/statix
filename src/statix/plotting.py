"""
Convinience functions for plotting XMM data

@author: A.Ruiz
"""
import numpy as np
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.text import Text
from regions import CirclePixelRegion, PixCoord


def src_regions(srclist, color="#CF5C36", linestyle="-"):
    regions, labels = [], []
    for i, src in enumerate(srclist):
        try:
            radius = 2 * src["EXTENT"]
        except KeyError:
            radius = 8

        regions.append(
            CirclePixelRegion(
                PixCoord(x=src["X_IMA"], y=src["Y_IMA"]), radius=radius
            ).as_artist(edgecolor=color, linewidth=2.5, linestyle=linestyle)
        )
        labels.append(
            Text(
                x=src["X_IMA"],
                y=src["Y_IMA"],
                text=f"    {i}",
                color=color,
                fontsize="x-large",
                fontweight="bold",
                #horizontalalignment="right"
            )
        )
    
    return zip(regions, labels)


def plot_image(
    axis,
    image,
    title=None,
    with_inset=False,
    srclist=None,
    use_norm=True,
    norm=None,
    cmap=None,
    add_labels=False
):
    if use_norm and not norm:
        norm = simple_norm(image, 'log')
        
    if not cmap:
        cmap = plt.cm.gray

    axis.imshow(image, origin="lower", norm=norm, cmap=cmap)
    
    if srclist:
        for reg, label in src_regions(srclist, color="green"):
            axis.add_artist(reg)
                
            if add_labels:
                axis.add_artist(label)

    axis.set_xticks([])
    axis.set_yticks([])

    if title:
        axis.set_title(f"{title}\n{image.shape}")

    if with_inset:
        # inset axes....
        #axins = axis.inset_axes([0, 0.05, 0.9, 0.3])
        axins = axis.inset_axes([0, 0.02, 0.5, 0.5])

        if use_norm and not norm:
            norm = simple_norm(image, 'asinh')

        axins.imshow(image, origin="lower", norm=norm, cmap=cmap)

        if srclist:
            for reg, label in src_regions(srclist):
                axins.add_artist(reg)
                    

        # sub region of the original image
        axins.set_ylim(200, 280)
        #axins.set_xlim(210, 450)
        axins.set_xlim(285, 365)
        axins.set_xticks([])
        axins.set_yticks([])
        
        axis.indicate_inset_zoom(axins, edgecolor="r", alpha=1)


def plot_lightcurve(ax, lc, lc_bb, animated=False):
    im = ax.plot(lc[:, 0], color="k", lw=1, animated=animated)
    im = ax.plot(lc[:, 1], color="grey", lw=1, animated=animated)

    idx = []
    idx_start = 0
    for row in lc_bb:
        idx.append(idx_start + row[2] / 2 - 0.5)
        idx_start += row[2]

    im = ax.errorbar(
        idx, 
        lc_bb[:, 3] / lc_bb[:, 2], 
        xerr=lc_bb[:, 2]/2, 
        yerr=np.sqrt(lc_bb[:, 3]) / lc_bb[:, 2],
        color="k",
        lw=0,
        elinewidth=2.5,
        capsize=5,
        capthick=2.5,
        marker="o",
        ms=8,
        animated=animated,
    )
    im = ax.errorbar(
        idx, 
        lc_bb[:, 4] / lc_bb[:, 2], 
        xerr=lc_bb[:, 2]/2, 
        yerr=np.sqrt(lc_bb[:, 4]) / lc_bb[:, 2],
        color="darkgrey",
        lw=0,
        elinewidth=2.5,
        capsize=5,
        capthick=2.5,
        marker="o",
        ms=8,
        animated=animated,
    )

    ax.set_xlim(-0.8, 32.3)
    ax.set_ylim(0)

    ax.set_xlabel("frame index")
    ax.set_ylabel("counts")
    
    return im
