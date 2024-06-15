"""
Convinience functions for plotting XMM data

@author: A.Ruiz
"""
import numpy as np
from astropy.nddata import Cutout2D
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


def plot_cutout(ax, image, xsrc, ysrc, size=20, label=None, scale="linear"):
    cutout = Cutout2D(image.data, (xsrc, ysrc), wcs=image.wcs, size=size)

    norm_cutout = simple_norm(cutout.data, scale)#, max_percent=99)
    ax.imshow(cutout.data, origin="lower", cmap="hot", norm=norm_cutout)
    ax.set_axis_off()
    ax.autoscale(False, axis="both")

    if label is not None:
        ax.text(0.02, 0.9, label, transform=ax.transAxes, color="w", fontsize="xx-large", fontweight="bold")

    return cutout.wcs


def plot_lightcurve(ax, lc, time_edges, lc_bb, animated=False):
    lc_mid_point = (time_edges[:-1] + time_edges[1:]) / 2 - time_edges[0]
    im = ax.plot(lc_mid_point, lc[:, 0], color="k", lw=1, animated=animated)
    im = ax.plot(lc_mid_point, lc[:, 1], color="grey", lw=1, animated=animated)

    for t in time_edges:
        ax.axvline(t - time_edges[0], ls=":", color="k")


    lc_bb_mid_point = (lc_bb[:, 0] + lc_bb[:, 1]) / 2 - lc_bb[0, 0]
    xerr = (lc_bb[:, 1] - lc_bb[:, 0]) / 2
    
    im = ax.errorbar(
        lc_bb_mid_point, 
        lc_bb[:, 3] / lc_bb[:, 2], 
        xerr=xerr, 
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
    mask_not_significant = lc_bb[:, 5] < 1
    im = ax.errorbar(
        lc_bb_mid_point[mask_not_significant], 
        lc_bb[mask_not_significant, 3] / lc_bb[mask_not_significant, 2], 
        xerr=xerr[mask_not_significant], 
        yerr=np.sqrt(lc_bb[mask_not_significant, 3]) / lc_bb[mask_not_significant, 2],
        color="r",
        lw=0,
        elinewidth=2.5,
        capsize=5,
        capthick=2.5,
        marker="o",
        ms=8,
        animated=animated,
    )    
    im = ax.errorbar(
        lc_bb_mid_point, 
        lc_bb[:, 4] / lc_bb[:, 2], 
        xerr=xerr, 
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

    # ax.set_xlim(-10,  lc_bb[-1, 1] - lc_bb[0, 0] + 10)
    # ax.set_ylim(-0.5)

    ax.set_xlabel(f"time +{lc_bb[0, 0]:.2f} / s")
    ax.set_ylabel("counts per frame")
    
    return im
