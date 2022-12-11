"""
Convinience functions for plotting XMM data

@author: A.Ruiz
"""
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
