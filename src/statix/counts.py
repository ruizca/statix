import logging

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import hstack
from photutils import SourceCatalog
from photutils.aperture import (
    aperture_photometry,
    CircularAperture,
    SkyEllipticalAperture,
)
from scipy.stats import poisson

from . import psf

logger = logging.getLogger(__name__)


def extract_segmentation(exposure, image_segmented, srclist):
    cube = exposure.cube.data
    cube_bkg = exposure.bkgcube.data

    nz = exposure.cube.shape[0]
    nsrc = len(srclist)
    lc = np.zeros((nsrc, nz, 2))

    for src_num in range(nsrc):
        for zframe in range(nz):
            cat = SourceCatalog(
                cube[zframe, :, :], image_segmented, background=cube_bkg[zframe, :, :]
            )
            lc[nsrc, zframe, 0] = cat[src_num].source_sum
            lc[nsrc, zframe, 1] = cat[src_num].background_sum

    srclist["LC"] = lc

    return srclist


def extract_aperture_cube(exposure, srclist):
    cube = exposure.cube.data
    cube_bkg = exposure.bkgcube.data
    mask = exposure.mask.data

    return _extract_counts_aperture(cube, cube_bkg, mask, srclist)


def extract_aperture_image(exposure, srclist):
    cube = exposure.image.data[np.newaxis,]
    cube_bkg = exposure.bkgimage.data[np.newaxis,]
    mask = exposure.mask.data

    return _extract_counts_aperture(cube, cube_bkg, mask, srclist)


def _extract_counts_aperture(cube, cube_bkg, mask, srclist):
    nframes = cube.shape[0]
    nsrc = len(srclist)
    lc = np.zeros((nsrc, nframes, 2))

    positions = np.array([srclist["X_IMA"], srclist["Y_IMA"]]).T
    aperture = CircularAperture(positions, r=7.0)

    for idx_frame in range(nframes):
        lc[:, idx_frame, 0] = _get_frame_photometry(cube[idx_frame, :, :], aperture, mask)
        lc[:, idx_frame, 1] = _get_frame_photometry(cube_bkg[idx_frame, :, :], aperture, mask)

    srclist["LC"] = lc

    return srclist

def extract_aperture_psf_cube(exposure, *args):
    cube = exposure.cube.data
    cube_bkg = exposure.bkgcube.data
    mask = exposure.mask.data
    wcs = exposure.image.wcs
    camera = exposure.camera

    return _extract_counts_aperture_psf(cube, cube_bkg, mask, wcs, camera, *args)


def extract_aperture_psf_image(exposure, *args):
    cube = exposure.image.data[np.newaxis,]
    cube_bkg = exposure.bkgimage.data[np.newaxis,]
    mask = exposure.mask.data
    wcs = exposure.image.wcs
    camera = exposure.camera

    return _extract_counts_aperture_psf(cube, cube_bkg, mask, wcs, camera, *args)


def _extract_counts_aperture_psf(
    cube, cube_bkg, mask, wcs, camera, srclist, psf_energy, eef
):
    nz = cube.shape[0]
    nsrc = len(srclist)
    lc = np.zeros((nsrc, nz, 2))

    # Group sources in nearby regions with the same psf parameters
    srclist_psf_grouped = _add_psf_data(camera, srclist, psf_energy, eef)

    # For each of these groups, extract the src and bkg light curves using aperture_photometry
    indices = srclist_psf_grouped.groups.indices
    groups = srclist_psf_grouped.groups

    logger.info("Extracting light-curves...")
    for group_first_idx, group in zip(indices, groups):
        aperture = _psf_aperture(group)
        aperture_pixels = aperture.to_pixel(wcs)
        
        for j, m in enumerate(aperture_pixels.to_mask(method="center")):
            src_mask = m.to_image(cube.shape[1:])
            lc[group_first_idx + j, :, 0] = _get_photometry(cube, src_mask, mask)
            lc[group_first_idx + j, :, 1] = _get_photometry(cube_bkg, src_mask, mask)

    srclist_psf_grouped["LC"] = lc

    return srclist_psf_grouped


def _add_psf_data(camera, srclist, psf_energy, eef):
    sky_coords = SkyCoord(srclist["RA"], srclist["DEC"])
    det_coords = camera.coordinates(sky_coords)
    psfdata = psf.get_data(det_coords, psf_energy, eef)
    srclist_psf = hstack([srclist, psfdata], join_type="exact")

    return srclist_psf.group_by("psfid")


def _psf_aperture(psfdata):
    return SkyEllipticalAperture(
        SkyCoord(psfdata["RA"], psfdata["DEC"]), 
        psfdata["PSF_a"][0], 
        psfdata["PSF_b"][0], 
        psfdata["PSF_pa"][0],
    )


def _get_photometry(data, src_mask, mask):
    src_data = data * src_mask * mask
    return src_data.sum(axis=(1,2))


def _get_frame_photometry(frame, aperture, mask, wcs=None):
    photometry = aperture_photometry(frame, aperture, mask=mask, wcs=wcs, method="center")
    return photometry["aperture_sum"]


def poisson_probability(src_counts, bkg_counts, log=False):
    if log: 
        p = -poisson.logsf(src_counts - 1, bkg_counts)
    else:
        p = poisson.sf(src_counts - 1, bkg_counts)

    return p
