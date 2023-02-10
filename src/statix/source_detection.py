import logging
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from photutils import (
    deblend_sources,
    detect_sources,
    SourceCatalog,
    find_peaks,
)
from photutils.utils.exceptions import NoDetectionsWarning

from . import counts, timebins

logger = logging.getLogger(__name__)
warnings.filterwarnings("error", category=NoDetectionsWarning)

try:
    from . import xmmsas

except ImportError as e:
    logger.warn(e)
    logger.warn("SAS-related functions not available!!!")


def msvst2d(
    exposure,
    inpainting=True,
    detmode="peaks",
    nsrcs_limit=np.inf,
    photomode="aperture_psf",
    ecf=7.3866,
    **kwargs
):
    """
    Detect sources using 2D MSVST denoising and 
    poisson statistic to identify significant candidates.

    Parameters
    ----------
    exposure : Exposure,
        Exposure object for running the source detection algorithm.
    inpainting : bool, optional
        Apply an inpainting algorithm for filling CCD gaps in the
        data before denoising. By default True.
    detmode : str, optional
        Method for detection of source candidates. By default "peaks"
    nsrcs_limit : int, optional
        If the number of detected candidates is greater than this number,
        light curves are not extracted.
    photomode : "aperture" or "aperture_psf", optional
        Photometry mode, by default "aperture_psf". 
        "aperture" uses a circular aperture with 7 pixels (~20 arcsec) 
        radius for all sources (fast).
        "aperture_psf" uses elliptical apertures, with their shapes defined 
        by the detector's PSF parameters at the position of each source.
    ecf : float, optional
        Energy conversion factor for calculating physical fluxes,
        in units of 1e11 counts cm2  erg-1. By default 7.3866 (value for 0.5-2 keV).
    eef : int, optional
        Enclosed energy fraction used for the "aperture_psf" photometry
        mode. Available values are 60, 70 and 80, corresponding
        to 60%, 70% anf 80% energy fractions respectively. By defaul 70.

    Returns
    -------
    srclist : Table
        Catalogue of detected sources
    image_msvst : Cube
        2D MSVST transform of the exposure's data cube.
    """
    image = exposure.image

    if inpainting:
        logger.info("Filling image gaps...")
        image = image.fill_gaps(exposure.mask.data, method="mca")

    logger.info("Denoising image using 2D MSVST...")
    kwargs_denoise, kwargs = _parse_kwargs_denoise2d(kwargs)
    image_msvst = image.denoise(**kwargs_denoise)

    if inpainting:
        image_msvst = exposure.mask_fov(image_msvst)

    if detmode == "peaks":
        logger.info("Finding peaks in denoised image...")
        srclist = peak_detection(image_msvst, exposure.mask, sigma=6)
        image_segmented = None
    
    elif detmode == "segment":
        logger.info("Segmentation of denoised image...")
        srclist, image_segmented = image_segmentation(image_msvst)
    else:
        raise ValueError(f"Unknown detmode: {detmode}")

    nsrcs = len(srclist)

    if not nsrcs:
        logger.warn(f"No source candidates detected.")

    elif nsrcs > nsrcs_limit:
        logger.warn(f"{nsrcs} source candidates found. Skipping count extraction.")
    
    else:
        psf_energy, eef = _parse_kwargs_psf(exposure, photomode, kwargs)

        logger.info("Extracting counts for %d sources...", nsrcs)
        srclist = extract_counts(
            exposure, srclist, psf_energy, eef, image_segmented, photomode, dim=2
        )

        logger.info("Calculating fluxes...")
        srclist = calc_fluxes(srclist, exposure.expmap.data, ecf, eef)
        srclist = final_catalog(srclist, photomode, dim=2)

    return srclist, image_msvst


def msvst2d1d(
    exposure,
    inpainting=True,
    detmode="peaks",
    nsrcs_limit=np.inf,
    photomode="aperture_psf",
    time_sigma_level=3,
    ecf=7.3866,
    **kwargs,
):
    """
    Detect sources using 2D+1D MSVST denoising and 
    Bayesian Blocks to identify significant time intervals.

    Parameters
    ----------
    exposure : Exposure,
        Exposure object for running the source detection algorithm.
    inpainting : bool, optional
        Apply an inpainting algorithm for filling CCD gaps in the
        data before denoising. By default True.
    detmode : str, optional
        Method for detection of source candidates. By default "peaks"
    nsrcs_limit : int, optional
        If the number of detected candidates is greater than this number,
        light curves are not extracted.
    photomode : "aperture" or "aperture_psf", optional
        Photometry mode, by default "aperture_psf". 
        "aperture" uses a circular aperture with 7 pixels (~20 arcsec) 
        radius for all sources (fast).
        "aperture_psf" uses elliptical apertures, with their shapes defined 
        by the detector's PSF parameters at the position of each source.
    time_sigma_level : float, optional
        Probability limit (in sigmas) for a time bin being considered 
        significant, by default 3-sigma.
    ecf : float, optional
        Energy conversion factor for calculating physical fluxes,
        in units of 1e11 counts cm2  erg-1. By default 7.3866 (value for 0.5-2 keV).
    eef : int, optional
        Enclosed energy fraction used for the "aperture_psf" photometry
        mode. Available values are 60, 70 and 80, corresponding
        to 60%, 70% anf 80% energy fractions respectively. By defaul 70.
    **kwargs: arguments for MSVST denoising. Check the documentation of the MSVST
        package for a detail description of the parameters.

    Returns
    -------
    srclist : Table
        Catalogue of detected sources
    cube_msvst : Cube
        2D+1D MSVST transform of the exposure's data cube.
    """
    if inpainting:
        cube = exposure.cube_inpaint
    else:
        cube = exposure.cube

    logger.info("Denoising cube using 2D+1D MSVST...")
    kwargs_denoise, kwargs = _parse_kwargs_denoise2d1d(kwargs)
    cube_msvst = cube.denoise(**kwargs_denoise)

    # logger.warn("Using denoised cube of previous MSVST run!")
    # parent, prefix, suffix = exposure.files.get_path_parts()
    # cube_msvst_file = exposure.files._set_path("cube_msvst_psf_3sig", parent, prefix, suffix)
    # cube_msvst = Cube(cube_msvst_file)

    if inpainting:
        cube_msvst = exposure.mask_fov(cube_msvst)

    cube_msvst_time_integrated = cube_msvst.time_integrated
    cube_msvst_time_integrated.wcs = exposure.image.wcs

    if detmode == "peaks":
        logger.info("Finding peaks in denoised cube...")
        srclist = peak_detection(cube_msvst_time_integrated, exposure.mask, sigma=6)
        image_segmented = None

    elif detmode == "segment":
        logger.info("Segmentation of denoised cube...")
        srclist, image_segmented = image_segmentation(cube_msvst_time_integrated)
    else:
        raise ValueError(f"Unknown detmode: {detmode}")

    nsrcs = len(srclist)

    if not nsrcs:
        logger.warn(f"No source candidates detected.")

    elif nsrcs > nsrcs_limit:
        logger.warn(f"{nsrcs} source candidates found. Skipping count extraction.")
    
    else:
        psf_energy, eef = _parse_kwargs_psf(exposure, photomode, kwargs)

        logger.info("Extracting counts in optimized time intervals for %d sources...", nsrcs)
        srclist = extract_counts(
            exposure, srclist, psf_energy, eef, image_segmented, photomode
        )
        srclist = time_binning(srclist, time_sigma_level, cube.time_edges)

        logger.info("Calculating fluxes...")
        srclist = calc_fluxes(srclist, exposure.expmap.data, ecf, eef, cube.shape[0])
        srclist = final_catalog(srclist, photomode)

    return srclist, cube_msvst


def _parse_kwargs_denoise2d(kwargs):
    kwargs_default = {
        "coupled": True,
        "threshold_mode": 0,
        "min_scalexy": 2,
        "max_scalexy": 4,
        "border_mode": 2,
        "sigma_level": 3,
        "kill_last": True,
        "detpos": True,
        "use_non_default_filter": True,
        "verbose": False,
    }
    return _parse_kwargs_denoise(kwargs, kwargs_default)


def _parse_kwargs_denoise2d1d(kwargs):
    kwargs_default = {
        "threshold_mode": 0,
        "min_scalexy": 2,
        "max_scalexy": 4,
        "min_scalez": 1,
        "max_scalez": 4,
        "border_mode": 2,
        "sigma_level": 4,
        "kill_last": True,
        "detpos": True,
        "use_non_default_filter": True,
        "varmodcorr": True,
        "verbose": False,
    }
    return _parse_kwargs_denoise(kwargs, kwargs_default)


def _parse_kwargs_denoise(kwargs, kwargs_default):
    if kwargs is None:
        kwargs = {}

    kwargs_denoise = {key: kwargs.pop(key, item) for key, item in kwargs_default.items()}
    
    return kwargs_denoise, kwargs


def _parse_kwargs_psf(exposure, photomode, kwargs):
    energy = exposure.eband.mean

    # enclosed energy fraction, determined by
    # the aperture radius used to extract counts
    if photomode == "aperture_psf":
        eef = kwargs.pop("eef", 70)
    
    elif photomode == "aperture":
        eef = 88  # PN 30 arcsec (7 pixels)  radius
    
    else:
        eef = 100

    return energy, eef


def peak_detection(image, mask, sigma=3):
    mean, median, std = sigma_clipped_stats(image.data[mask.data > 0], sigma=sigma)

    try:
        srclist = find_peaks(image.data, mean, box_size=3, wcs=image.wcs)
        srclist["X_IMA"] = srclist["x_peak"].astype(float)
        srclist["Y_IMA"] = srclist["y_peak"].astype(float)
        srclist["EXTENT"] = 4

        if image.wcs is not None:
            srclist["RA"] = srclist["skycoord_peak"].ra
            srclist["DEC"] = srclist["skycoord_peak"].dec
        else:
            srclist["RA"] = np.nan
            srclist["DEC"] = np.nan

    except NoDetectionsWarning:
        srclist = Table(names=["X_IMA", "Y_IMA", "EXTENT", "RA", "DEC"])

    return srclist


def image_segmentation(image, deblend=True):
    threshold = _calc_threshold(image.data)
    image_segmented = detect_sources(image.data, threshold, npixels=5)

    if deblend:
        image_segmented = deblend_sources(
            image.data,
            image_segmented,
            npixels=5,
            nlevels=32,
            contrast=0.1,
            mode="exponential",
        )
    cat = SourceCatalog(image.data, image_segmented)

    srclist = cat.to_table()
    srclist["X_IMA"] = srclist["xcentroid"].value
    srclist["Y_IMA"] = srclist["ycentroid"].value
    srclist["EXTENT"] = srclist["equivalent_radius"].value / 2

    coords = SkyCoord.from_pixel(srclist["X_IMA"], srclist["Y_IMA"], image.wcs)
    srclist["RA"] = coords.ra
    srclist["DEC"] = coords.dec

    return srclist, image_segmented


def _calc_threshold(image, max_fraction=1 / 50, iter=7):
    threshold = np.zeros_like(image, dtype=float)
    for _ in range(iter):
        segm = detect_sources(image, threshold=threshold, npixels=5)
        cat = SourceCatalog(image, segm)

        threshold = np.zeros_like(image, dtype=float)
        for src in cat:
            threshold[segm.data == src.id] = src.max_value * max_fraction

    return threshold


def extract_counts(
    exposure, srclist, psf_energy, eef, image_segmented=None, mode="aperture_psf", dim=3
):
    if dim < 2 or dim > 3:
        raise ValueError(f"Wrong dimensions: {dim}") 

    if mode == "aperture_psf":
        if dim == 3:
            srclist = counts.extract_aperture_psf_cube(exposure, srclist, psf_energy, eef)
        else:
            srclist = counts.extract_aperture_psf_image(exposure, srclist, psf_energy, eef)

    elif mode == "aperture":
        if dim == 3:
            srclist = counts.extract_aperture_cube(exposure, srclist)
        else:
            srclist = counts.extract_aperture_image(exposure, srclist)

    elif mode == "segmentation":
        srclist = counts.extract_segmentation(exposure, image_segmented, srclist)

    else:
        raise ValueError(f"Unknown photometry mode: {mode}")

    if dim == 2:
        srclist = _lightcurve_to_counts(srclist)

    return srclist


def _lightcurve_to_counts(srclist):
    lc = srclist["LC"].data
    srclist["SRC_COUNTS"] = lc[:, 0, 0]
    srclist["BKG_COUNTS"] = lc[:, 0, 1]
    srclist["DET_ML"] = counts.poisson_probability(
        srclist["SRC_COUNTS"], srclist["BKG_COUNTS"], log=True
    )
    srclist.remove_column("LC")

    return srclist


def time_binning(srclist, sigma_level=3, edges=None):
    srclist["SRC_COUNTS"] = -1.0
    srclist["BKG_COUNTS"] = -1.0
    srclist["DET_ML"] = -1.0
    srclist["OPTFRAMES"] = -1
    
    lc = srclist["LC"].data
    lc_bb = []
    lc_bb_max_length = 0

    nsrc = len(srclist)
    for src_num in range(nsrc):
        lc_bb_src, src, bkg, logp, f = timebins.optimal(lc[src_num, :, :], sigma_level)
        srclist["SRC_COUNTS"][src_num] = src
        srclist["BKG_COUNTS"][src_num] = bkg
        srclist["DET_ML"][src_num] = logp
        srclist["OPTFRAMES"][src_num] = f

        lc_bb.append(_add_time_bb_edges(lc_bb_src, edges))
        if lc_bb_src.shape[0] > lc_bb_max_length:
            lc_bb_max_length = lc_bb_src.shape[0]

    srclist = _add_lc_bb_column(srclist, lc_bb, lc_bb_max_length)

    return srclist


def _add_time_bb_edges(lc, edges):
    lc_updated = np.zeros((lc.shape[0], lc.shape[1] + 2))
    lc_updated[:, 2:] = lc

    idx_start = 0
    for i, row in enumerate(lc):
        idx_end = idx_start + int(row[0])
        lc_updated[i, 0] = edges[idx_start]
        lc_updated[i, 1] = edges[idx_end]

        idx_start = idx_end

    return lc_updated


def _add_lc_bb_column(srclist, lc_bb, max_length):
    lc_bb_reg = np.full([len(srclist), max_length, lc_bb[0].shape[1]], np.nan)
    
    for i, row in enumerate(lc_bb):
        lc_bb_reg[i, :row.shape[0], :] = row

    idx = srclist.colnames.index("LC") + 1

    try:
        srclist.add_column(lc_bb_reg, name="LC_BB", index=idx)
    
    except Exception:
        srclist.replace_column("LC_BB", lc_bb_reg)

    return srclist


def calc_fluxes(srclist, expmap, ecf=7.3866, eef=70, zsize=1):
    ecf = ecf * 1e11  # counts cm2 / erg
    eef = eef / 100

    if zsize > 1:
        fraction_good_frames = _get_good_frames(srclist["OPTFRAMES"], zsize)
    else:
        fraction_good_frames = 1
    
    exptime = _get_exposure_time(srclist, expmap) * fraction_good_frames

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        srclist["FLUX"] = (
            srclist["SRC_COUNTS"] - srclist["BKG_COUNTS"]
        ) / exptime / ecf / eef << u.erg / u.s / u.cm**2

    srclist.meta["EEF"] = eef
    srclist.meta["ECF"] = ecf

    return srclist


def _get_good_frames(optframes, zsize):
    return np.array([np.binary_repr(f).count("1") for f in optframes]) / zsize


def _get_exposure_time(srclist, expmap):
    xpix = srclist["X_IMA"].astype(int)
    ypix = srclist["Y_IMA"].astype(int)

    return expmap[ypix, xpix]


def final_catalog(srclist, photomode, dim=3):
    keep_columns = [
        "X_IMA",
        "Y_IMA",
        "RA",
        "DEC",
        "SRC_COUNTS",
        "BKG_COUNTS",
        "DET_ML",
        "FLUX",
    ]

    if photomode == "aperture_psf":
        keep_columns += ["PSF_a", "PSF_b", "PSF_pa"]

    if dim == 3:
        keep_columns += ["LC", "LC_BB", "OPTFRAMES"]

    srclist.keep_columns(keep_columns)

    return srclist


def emldetect(exposure, overwrite=False, **kwargs):
    if not overwrite:
        logger.warn("Loading previous emldetect run.")
        try:
            return _load_emldetect_catalogue(exposure, **kwargs)

        except FileNotFoundError:
            logger.warn("No emldetect catalogue found!")

    logger.info("Running SAS emldetect...")
    xmmsas.emldetect(
        exposure.files.evt, 
        exposure.files.att, 
        emin=exposure.eband.min, 
        emax=exposure.eband.max, 
        **kwargs
    )

    return _load_emldetect_catalogue(exposure, **kwargs)


def _load_emldetect_catalogue(exposure, likemin=10, **kwargs):
    # Reads the results of a previous run of emldetect
    parent, prefix, suffix = exposure.files.get_path_parts()
    srclist_path = parent.joinpath(f"{prefix}-summarylist_DETML{likemin}{suffix}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyWarning)
        return Table.read(srclist_path)


def simput(exposure):
    logger.warn("This method just loads the input catalogue for SIXTE simulated data.")
    try:
        return _load_simput_catalogue(exposure)

    except FileNotFoundError:
        raise ValueError(f"Exposure {exposure.obsid} is not a simulation.")


def _load_simput_catalogue(exposure):
    parent, prefix, _ = exposure.files.get_path_parts()
    srclist_path = parent.joinpath(f"{prefix}_sources.simput")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyWarning)

        try:
            simputcat = Table.read(srclist_path, hdu=1)

        except FileNotFoundError:
            srclist_path = parent.joinpath(f"{prefix}_catalogue.simput")
            simputcat = Table.read(srclist_path, hdu=1)

    return simputcat
