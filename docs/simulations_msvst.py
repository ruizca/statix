import logging
from pathlib import Path

from joblib import Parallel, delayed

from statix.exposure import Exposure
from statix.utils import catch_obsid_error, track_joblib

def main():
    logging.getLogger().setLevel(logging.INFO)

    n_jobs = 50
    n_sims = 1000
    simulations_path = Path("/storage/ruizca/pyxsel_data/sixtesims/newgoodbkg")
    photomode = "aperture_psf"

    ids = list(range(n_sims))

    if len(ids) > 1:
        with track_joblib(desc="Detecting sources with STATiX", total=len(ids)):
            Parallel(n_jobs=n_jobs)(
                delayed(detect_sources)(id, simulations_path, photomode) for id in ids
            )
    else:
        detect_sources(ids[0], simulations_path, photomode, logtofile=False)


@catch_obsid_error
def detect_sources(id, simulations_path, photomode="aperture", logtofile=True):
    photomode_tag = set_photomode_tag(photomode)
    msvst_sigma_levels = [3, 4, 5]
    time_sigma_level = 3

    if logtofile:
        logger_path = simulations_path / f"goodbkg_msvst{photomode_tag}_{id:03d}.log"
        set_logger(logger_path)

    exp = set_exposure(id, simulations_path)
    for sigma_level in msvst_sigma_levels:
        logging.info(f"MSVST sigma_level = {sigma_level}")
        srclist_msvst, cube_msvst = exp.detect_sources(
            method="msvst2d1d",
            inpainting=True,
            detmode="peaks",
            photomode=photomode,
            time_sigma_level=time_sigma_level,
            eef=70,
            sigma_level=sigma_level,
            border_mode=2,
            min_scalexy=2,
            max_scalexy=4,
            min_scalez=1,
            max_scalez=4,
        )

        srclist_path = set_srclist_path(exp.files, f"{photomode_tag}_tt{time_sigma_level}", sigma_level)
        srclist_msvst.write(srclist_path.as_posix(), format="fits", overwrite=True)

        cube_msvst_path = set_cube_msvst_path(exp.files, photomode_tag, sigma_level)
        cube_msvst.save_as_fits(cube_msvst_path.as_posix())


def set_photomode_tag(photomode):
    if photomode == "aperture_psf":
        photomode_tag = "_psf"
    else:
        photomode_tag = ""

    return photomode_tag


def set_logger(filename, fmt=None):
    filehandler = logging.FileHandler(filename, "w")

    if fmt is None:
        infofmt = "%(levelname)s:%(asctime)s:%(module)s:%(funcName)s: %(message)s"
        fmt = logging.Formatter(infofmt, datefmt="%I:%M:%S")
    filehandler.setFormatter(fmt)

    # root logger - Good to get it only once.
    logger = logging.getLogger()

    # remove the existing file handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
    logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)

    return logger


def set_exposure(id, simulations_path):
    exposure_path = _set_exposure_path(id, simulations_path)
    evl_path = _set_evl_path(id, exposure_path)

    return Exposure(evl_path)


def _set_exposure_path(id, simulations_path):
    return simulations_path / Path(f"{id:010}")


def _set_evl_path(id, exposure_path):
    return exposure_path / Path(f"P{id:010}PNS001-objevlifilt.FIT")


def set_srclist_path(expfiles, photomode_tag, sigma_level):
    return _set_product_path(expfiles, "srclist_msvst", photomode_tag, sigma_level)


def set_cube_msvst_path(expfiles, photomode_tag, sigma_level):
    return _set_product_path(expfiles, "cube_msvst", photomode_tag, sigma_level)


def _set_product_path(expfiles, tag, photomode_tag, sigma_level):
    parent, prefix, suffix = expfiles.get_path_parts()
    tag = f"{tag}{photomode_tag}_{sigma_level}sig"

    return parent.joinpath(f"{prefix}-{tag}{suffix}")


if __name__ == "__main__":
    main()
