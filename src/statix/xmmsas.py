# -*- coding: utf-8 -*-
import logging
import os
from datetime import datetime
from tempfile import NamedTemporaryFile

import numpy as np
from astropy.io import fits
from astropy.table import Table
from rich.progress import track

import pxsas
from astropy import units as u

from .utils import working_directory, all_logging_disabled

logger = logging.getLogger(__name__)

IMG_XSIZE = 600
IMG_YSIZE = 600


def make_ccf(path, date=None):
    if date is None:
        date = datetime.now()
        date = date.isoformat()

    if not path.exists():
        path.mkdir(parents=True)

    cif_path = path.joinpath("ccf.cif")
    
    pxsas.run(
        "cifbuild",
        calindexset=str(cif_path),
        withobservationdate=True,
        observationdate=date,
    )
    return cif_path


def make_image(evl_path, detector="PN", emin=500, emax=2000, flag=0):
    imageset = _set_imageset(evl_path)
    expression = _set_image_expression(detector, emin, emax, flag)

    _extract_image(evl_path, imageset, expression)

    return imageset


def make_cube(
    evl_path, detector="PN", emin=500, emax=2000, flag=0, zsize=32, gti_path=None
):
    cubeset = _set_cubeset(evl_path)
    cube = _set_cube(evl_path, zsize)

    time_edges = _set_time_edges(evl_path, zsize, gti_path)
    expression_image = _set_image_expression(detector, emin, emax, flag)

    for zidx in track(range(zsize), description="Extracting 3D cube..."):
        expression_time = _set_time_expression(zidx, time_edges)
        expression = f"{expression_image} && {expression_time}"

        zframe = _extract_zframe(evl_path, expression)
        cube[zidx, :, :] = _pad_zframe(cube.shape, zframe)

    _save_cube_to_fits(cube, time_edges, cubeset)

    return cubeset

def make_expmap(
    evl_path,
    att_path,
    emin=500,
    emax=2000,
):
    set_sas_ccf(evl_path.parent)

    imageset = _set_imageset(evl_path)
    expimageset = _set_expmapset(evl_path)

    with working_directory(evl_path.parent):
        pxsas.run(
            "eexpmap",
            eventset=evl_path,
            attitudeset=att_path,
            expimageset=expimageset,
            imageset=imageset,
            pimin=emin,
            pimax=emax,
            withdetcoords="no",
            withvignetting="yes",
            usefastpixelization="no",
            usedlimap="no",
            attrebin=4,
        )

    return expimageset

def emldetect(
    evl_path,
    att_path,
    detector="PN",
    likemin=10,
    emin=500,
    emax=2000,
    ecf=7.3866,
    bkg_path=None,
):
    set_sas_ccf(evl_path.parent)

    with working_directory(evl_path.parent):
        make_image(evl_path, detector, emin=emin, emax=emax)

        if bkg_path is None:
            _edetect_chain(evl_path, att_path, emin, emax, ecf, likemin)
            _srcmatch(evl_path, likemin)
        else:
            # This only works if a previous run of _edetect_chain has been done.
            _edetect_custom_bkg(evl_path, bkg_path, emin, emax, ecf, likemin)
            _srcmatch_bkg(evl_path, likemin)


def _set_imageset(evl_path):
    return _set_fileset(evl_path, "img")


def _set_cubeset(evl_path):
    return _set_fileset(evl_path, "cube")


def _set_expmapset(evl_path):
    return _set_fileset(evl_path, "imgexp")


def _set_maskset(evl_path):
    return _set_fileset(evl_path, "imgmask")


def _set_bkgimageset(evl_path):
    return _set_fileset(evl_path, "imgbkg")


def _set_fileset(path, strid):
    parent = path.parent
    prefix = path.stem.split("-")[0]
    suffix = path.suffix

    return parent.joinpath(f"{prefix}-{strid}{suffix}")


def _set_image_expression(detector, emin, emax, flag):
    if detector == "PN":
        # detector_flag = "#XMMEA_EP"
        detector_flag = "(FLAG & 0xcfa0000)==0"
        pattern = "[0:4]"
    else:
        detector_flag = "(FLAG & 0x766b0808)==0"
        pattern = "[0:12]"

    expression = (
        f"{detector_flag} && "
        f"(PI in [{emin}:{emax}]) && "
        f"(PATTERN in {pattern}) && "
        f"(FLAG=={flag})"
    )
    return expression


def _set_cube(evl_path, zsize):
    imageset = _set_imageset(evl_path)
    header = fits.getheader(imageset)
    xsize, ysize = header["NAXIS1"], header["NAXIS2"]

    return np.zeros((zsize, ysize, xsize))


def _extract_image(evl_path, imageset, expression):
    pxsas.run(
        "evselect",
        table=f"{evl_path.as_posix()}:EVENTS",
        imageset=imageset,
        expression=expression,
        xcolumn="X",
        ycolumn="Y",
        ximagesize=IMG_XSIZE,
        yimagesize=IMG_YSIZE,
        withimagedatatype="true",
        imagedatatype="Real32",
        squarepixels="true",
        imagebinning="imageSize",
        withimageset="true",
        writedss="true",
        keepfilteroutput="false",
        updateexposure="true",
    )


def _extract_zframe(evl_path, expression):
    with NamedTemporaryFile() as image_file:
        with all_logging_disabled(highest_level=logging.WARNING):
            _extract_image(evl_path, image_file.name, expression)

        return fits.getdata(image_file.name)


def _pad_zframe(shape, frame):
    padding_width_x = shape[2] - frame.shape[0]
    padding_width_y = shape[1] - frame.shape[1]

    frame_padded = np.pad(
        frame, [(0, padding_width_y), (0, padding_width_x)], mode="constant"
    )

    return frame_padded


def _set_time_edges(evl_path, zsize, gti_path=None, hdu=1):
    if gti_path is False:
        logger.info("Not using GTI information.")
        edges = _time_edges_nogti(evl_path, zsize)
    else:
        edges = _time_edges_gti(evl_path, gti_path, hdu, zsize)

    return edges


def _time_edges_nogti(evl_path, zsize):
    header = fits.getheader(evl_path, 1)
    tmin = header["TSTART"]
    tmax = header["TSTOP"]

    return np.linspace(tmin, tmax, num=zsize + 1)


def _time_edges_gti(evl_path, gti_path, hdu, zsize):
    if gti_path is not None:
        logger.info("Using GTIs defined in external file.")
        gti = _read_external_gti(gti_path, hdu)
    else:
        logger.info("Using GTIs defined in the event list.")
        gti = _read_internal_gti(evl_path)

    duration = np.sum(gti["STOP"] - gti["START"])
    dt_frame_target = duration / zsize

    # INITIALISE: THE STARTING POINT IS THE FIRST
    # GTI INTERVAL
    i = 0
    t0, t1 = gti[i]["START"], gti[i]["STOP"]
    dt_frame_remaining = dt_frame_target

    edges = [t0]
    while duration > 0:
        # GIVEN a time interval [t0, t1] and
        # a desired duration dt_frame_target
        # estimate how much frwrd time can one move
        # without exceeding t1 (ie remain in the time inerval)
        #
        # It returns the new fwrd point in time (t0)
        # If the desired deltat is larger than the
        # interval [t0, t1] then only part of the desired
        # duration can be achieved. The returned parameter
        # deltat is then the left over time required to
        # achieve the target deltat.
        # It also returns the elapsed time for the
        # frwrd move in time.
        t0, dt_frame_remaining, t_elapsed = _fwrd(t0, t1, dt_frame_remaining)
        duration -= t_elapsed

        if t0 == -1:
            # This is the case where the desired deltat is larger than the
            # intervalq [t0, t1]. In this case only part of the target DT_FRAME
            # can be achieved. The remaining has to completed from the next GTI
            # interval
            i += 1
            t0, t1 = gti[i]["START"], gti[i]["STOP"]
        else:
            # This is the case where the target DT_FRAME has been achieved.
            dt_frame_remaining = dt_frame_target
            edges.append(t0)

    return np.array(edges)


def _read_external_gti(gti_path, hdu):
    return Table.read(gti_path, hdu=hdu)


def _read_internal_gti(evl_path):
    gti_index = []
    with fits.open(evl_path) as hdul:
        for i, h in enumerate(hdul):
            if h.name.find("GTI") == 0:
                gti_index = i
                break

        gti = Table(hdul[gti_index].data)

    gti["STOP"] = gti["STOP"] - gti["START"][0] << u.s
    gti["START"] = gti["START"] - gti["START"][0] << u.s

    return gti


def _fwrd(t0, t1, dt_target):
    """
    Given an interval gti with start time 't0' and end time 't1', 
    and a desired duration 'dt_target', estimate the next time stamp, 
    't_edge', so that 't_edge = t0 + deltat'.

    If the interval duration 'dt' is larger than 'dt_target' then 
    move forward in time by 't0 + dt_target' and return the new
    time stap 't = t0 + dt_target'. In this case the elapsed time is 
    'dt_target'. The remaining time required to achieve 'dt_target' 
    is then 'dt_remaining = 0'. It represents the left over time to 
    complete the desired target duration

    If the interval duration 'dt' is smaller than 'dt_target' then 
    move forward in time to t1. The new time stamp is then -1. In 
    this case the  elapsed time is 't_elapsed = dt' (i.e. the full 
    interval). In this case the desired duration has not been 
    completed and remains a left over time duration 
    'dt_remaining = dt_target - dt'
    """
    dt = t1 - t0

    if dt < dt_target:
        t_edge, t_elapsed = -1, dt
        dt_remaining = dt_target - dt
    else:
        t_edge, t_elapsed = t0 + dt_target, dt_target
        dt_remaining = 0

    return t_edge, dt_remaining, t_elapsed


def _set_time_expression(idx, edges):
    t0 = edges[idx]
    t1 = edges[idx + 1]

    return f"(TIME >= {t0:.04f} && TIME < {t1:.04f})"


def _save_cube_to_fits(data, time_edges, output_path):
    primary_hdu = fits.PrimaryHDU(data)

    c1 = fits.Column(name="TIME_EDGES", array=time_edges, format="D")
    table_hdu = fits.BinTableHDU.from_columns([c1])
    table_hdu.header["EXTNAME"] = "TIMEBINS"

    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(output_path, overwrite=True)


def save_to_fits(data, output_path):
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(output_path, overwrite=True)


def set_sas_odf(path):
    # `path` is either the ODF folder, or the PPS folder
    # where a summary file for the ODF has been generated
    path_absolute = path.resolve()
    try:
        sas_odf = next(path_absolute.glob("*SUM.SAS"))
    except StopIteration:
        sas_odf = path_absolute

    os.environ["SAS_ODF"] = sas_odf.as_posix()


def set_sas_ccf(path, ccf_file="ccf.cif"):
    path_absolute = path.resolve() / ccf_file
    os.environ["SAS_CCF"] = path_absolute.as_posix()


def event_list_from_odf(odf_path, pps_path, detector="PN"):
    set_sas_odf(odf_path)
    set_sas_ccf(pps_path)

    with working_directory(pps_path):
        pxsas.run("cifbuild")
        pxsas.run("odfingest")
        set_sas_odf(pps_path)

        if detector == "PN":
            pxsas.run("epproc")
        else:
            raise NotImplementedError

    return _find_event_lists(pps_path, detector)


def _find_event_lists(pps_path, detector):
    evl = []
    for f in pps_path.glob(f"*{detector}_*Evts.ds"):
        datamode = fits.getval(f, "DATAMODE")

        if datamode.find("TIMING") == 0:
            logger.warn("The observations %s is in Timing mode. Cannot be processed.", f)
        else:
            evl.append(f)

    if not len(evl):
        logger.warn("No valids event lists found!")
        evl = None

    return evl


def filter_event_list(evl_path, pps_path):
    with working_directory(pps_path):
        pxsas.run("espfilt", eventset=evl_path.as_posix())  ## SAS 19
        # pxsas.run("espfilt", eventfile=evl_path.as_posix(), method="ratio", keepinterfiles="yes")  ## SAS 20


def _set_eboxl_list(evl_path):
    return _set_fileset(evl_path, "eboxlist_l")


def _set_eboxm_list(evl_path):
    return _set_fileset(evl_path, "eboxlist_m")


def _set_eboxm_list_bkg(evl_path):
    return _set_fileset(evl_path, "bkg_eboxlist_m")


def _set_eml_list(evl_path, likemin):
    return _set_fileset(evl_path, f"emllist_DETML{likemin}")


def _set_eml_list_bkg(evl_path, likemin):
    return _set_fileset(evl_path, f"bkg_emllist_DETML{likemin}")


def _edetect_chain(evl_path, att_path, emin=500, emax=2000, ecf=7.3866, likemin=10):
    imageset = _set_imageset(evl_path)
    eboxl_list = _set_eboxl_list(evl_path)
    eboxm_list = _set_eboxm_list(evl_path)
    eml_list = _set_eml_list(evl_path, likemin)

    # TODO: I'm using the ecf value for 0.2-2 keV band.
    # At this point I don't care about the actual flux values
    pxsas.run(
        "edetect_chain",
        imagesets=imageset,
        eventsets=evl_path,
        attitudeset=att_path,
        pimin=emin,
        pimax=emax,
        ecf=ecf,
        eboxl_list=eboxl_list,
        eboxm_list=eboxm_list,
        esp_nsplinenodes=16,
        eml_list=eml_list,
        esen_mlmin=15,
        eboxl_likemin=likemin - 2,
        eboxm_likemin=likemin - 2,
        likemin=likemin,
    )


def _edetect_custom_bkg(
    evl_path, bkgimageset, emin=500, emax=2000, ecf=7.3866, likemin=10
):
    imageset = _set_imageset(evl_path)
    expmapset = _set_expmapset(evl_path)
    maskset = _set_maskset(evl_path)
    eboxm_list = _set_eboxm_list_bkg(evl_path)
    eml_list = _set_eml_list_bkg(evl_path, likemin)

    _eboxdetect(
        imageset,
        expmapset,
        eboxm_list,
        maskset=maskset,
        bkgimageset=bkgimageset,
        likemin=likemin - 2,
        emin=emin,
        emax=emax,
    )
    _emldetect(
        imageset,
        expmapset,
        bkgimageset,
        eboxm_list,
        eml_list,
        maskset=maskset,
        emin=emin,
        emax=emax,
        ecf=ecf,
        likemin=likemin,
        determineerrors="yes",
    )


def _eboxdetect(
    imageset,
    expmapset,
    eboxlist,
    maskset=None,
    bkgimageset=None,
    likemin=8,
    emin=200,
    emax=10000,
):
    if maskset is not None:
        withdetmask = "yes"
    else:
        withdetmask = "no"

    if bkgimageset is not None:
        usemap = "yes"
    else:
        usemap = "no"

    pxsas.run(
        "eboxdetect",
        imagesets=imageset,
        expimagesets=expmapset,
        boxlistset=eboxlist,
        withdetmask=withdetmask,
        detmasksets=maskset,
        usemap=usemap,
        bkgimagesets=bkgimageset,
        likemin=likemin,
        pimin=emin,
        pimax=emax,
    )


def _emldetect(
    imageset,
    expmapset,
    bkgimageset,
    eboxlist,
    emllist,
    maskset=None,
    emin=300,
    emax=12000,
    ecf=2.0,
    likemin=10,
    determineerrors="no",
):
    if maskset is not None:
        withdetmask = "yes"
    else:
        withdetmask = "no"

    pxsas.run(
        "emldetect",
        imagesets=imageset,
        expimagesets=expmapset,
        bkgimagesets=bkgimageset,
        detmasksets=maskset,
        boxlistset=eboxlist,
        mllistset=emllist,
        withdetmask=withdetmask,
        pimin=emin,
        pimax=emax,
        ecf=ecf,
        mlmin=likemin,
        determineerrors=determineerrors,
    )


def _srcmatch(evl_path, likemin):
    inputlistsets = _set_eml_list(evl_path, likemin)
    outputlistset = _set_fileset(evl_path, f"summarylist_DETML{likemin}")

    nsrcs = fits.getval(inputlistsets, "NAXIS2", ext=1)

    if nsrcs:
        pxsas.run(
            "srcmatch",
            inputlistsets=inputlistsets,
            outputlistset=outputlistset,
            htmloutput="/dev/null",
        )
    else:
        logger.warn("No sources detected, summary list is not created.")


def _srcmatch_bkg(evl_path, likemin):
    inputlistsets = _set_eml_list_bkg(evl_path, likemin)
    outputlistset = _set_fileset(evl_path, f"bkg_summarylist_DETML{likemin}")

    nsrcs = fits.getval(inputlistsets, "NAXIS2", ext=1)

    if nsrcs:
        pxsas.run(
            "srcmatch",
            inputlistsets=inputlistsets,
            outputlistset=outputlistset,
            htmloutput="/dev/null",
        )
    else:
        logger.warn("No sources detected, summary list is not created.")
