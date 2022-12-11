# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 10:55:26 2021

@author: ruizca
"""
import logging
from functools import cached_property
from pathlib import Path

from astropy.io import fits

from . import source_detection
from .background import BkgCube, BkgImage
from .camera import Camera, Orientation
from .ebands import Eband
from .image import Cube, ExpMap, Image, Mask

logger = logging.getLogger(__name__)


class Exposure:
    def __init__(self, event_list_file, attitude_file=None, eband="SOFT"):
        self.files = ExposureFiles(event_list_file, attitude_file)
        self.orientation, self.camera, self.obsid, self.expid = self._set_attributes()
        self.eband = Eband[eband]
        self.data = ExposureData(self.files, self.camera.tag, self.eband)

    def _set_attributes(self):
        header = fits.getheader(self.files.evt)

        if header["DATAMODE"] != "IMAGING":
            raise ValueError("This exposure is not in imaging mode.")

        orientation = Orientation(header["RA_NOM"], header["DEC_NOM"], header["PA_PNT"])
        camera = Camera(header["INSTRUME"], orientation)

        return orientation, camera, header["OBS_ID"], header["EXPIDSTR"]

    def __repr__(self) -> str:
        return (
            f"Exp.ID: {self.expid} [{self.camera} camera], Obs.ID: {self.obsid}\n"
            f"Energy band for products: {self.eband.name} "
            f"({self.eband.min/1000}-{self.eband.max/1000} keV)\n"
            f"Nominal pointing: {self.orientation.pointing.to_string('hmsdms')}\n"
            f"PA: {self.orientation.pa}"
        )

    @property
    def image(self):
        return self.data.image

    @property
    def cube(self):
        return self.data.cube

    @property
    def cube_inpaint(self):
        return self.data.cube_inpaint

    @property
    def expmap(self):
        return self.data.expmap

    @property
    def mask(self):
        return self.data.mask

    @property
    def mask_fov(self):
        return self.data.mask_fov

    @property
    def bkgimage(self):
        return self.data.bkgimage

    @property
    def bkgcube(self):
        return self.data.bkgcube

    def detect_sources(self, method="msvst2d1d", **kwargs):
        logger.info(f"Detecting sources using {method} algorithm...")

        try:
            source_detection_algo = getattr(source_detection, method)

        except AttributeError:
            raise ValueError(f"Detection method '{method}' is not defined!!!")

        return source_detection_algo(self, **kwargs)


class ExposureFiles:
    def __init__(self, event_list_file, attitude_file):
        self.evt = Path(event_list_file)
        self.att = self._set_att(attitude_file)
        
        path_parts = self.get_path_parts()
        self.exp = self._set_path("imgexp", *path_parts)
        self.img = self._set_path("img", *path_parts)
        self.cube = self._set_path("cube", *path_parts)
        self.cube_inpaint = self._set_path("cube_inpaint", *path_parts)

    @staticmethod
    def _set_att(attitude_file):
        if attitude_file is None:
            att = None
        else:
            att = Path(attitude_file)

        return att

    def get_path_parts(self):
        # We assume that files follow the default naming convention of SAS
        parent = self.evt.parent
        prefix = self.evt.stem.split("-")[0]
        suffix = self.evt.suffix

        return parent, prefix, suffix

    @staticmethod
    def _set_path(tag, parent, prefix, suffix):
        return parent.joinpath(f"{prefix}-{tag}{suffix}")


class ExposureData:
    def __init__(self, files, camera, eband):
        self.files = files
        self.camera = camera
        self.eband = eband

        self._detector_frame = None
        self._image = None
        self._mask = None
        self._mask_fov = None
        self._bkgimage = None
        self._expmap = None
        self._cube = None
        self._cube_inpaint = None
        self._bkgcube = None

    @cached_property
    def image(self):
        if not self._image:
            self._image = self._set_image()

        return self._image

    def _set_image(self):
        if not self.files.img.exists():
            Image.make(
                self.files.evt, detector=self.camera, emin=self.eband.min, emax=self.eband.max
            )

        return Image(self.files.img)

    @cached_property
    def cube(self):
        if not self._cube:
            self._cube = self._set_cube()

        return self._cube

    def _set_cube(self):
        if not self.files.cube.exists():
            # First we try creating the cube taking into account the GTI 
            # included in the event list, if it fails because they are not
            # included, we just create the cube without considering the GTI
            kwargs = {
                "detector": self.camera, 
                "emin": self.eband.min, 
                "emax": self.eband.max,
            }
            try:
                # We call the image here because the image file has to
                # exists for creating the cube
                _ = self.image
                Cube.make(self.files.evt, **kwargs)
            
            except KeyError:
                logger.warn("No GTI extensions in the event file!")
                Cube.make(self.files.evt, gti_path=False, **kwargs)

        return Cube(self.files.cube)

    @cached_property
    def cube_inpaint(self):
        if not self._cube_inpaint:
            self._cube_inpaint = self._set_cube_inpaint(method="mca")

        return self._cube_inpaint

    def _set_cube_inpaint(self, method):
        try:
            cube = Cube(self.files.cube_inpaint, time_edges=self.cube.time_edges)
            logger.info(f"Existing inpainted cube loaded!")

        except FileNotFoundError:
            logger.info(f"Filling cube gaps ({method})...")
            cube = self.cube.fill_gaps(
                self.mask.data, method, filename=self.files.cube_inpaint
            )

        return cube

    @cached_property
    def expmap(self):
        if not self._expmap:
            self._expmap = self._set_expmap()

        return self._expmap

    def _set_expmap(self):
        if not self.files.exp.exists():
            ExpMap.make(
                self.files.evt, self.files.att, emin=self.eband.min, emax=self.eband.max
            )

        return ExpMap(self.files.exp)

    @cached_property
    def mask(self):
        if not self._mask:
            self._mask = Mask(self.expmap)

        return self._mask

    @cached_property
    def mask_fov(self):
        if not self._mask_fov:
            self._mask_fov = Mask(self.expmap, fov=True)

        return self._mask_fov

    @cached_property
    def bkgimage(self):
        if not self._bkgimage:
            logger.info("Calculating background image")
            self._bkgimage = BkgImage(
                self.image, mask=self.mask, convolve=True, inpaint=True, sigma_level=2, radius_factor=1
            )

        return self._bkgimage

    @cached_property
    def bkgcube(self):
        if not self._bkgcube:
            logger.info("Calculating background cube")
            self._bkgcube = BkgCube(
                self.cube, mask=self.mask, convolve=True, inpaint=True, sigma_level=2, radius_factor=1
            )

        return self._bkgcube
