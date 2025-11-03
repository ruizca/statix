# -*- coding: utf-8 -*-
"""
Module handling image and cube data for XMM-Newton.
"""
import logging
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from mocpy import MOC
from msvst import MSVST2D, MSVST2D1D

from . import inpaint

logger = logging.getLogger(__name__)

try:
    from . import xmmsas

except ImportError as e:
    logger.warn(e)
    logger.warn("SAS-related functions not available!!!")


class ImageBase:
    """
    Base class for Image and Cube objects.
    """
    def __init__(self, filename=None, data=None, wcs=None):
        if filename:
            self.data, self.wcs, _ = self._read(filename)
        else:
            self.data = data
            self.wcs = wcs

    def __str__(self):
        return f"{'x'.join(str(s) for s in self.shape)} {self.__class__.__name__}"

    def _read(self, image_file):
        with fits.open(image_file) as hdu:
            wcs = self._set_wcs(hdu)
            data = hdu[0].data
            gti = None

            if data.ndim > 2:
                gti = self._read_gtis(hdu)

        return data, wcs, gti

    @staticmethod
    def _read_gtis(hdu):
        try:
            gti = Table.read(hdu["GTI"])
        
        except KeyError:
            logger.warning("No GTI extension in file.")
            gti = None
    
        return gti

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _set_wcs(hdulist):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FITSFixedWarning)
            wcs = WCS(hdulist[0].header, hdulist)

        return wcs

    def save_as_fits(self, filename):
        save_fits(self.data, filename)


class Image(ImageBase):
    """
    Image class.
    
    Parameters
    ----------
    filename : str, optional 
        Path to a FITS file to read the image from.
    data : ndarray, optional
        2D array with image data.
    wcs : WCS, optional
        WCS object for the image.

    Attributes
    ----------
    data : ndarray
        2D array with image data.
    wcs : WCS
        WCS object for the image.
    shape : tuple
        Shape of the image data array.

    Methods
    -------
    fill_gaps(mask, method="conv", kernel=None): 
        Returns a new Image object with the gaps defined in mask filled with counts.
    denoise(output_file=None, **kwargs): 
        Returns a new Image object denoised using MSVST2D.
    make(event_list_file, **kwargs): 
        Class method to create an Image from an event list file using SAS.
    """
    def fill_gaps(self, mask, method="conv", kernel=None):
        """
        Returns a new Image object with the gaps defined in mask filled with counts.
        """
        image_filled = inpaint.image_fill_ccd_gaps(self.data, mask, method, kernel)

        return Image(data=image_filled, wcs=self.wcs)

    def denoise(self, output_file=None, **kwargs):
        with NamedTemporaryFile(suffix=".fits") as temp:
            self.save_as_fits(temp.name)

            temp_path = Path(temp.name)
            msvst_file = MSVST2D.denoise(temp_path, output_file, **kwargs)
            image_denoised = Image(msvst_file)
            image_denoised.wcs = self.wcs

        if not output_file:
            msvst_file.unlink()

        return image_denoised

    @classmethod
    def make(cls, event_list_file, **kwargs):
        xmmsas.make_image(event_list_file, **kwargs)


class ExpMap(ImageBase):
    """
    Exposure map class.

    Parameters
    ----------
    filename : str, optional
        Path to a FITS file to read the exposure map from.
    data : ndarray, optional
        2D array with exposure map data.
    wcs : WCS, optional
        WCS object for the exposure map.

    Attributes
    ----------
    data : ndarray
        2D array with exposure map data.
    wcs : WCS
        WCS object for the exposure map.
    shape : tuple
        Shape of the exposure map data array.

    Methods
    -------
    make(event_list_file, attitude_file, **kwargs): 
        Class method to create an ExposureMap from event list and attitude files using SAS.
    """
    @classmethod
    def make(cls, event_list_file, attitude_file, **kwargs):
        xmmsas.make_expmap(event_list_file, attitude_file, **kwargs)


class Cube(ImageBase):
    """
    Cube class.

    Parameters
    ----------
    filename : str, optional
        Path to a FITS file to read the cube from.
    data : ndarray, optional
        3D array with cube data.
    wcs : WCS, optional
        WCS object for the cube.
    gti : Table, optional
        Table with Good Time Intervals.

    Attributes
    ----------
    data : ndarray
        3D array with cube data.
    wcs : WCS
        WCS object for the cube.
    gti : Table
        Table with Good Time Intervals.
    shape : tuple
        Shape of the cube data array.
    time_edges : ndarray
        Time edges of the cube frames.
    time_midpoints : ndarray
        Time midpoints of the cube frames.
    time_integrated : Image
        Image object with the time-integrated data.

    Methods
    -------
    fill_gaps(mask, method="conv", kernel=None, filename=None, **kwargs): 
        Returns a new Cube object with the gaps defined in mask filled with counts.
    denoise(output_file=None, **kwargs): 
        Returns a new Cube object denoised using MSVST2D1D.
    make(event_list_file, **kwargs): 
        method to create a Cube from an event list file using SAS.
    """
    def __init__(self, filename=None, data=None, wcs=None, gti=None):
        if filename:
            self.data, self.wcs, self.gti = self._read(filename)
        else:
            self.data = data
            self.wcs = wcs
            self.gti = gti

    def __str__(self):
        return f"{'x'.join(str(s) for s in self.shape)} {self.__class__.__name__}"

    @property
    def time_edges(self):
        frame_idx = np.arange(self.shape[0] + 1)
        return self.wcs.temporal.pixel_to_world_values(frame_idx)

    @property
    def time_midpoints(self):
        tidx = np.arange(self.shape[0]) + 0.5
        return self.wcs.temporal.pixel_to_world_values(tidx)

    @property
    def time_integrated(self):
        return Image(data=self._project_axis(axis=0), wcs=self.wcs.celestial)

    def _project_axis(self, axis):
        return self.data.sum(axis=axis)

    def save_as_fits(self, filename, only_data=False, overwrite=True):
        if only_data:
            save_fits(self.data, filename)
        else:
            primary_hdu = fits.PrimaryHDU(self.data, header=self.wcs.to_header())
            wcs_table = self._wcs_table()
            hdu_list = [primary_hdu, wcs_table]
            
            if self.gti is not None:
                gti_hdu = fits.BinTableHDU(self.gti)
                gti_hdu.header["EXTNAME"] = "GTI"
                hdu_list.append(gti_hdu)

            hdul = fits.HDUList(hdu_list)
            hdul.writeto(filename, overwrite=overwrite)
            
    def _wcs_table(self):
        idx = np.arange(len(self.time_edges), dtype=np.float32)

        wcs_table = Table()
        wcs_table["TimeIndex"] = [idx]
        wcs_table["TimeCoord"] = [self.time_edges]
        wcs_table["TimeCoord"].unit = self.wcs.wcs.cunit[-1]
        wcs_table.meta["EXTNAME"] = "WCS-table"

        return fits.BinTableHDU(wcs_table)


    def fill_gaps(self, mask, method="conv", kernel=None, filename=None, **kwargs):
        """
        Returns a new Cube object with the gaps defined in mask filled with counts.
        """
        cube_filled_data = inpaint.cube_fill_ccd_gaps(self.data, mask, method, kernel, **kwargs)
        cube_filled = Cube(data=cube_filled_data, wcs=self.wcs, gti=self.gti)

        if filename:
            cube_filled.save_as_fits(filename)

        return cube_filled

    def denoise(self, output_file=None, **kwargs):
        with NamedTemporaryFile(suffix=".fits") as temp:
            self.save_as_fits(temp.name, only_data=True)

            temp_path = Path(temp.name)
            msvst_file = MSVST2D1D.denoise(temp_path, output_file, **kwargs)
            
            cube_denoised = Cube(msvst_file)#, wcs=self.wcs)
            cube_denoised.wcs = self.wcs
            cube_denoised.gti = self.gti

        if not output_file:
            msvst_file.unlink()

        return cube_denoised

    @classmethod
    def make(cls, event_list_file, **kwargs):
        xmmsas.make_cube(event_list_file, **kwargs)


class Mask:
    """
    Mask class.
    This class is used to create and apply masks to images and cubes.

    Parameters
    ----------
    expmap : str or ExposureMap 
        Path to SAS generated mask or ExposureMap object.
    fexp : float, optional
        Fraction of the maximum exposure to define the exposure limit for the mask. Default is 0.3.
    fov : bool, optional
        If True, the mask is created based on the field of view (FOV) using MOC. If False, the mask is created based on the
        exposure map values. Default is False.
    is_mask : bool, optional
        If True, the input expmap is already a mask. Default is False.

    Attributes
    ----------
    data : ndarray
        2D array with the mask data.
    shape : tuple
        Shape of the mask data array.
    """
    def __init__(self, expmap, fexp=0.3, fov=False, is_mask=False):
        if is_mask:
            self.data = fits.getdata(expmap)
        else:
            self.data = self._make(expmap, fov, fexp)

    def __call__(self, target):
        if isinstance(target, Image):
            return self._apply_to_image(target)
        elif isinstance(target, Cube):
            return self._apply_to_cube(target)
        else:
            raise ValueError("Mask can only by applied to Image or Cube objects!")

    def _apply_to_image(self, img):
        if img.shape != self.shape:
            raise ValueError("Mask and Image have inconsistent shapes.")

        return Image(data=self.data * img.data, wcs=img.wcs)

    def _apply_to_cube(self, cube):
        if cube.shape[1:] != self.shape:
            raise ValueError("Mask and Cube have inconsistent shapes.")

        masked_cube = np.zeros_like(cube.data)
        for idx_frame in range(cube.shape[0]):
            masked_cube[idx_frame, :, :] = self.data * cube.data[idx_frame, :, :]

        return Cube(data=masked_cube, wcs=cube.wcs)

    @property
    def shape(self):
        return self.data.shape

    def _make(self, expmap, fov=False, fexp=0.3):
        if fov:
            mask = self._fov_mask(expmap, fexp)
        else:
            # expmap = fits.getdata(expmap_file)
            mask = self._expmap_to_mask(expmap.data, np.max(expmap.data) * fexp)

        return mask

    def _fov_mask(self, expmap, fexp):
        # with fits.open(expmap_file) as hdu:
        #     expmap = hdu[0].data
        mask_ccd_gaps = self._expmap_to_mask(expmap.data, np.max(expmap.data) * fexp)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FITSFixedWarning)
            # wcs_image = WCS(hdu[0].header)
            hdu = fits.ImageHDU(expmap.data, expmap.wcs.to_header())
            moc_fov = MOC.from_fits_image(hdu, max_norder=15, mask=mask_ccd_gaps)

        # We add the cells in the moc border, so the narrow gaps are included in
        # the new moc, getting a first approximation for the fov moc. Then we remove
        # the cells in the moc border several times, to mask border effects.
        for _ in range(4):
            moc_fov = moc_fov.add_neighbours()

        for _ in range(6):
            moc_fov = moc_fov.remove_neighbours()

        xrange = np.arange(mask_ccd_gaps.shape[1])
        yrange = np.arange(mask_ccd_gaps.shape[0])
        x_coords, y_coords = np.meshgrid(xrange, yrange)
        ra, dec = expmap.wcs.wcs_pix2world(x_coords, y_coords, 0)

        fov_mask = moc_fov.contains(ra << u.deg, dec << u.deg)
        mask = np.zeros_like(mask_ccd_gaps)
        mask[fov_mask] = 1

        return mask

    @staticmethod
    def _expmap_to_mask(expmap, explimit=1e-5):
        mask = np.zeros(expmap.shape, dtype=int)
        mask[expmap > explimit] = 1

        return mask


def save_fits(data, filename, header=None):
    hdu = fits.PrimaryHDU(data, header)
    hdu.writeto(filename, overwrite=True)
