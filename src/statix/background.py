import logging
import warnings

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable
from regions import PixCoord, CirclePixelRegion, CircleAnnulusPixelRegion
from scipy.signal import oaconvolve

from .source_detection import peak_detection

logger = logging.getLogger(__name__)


class BkgArray:
    def _load_data(self, filename):
        return fits.getdata(filename)

    def _identify_source_regions(self, image, sigma_level, radius_factor, inpaint=False):
        if inpaint:
            image = image.fill_gaps(self.mask.data, method="mca")
        
        image_denoised = self._msvst_denoise_image(image, sigma_level)
        candidates = self._find_candidates(image_denoised)
        src_regions = self._set_src_regions(candidates, radius_factor)

        return src_regions

    def _msvst_denoise_image(self, image, sigma_level):
        return image.denoise(
            coupled=True,
            threshold_mode=1,
            border_mode=2,
            max_scalexy=3,
            min_scalexy=1,
            sigma_level=sigma_level,
            use_non_default_filter=True,
            kill_last=True,
            detpos=True,
            verbose=False,
        )

    # def _find_candidates(self, image_denoised):
    #     threshold = np.percentile(image_denoised.data, 95.4)
    #     segm = detect_sources(image_denoised.data, threshold, npixels=5)
    #     cat = source_properties(image_denoised.data, segm)

    #     return cat.to_table()

    def _find_candidates(self, image_denoised):
        cat = peak_detection(image_denoised, self.mask, sigma=6)
        cat["xcentroid"] = cat["x_peak"].astype(float)
        cat["ycentroid"] = cat["y_peak"].astype(float)
        cat["equivalent_radius"] = 5 * np.ones(len(cat)) << u.pix

        return QTable(cat)

    def _set_src_regions(self, candidates, radius_factor):
        src_regions = []
        for src in candidates:
            center = PixCoord(x=src["xcentroid"], y=src["ycentroid"])
            radius = radius_factor * src["equivalent_radius"].value
            src_regions.append(CirclePixelRegion(center=center, radius=radius))

        return src_regions

    def _mask_reg(self, reg, shape=None):
        if shape is None:
            shape = self.data.shape

        mask_reg = reg.to_mask(mode="center")
        mask_reg = mask_reg.to_image(shape)
        mask_reg = np.logical_and(mask_reg > 0, self.mask.data > 0)

        return mask_reg

    def _mask_bkg(self, reg, shape=None, inner_factor=2, outer_factor=5):
        reg_bkg = CircleAnnulusPixelRegion(
            center=reg.center,
            inner_radius=inner_factor * reg.radius,
            outer_radius=outer_factor * reg.radius,
        )
        return self._mask_reg(reg_bkg, shape)

    @staticmethod
    def _convolution_kernel(radius=15, inner_factor=1, outer_factor=2):
        kernel_reg = CircleAnnulusPixelRegion(
            center=PixCoord(x=0, y=0),
            inner_radius=inner_factor * radius,
            outer_radius=outer_factor * radius,
        )
        return kernel_reg.to_mask().data


class BkgImage(BkgArray):
    def __init__(
        self, image, mask=None, sigma_level=3, radius_factor=1.0, convolve=False, inpaint=False
    ):
        self.data = image.data
        # self.mask = self._set_exposure_mask(mask)
        self.mask = mask

        self.src_regions = self._identify_source_regions(
            image, sigma_level, radius_factor
        )
        if self.src_regions:
            self.data = self._empty_source_regions()
            self.data = self._fill_holes_with_bkg_counts()
        else:
            logger.warn("The whole image is background!")

        if convolve:
            self.data = self._convolve()

    def _set_exposure_mask(self, mask):
        if not mask:
            exp_mask = np.ones_like(self.data)
        else:
            exp_mask = mask.data

        return exp_mask

    def _empty_source_regions(self):
        image_with_holes = self.data.copy()
        total_srcmask = np.zeros_like(self.data, dtype=float)

        for reg in self.src_regions:
            mask = reg.to_mask(mode="center")
            total_srcmask += mask.to_image(self.data.shape)

        image_with_holes[total_srcmask > 0] = 0

        return image_with_holes

    def _fill_holes_with_bkg_counts(self):
        image_src_filled = self.data.copy()

        for reg in self.src_regions:
            mask_src = self._mask_reg(reg)
            mask_bkg = self._mask_bkg(reg)

            # Select n values (equal to the number of pixels in the src region)
            # randomly from the values within the bkg region, and assign
            # them to the source region
            image_src_filled[mask_src] = np.random.choice(
                self.data[mask_bkg], self.data[mask_src].size
            )

        return image_src_filled

    def _convolve(self, radius=15, inner_factor=1, outer_factor=5):
        kernel = self._convolution_kernel(
            radius=radius, inner_factor=inner_factor, outer_factor=outer_factor
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            area_corr = oaconvolve(self.mask.data, kernel, mode="same") / kernel.sum()
            data_convolved = oaconvolve(self.data, kernel, mode="same") / area_corr / kernel.sum()

        return self.apply_mask(data_convolved)

    def apply_mask(self, data):
        data[self.mask.data < 1] = 0
        return data


class BkgCube(BkgArray):
    def __init__(
        self, cube, mask=None, sigma_level=3, radius_factor=1.0, convolve=False, inpaint=False
    ):
        self.data = cube.data
        # self.mask = self._set_exposure_mask(mask)
        self.mask = mask

        self.src_regions = self._identify_source_regions(
            cube.time_integrated, sigma_level, radius_factor, inpaint
        )

        if self.src_regions:
            self.data = self._empty_source_regions()
            self.data = self._fill_holes_with_bkg_counts()
        else:
            logger.warn("The whole image is background!")

        if convolve:
            self.data = self._convolve()

    def _set_exposure_mask(self, mask):
        if not mask:
            exp_mask = np.ones(self.data.shape[1:], dtype=int)
        else:
            exp_mask = mask.data

        return exp_mask

    def _empty_source_regions(self):
        mask_shape = self.data.shape[1:]
        cube_with_holes = self.data.copy()
        total_srcmask = np.zeros(mask_shape, dtype=float)

        for reg in self.src_regions:
            mask = reg.to_mask(mode="center")
            total_srcmask += mask.to_image(mask_shape)

        cube_with_holes[:, total_srcmask > 0] = 0

        return cube_with_holes

    def _fill_holes_with_bkg_counts(self):
        cube_src_filled = self.data.copy()
        mask_shape = self.data.shape[1:]

        for reg in self.src_regions:
            mask_src = self._mask_reg(reg, shape=mask_shape)
            mask_bkg = self._mask_bkg(reg, shape=mask_shape)

            for iz in range(self.data.shape[0]):
                # Select n values (equal to the number of pixels in the src region)
                # randomly from the values within the bkg region, and assign
                # them to the source region
                cube_src_filled[iz, mask_src] = np.random.choice(
                    self.data[iz, mask_bkg], self.data[iz, mask_src].size
                )

        return self.apply_mask(cube_src_filled)

    def _convolve(self, radius=15, inner_factor=1, outer_factor=5):
        kernel = self._convolution_kernel(radius, inner_factor, outer_factor)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            area_corr = oaconvolve(self.mask.data, kernel, mode="same") / kernel.sum()
            data_convolved = oaconvolve(
                self.data, kernel[np.newaxis,:,:], mode="same", axes=(1, 2)
            ) / area_corr / kernel.sum() 

        return self.apply_mask(data_convolved)

    def apply_mask(self, data):
        data[:, self.mask.data < 1] = 0
        return data
