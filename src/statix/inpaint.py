#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:07:05 2021

@author: ruizca
"""
import logging
import warnings

import numpy as np
from astropy.convolution import interpolate_replace_nans, Kernel2D, Tophat2DKernel
from scipy.ndimage import generic_filter

from . import mca

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def image_fill_ccd_gaps(image, mask2d, method="conv", kernel=None):
    kernel = _set_kernel(kernel)
    image_nans = image.copy()
    image_nans[mask2d < 1] = np.nan

    image_reconstructed = _reconstruct_frame(image_nans, kernel, method)

    image_inpaint = image.copy()
    image_inpaint[mask2d < 1] = image_reconstructed[mask2d < 1]

    return image_inpaint


def cube_fill_ccd_gaps(cube, mask2d, method="conv", kernel=None):
    kernel = _set_kernel(kernel)
    cube_nans = _masked_pixels_to_nan(cube, mask2d)
    cube_inpaint = np.zeros(cube.shape, dtype=float)

    for idx_frame in range(cube.shape[0]):
        cube_inpaint[idx_frame, :, :] = cube[idx_frame, :, :]

        if np.allclose(0, cube_nans[idx_frame, mask2d > 0]):
            warnings.warn(f"All zeros in frame {idx_frame}.")
            continue

        frame_reconstructed = _reconstruct_frame(
            cube_nans[idx_frame, :, :], kernel, method
        )
        cube_inpaint[idx_frame, mask2d < 1] = frame_reconstructed[mask2d < 1]

    return cube_inpaint


def _set_kernel(kernel):
    if not kernel:
        kernel = Tophat2DKernel(10)

    if not isinstance(kernel, Kernel2D):
        raise ValueError("Wrong format for kernel")

    return kernel


def _masked_pixels_to_nan(cube, mask):
    mask3d = np.repeat(mask[np.newaxis, :, :], cube.shape[0], axis=0)

    cube_nans = cube.copy()
    cube_nans[mask3d < 1] = np.nan

    return cube_nans


def _reconstruct_frame(frame, kernel, method):
    if method == "nanmean":
        frame_reconstructed = _nanmean_filter(frame, kernel)

    elif method == "conv":
        frame_reconstructed = _astropy_convolution(frame, kernel)

    elif method == "mca":
        frame_reconstructed = _mca(frame, kernel)

    else:
        raise ValueError(f"Unknown method: {method}")

    return _poisson(frame_reconstructed)


def _nanmean_filter(frame, kernel):
    footprint = np.zeros_like(kernel.array)
    footprint[kernel.array > 0] = 1

    frame_reconstructed = generic_filter(
        frame, np.nanmean, footprint=footprint, mode="wrap"
    )
    frame_reconstructed[np.isnan(frame_reconstructed)] = 0

    return frame_reconstructed


def _astropy_convolution(frame, kernel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame_reconstructed = interpolate_replace_nans(frame, kernel)

    frame_reconstructed[np.isnan(frame_reconstructed)] = 0

    return frame_reconstructed


def _mca(frame, kernel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            
        
        return mca.inpaint(
            frame,
            kernel,
            gamma=0.5,
            wavelet="db8",
            level=3,
            nblocks=64,
            linear=False,
            nonegative=False,
        )


def _poisson(frame):
    try:
        pframe = rng.poisson(frame)

    except ValueError:
        logger.info("Using gaussian approximation for poisson randomization")
        mask_big = frame > 1000
        pframe = np.zeros_like(frame)
        pframe[mask_big] = np.round(
            rng.normal(loc=frame[mask_big], scale=np.sqrt(frame[mask_big]))
        )
        pframe[~mask_big] = rng.poisson(frame[~mask_big])

    return pframe

