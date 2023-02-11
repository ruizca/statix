"""
Inpainting of images using morphological component analysis (MCA).
See Elad et al. 2005 (https://doi.org/10.1016/j.acha.2005.03.005) for details.

This code follows the Matlab algorithm presented here (MCALabWithUtilities.tgz)
https://fadili.users.greyc.fr/demos/WaveRestore/downloads/mcalab/Download.html

@author: A.Georgakakis & A.Ruiz
"""
import logging
import warnings

import numpy as np

from . import mca_transforms as mcat

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def inpaint(
    image,
    kernel=None,
    gamma=None,
    wavelet="Haar",
    level=0,
    nblocks=16,
    linear=False,
    nonegative=False,
    niters=79,
):
    """
    Inpainting algorithm based on Morphological Component Analysis (MCA). 
    The algorithm fills the NaN pixels contained in `image`.

    Parameters
    ----------
    image : 2D ndarray
        Image to be inpainted. Bad pixels of the image must be NaNs.
    kernel : Kernel2D or None, optional
        Astropy convolution Kernel. If not None, the input image is 
        smoothed using this kernel, otherwise it is processed as it is.
        This step is useful for Poisson images with low background.
        By default None.
    gamma : float or None, optional
        Factor for the total variation regularization step. If None, this 
        step is skipped. By default None.
    wavelet : str, optional
        Name of the wavelet for calculating the geometry layer. See the
        pywavelets package for the available options. By default "Haar"
    level : int, optional
        Maximum wavelet scale, by default 0.
    nblocks : int, optional
        Number of blocks in which the image is divided to calculate 
        the texture layer. This number must be a perfect square. By default 16.
    linear : bool, optional
        Use a linear step for calculating the succesive threshold levels, 
        instead of a geometric step. By default False.
    nonegative : bool, optional
        Impose a non-negativity criteria in the different layers, by default False.
    niters : int, optional
        Number of iterations for the algorithm, by default 79.

    Returns
    -------
    ndarray
        Inpainted image
    """
    image_padded = _pad_image_with_nans(image, level)
    _check_nblocks(nblocks, image_padded.shape)

    mask = _bad_pixels_mask(image_padded)
    image_padded = _image_nans_to_zeros(image_padded, mask)

    if kernel is not None:
        image_padded = mcat.convolution(image_padded, mask, kernel)

    level = _set_level(image_padded, level)
    layers = _set_initial_layers(image_padded)
    coeff = _first_pass_coefficients(image_padded, level, wavelet, nblocks)
    thresholds = _calc_thresholds(coeff, niters, linear)

    #for thresh in tqdm(thresholds, leave=False):
    for thresh in thresholds:
        image_resid = _residual_image(image_padded, mask, layers)
        layers[:, :, 0] = _geometry_layer(
            image_resid, layers[:, :, 0], level, wavelet, thresh, nonegative, gamma,
        )
        layers[:, :, 1] = _texture_layer(image_resid, layers[:, :, 1], nblocks, thresh)

    image_inpainted = _final_inpainted_image(layers)

    return _recover_original_shape(image_inpainted, image.shape)


def _pad_image_with_nans(image, level):
    # For using swt as implemented in pywavelets, the
    # image shape has to be a multiple of 2**level
    n, m = image.shape
    padding_width_x = _padding_size(n, level)
    padding_width_y = _padding_size(m, level)

    logger.debug(f"Padded image with size {n + padding_width_x}x{m + padding_width_y}")

    return np.pad(
        image,
        [(0, padding_width_x), (0, padding_width_y)],
        mode="constant",
        constant_values=np.nan,
    )


def _padding_size(size, level):
    return int(np.ceil(size / 2 ** level) * 2 ** level - size)


def _check_nblocks(nblocks, shape):
    _nblocks_is_perfect_square(nblocks)
    _image_can_be_splited_in_nblocks(nblocks, shape)


def _nblocks_is_perfect_square(nblocks):
    # nblocks should be a perfect square, so the image
    # can be divide in √nblocks x √nblocks
    decimal_part = np.sqrt(nblocks) - np.fix(np.sqrt(nblocks))

    if decimal_part:
        raise ValueError(f"nblocks ({nblocks}) must be perfect square!")


def _image_can_be_splited_in_nblocks(nblocks, shape):
    if shape[0] % np.sqrt(nblocks) or shape[1] % np.sqrt(nblocks):
        raise ValueError(
            f"The padded image {shape} cannot be divided in {nblocks} blocks of equal size."
        )


def _bad_pixels_mask(image):
    # We assume that all nan pixels are bad
    mask = np.ones_like(image)
    mask[np.isnan(image)] = 0

    return mask


def _image_nans_to_zeros(image, mask):
    image[mask == 0] = 0

    return image


def _set_level(x, level):
    max_level = int(np.log2(x.shape[0]))

    if level > max_level or level <= 0:
        warnings.warn(
            f"Selected level ({level}) not valid, using maximum level ({max_level})."
        )
        level = max_level

    return level


def _set_initial_layers(image):
    # The algorithm works by definning two layers
    # in the image: texture and geometry (aka cartoon)
    n, m = image.shape

    if n != m:
        raise ValueError("Padded image is not square!")

    layers = np.zeros((n, n, 2))
    layers[:, :, 0] = image

    return layers


def _first_pass_coefficients(image, level, wavelet, nblocks):
    return [mcat.wavelet(image, wavelet, level), mcat.dct2d_local(image, nblocks)]


def _initial_threshold(allcoeffs):
    """
    Calculate the starting threshold, which is the minimum of 
    maximal coefficients of the image in each dictionary.
    """
    buf = np.zeros(len(allcoeffs))

    for i, coeffs in enumerate(allcoeffs):
        tmp = np.concatenate([c.flatten() for c in coeffs[1:]])
        buf[i] = np.max(np.absolute(tmp))

    return buf.min()


def _calc_thresholds(coeff, niters, linear):
    deltamax = _initial_threshold(coeff)  # / 10
    deltamin = deltamax / (niters + 1)
    logger.debug(f"Initial threshold: {deltamax:.03f}; Final threshold: {deltamin:.03f}")

    if linear:
        thresholds = _linear_thresholds(deltamin, deltamax, niters)
    else:
        thresholds = _geom_thresholds(deltamin, deltamax, niters)

    return thresholds


def _linear_thresholds(deltamin, deltamax, niters):
    factor = (deltamax - deltamin) / niters
    return [deltamax - i * factor for i in range(niters)]


def _geom_thresholds(deltamin, deltamax, niters):
    factor = (deltamin / deltamax) ** (1 / niters)
    return [deltamax * factor ** i for i in range(niters)]


def _residual_image(image, mask, layers):
    return image - mask * np.sum(layers, axis=2)


def _geometry_layer(residual, layer, level, wavelet, thresh, nonegative, gamma):
    coeff = mcat.wavelet(residual + layer, wavelet, level)
    coeff = mcat.soft_thresholding(coeff, thresh)

    if nonegative:
        coeff = _non_negativity_constraint(coeff, wavelet, level)

    layer_new = mcat.iwavelet(coeff, wavelet)

    if gamma is not None:
        layer_new = _total_variation_regularization(layer_new, gamma)

    return layer_new


def _non_negativity_constraint(coeff, wavelet=None, level=None):
    # level is nblocks if coeff is the local DCT
    if wavelet is not None:
        temp = mcat.iwavelet(coeff, wavelet)
        temp = np.maximum(0.0, -temp)
        newcoeff = mcat.wavelet(temp, wavelet, level)
    else:
        temp = mcat.idct2d_local(coeff, level)
        temp = np.maximum(0.0, -temp)
        newcoeff = mcat.dct2d_local(temp, level)

    return [coeff[0]] + [c + nc for c, nc in zip(coeff[1:], newcoeff[1:])]


def _total_variation_regularization(x, gamma):
    coeffs = mcat.wavelet(x, wavelet="Haar", level=1, norm=False)
    coeffs = mcat.soft_thresholding(coeffs, gamma)

    return mcat.iwavelet(coeffs, wavelet="Haar", norm=False)


def _texture_layer(residual, layer, nblocks, thresh):
    coeff = mcat.dct2d_local(residual + layer, nblocks)
    coeff = mcat.soft_thresholding(coeff, thresh)

    return mcat.idct2d_local(coeff, nblocks)


def _final_inpainted_image(part):
    image = np.sum(part, axis=2)
    image[image < 0] = 0.0

    return image


def _recover_original_shape(image, shape):
    return image[: shape[0], : shape[1]]
