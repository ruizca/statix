import numpy as np
from pywt import swt2, iswt2, threshold
from scipy.fft import dctn, idctn
from scipy.signal import convolve


def dct2d_local(x, nblocks):
    d = np.sqrt(nblocks)
    c = [[dctn(hs, norm="ortho") for hs in np.hsplit(vs, d)] for vs in np.vsplit(x, d)]

    return [None, np.block(c)]


def idct2d_local(coeffs, nblocks):
    c = coeffs[1]
    d = np.sqrt(nblocks)
    x = [[idctn(hs, norm="ortho") for hs in np.hsplit(vs, d)] for vs in np.vsplit(c, d)]

    return np.block(x)


def wavelet(x, wavelet="db2", level=0, norm=False):
    wt = swt2(x, wavelet=wavelet, level=level, trim_approx=True, norm=norm)

    # We convert all coefficients into a list of 3D arrays
    # The first is the approximation, and is size 1 in the third axis (n, n, 1)
    # The rest are the details (cH, cV, cD), with size 3 in the third axis (n, n, 3)
    approx = [np.expand_dims(wt[0], axis=2)]
    details = [np.dstack(c) for c in wt[1:]]

    return approx + details


def iwavelet(wt, wavelet="db2", norm=False):
    approx = [np.squeeze(wt[0], axis=2)]
    details = [[np.squeeze(d, axis=2) for d in np.dsplit(c, 3)] for c in wt[1:]]

    return iswt2(approx + details, wavelet=wavelet, norm=norm)


def soft_thresholding(coeff, thresh):
    return [coeff[0]] + [threshold(c, thresh, mode="soft") for c in coeff[1:]]


def convolution(x, mask, kernel):
    k = _calc_img_kernel(kernel)
    area = (_calc_img_area(mask, k)).astype(float)
    counts = (_calc_img_counts(x, k, mask)).astype(float)

    mask_zeros = mask > 0
    counts[mask_zeros] = counts[mask_zeros]/area[mask_zeros]

    return counts

def _calc_img_kernel(kernel):
    data = kernel.array.copy()
    data[data > 0] = 1

    return data

def _calc_img_area(mask, kernel):
    area = convolve(mask, kernel, mode="same")
    area[mask == 0] = 0
    return area

def _calc_img_counts(data, kernel, mask):
    counts =  convolve(data, kernel, mode="same")
    counts[mask == 0] = 0
    return counts
    