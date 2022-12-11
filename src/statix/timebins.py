from itertools import count

import numpy as np
from astropy.stats import bayesian_blocks
from scipy.special import erfc

from .counts import poisson_probability


def optimal(lc, sigma_level):
    threshold = erfc(sigma_level/np.sqrt(2))

    zedges_lo, zedges_hi, lc_bb = _optimal_binning(lc, threshold)
    src_counts, bkg_counts, frames_bitflag = _extract_counts(zedges_lo, zedges_hi, lc)
    log_prob = poisson_probability(src_counts, bkg_counts, log=True)

    return lc_bb, src_counts, bkg_counts, log_prob, frames_bitflag


def _optimal_binning(lc, threshold, p0=0.1):
    nz = len(lc)
    z = np.arange(nz)

    bb_zedges = bayesian_blocks(z, lc[:, 0] + 1, fitness="events", p0=p0)
    lc_bb = np.zeros((len(bb_zedges) - 1, 3))

    for i, tmin, tmax in zip(count(), bb_zedges[:-1], bb_zedges[1:]):
        mask = np.logical_and(z >= tmin, z <= tmax)
        lc_bb[i, 0] = len(lc[mask, 0])
        lc_bb[i, 1] = lc[mask, 0].sum()
        lc_bb[i, 2] = lc[mask, 1].sum()

    significant_bins = poisson_probability(lc_bb[:, 1], lc_bb[:, 2]) < threshold

    zedges_lo = bb_zedges[:-1][significant_bins]
    zedges_hi = bb_zedges[1:][significant_bins]

    return zedges_lo, zedges_hi, lc_bb


def _extract_counts(zedges_lo, zedges_hi, lc):
    nz = len(lc)
    src_counts, bkg_counts, bitflag = 0, 0, 0

    for zframe in range(nz):
        for lo, hi in zip(zedges_lo, zedges_hi):
            if zframe >= lo and zframe <= hi:
                src_counts += lc[zframe, 0]
                bkg_counts += lc[zframe, 1]
                bitflag += 2**zframe

    return src_counts, bkg_counts, bitflag
