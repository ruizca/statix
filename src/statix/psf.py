from importlib import resources

from astropy import units as u
from astropy.coordinates import match_coordinates_sky, SkyCoord
from astropy.table import Table


def get_data(det_coords, energy, eef, remove_bad=True):
    # Returns an array for each `det_coord` with the PSF's
    # semimajor axis, semiminor axis and position angle
    if eef not in [60, 70, 80]:
        raise ValueError(f"Unknown EEF value: {eef}")

    psfgrid = grid(det_coords.frame.name, energy, remove_bad)
    psfdata = find_neighbours_in_grid(psfgrid, det_coords)
    psfdata = aperture_parameters(psfdata, eef, det_coords.frame.rotation)

    return psfdata[["psfid", "PSF_a", "PSF_b", "PSF_pa"]]


def grid(detector, energy, remove_bad=True):
    # Open psf table and get PSF params
    ref = resources.files("statix.data").joinpath(f"psf_{detector}_{energy}.fits")
    with resources.as_file(ref) as psfgrid_file:
        psfgrid = Table.read(psfgrid_file)

    psfgrid["psfid"] = range(len(psfgrid))
    psfgrid["DETX"] = psfgrid["DETX"] * 0.05 * u.arcsec
    psfgrid["DETY"] = psfgrid["DETY"] * 0.05 * u.arcsec

    if remove_bad:
        # remove entries in the psf table with bad values
        good_mask = psfgrid["EEF60"].data[:, 0] > 0
        psfgrid = psfgrid[good_mask]

    return psfgrid


def find_neighbours_in_grid(psfgrid, det_coords):
    # find the nearest point in the psf table for each source
    psfgrid_coords = SkyCoord(psfgrid["DETX"], psfgrid["DETY"], frame=det_coords.frame,)
    psfgrid_coords.frame.name = det_coords.frame.name
    idx, _, _ = match_coordinates_sky(det_coords, psfgrid_coords)

    return psfgrid[idx]


def aperture_parameters(psfdata, eef, rotation):
    # semimajor axis, ellipticity and angle (with respect to detector axis)
    eefdata = psfdata[f"EEF{eef}"].data
    psfdata["PSF_a"] = _semimajor_axis_arcsec(eefdata)
    psfdata["PSF_b"] = _semiminor_axis_arcsec(eefdata)
    psfdata["PSF_pa"] = _position_angle_deg(eefdata, rotation)

    return psfdata


def _semimajor_axis_arcsec(eef):
    return eef[:, 0] * u.arcsec


def _semiminor_axis_arcsec(eef):
    return eef[:, 0] * (1 - eef[:, 1]) * u.arcsec


def _position_angle_deg(eef, rotation):
    return rotation - eef[:, 2] * u.deg
