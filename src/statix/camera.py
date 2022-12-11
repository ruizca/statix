from functools import cached_property

from astropy import units as u
from astropy.coordinates import SkyCoord, SkyOffsetFrame


class Camera:
    NAMES = ["EPN", "EMOS1", "EMOS2"]
    ANGLES = {"EPN": 90 - 360, "EMOS1": -180, "EMOS2": 90 - 360}

    def __init__(self, name, orientation=None):
        self.name = self._set_name(name)
        self.type = self._set_type()
        self.orientation = orientation

        self._reference_frame = None

    def _set_name(self, name):
        if name in self.NAMES:
            return name
        else:
            raise ValueError(f"Unknown camera: {name}")

    def _set_type(self):
        if self.name == "EPN":
            return "pn"
        else:
            return "mos"

    def __repr__(self) -> str:
        return self.name

    @property
    def tag(self):
        return f"{self.name[1]}{self.name[-1]}"

    @cached_property
    def reference_frame(self):
        if not self.orientation:
            raise AttributeError("Orientation is not defined for this camera.")

        if not self._reference_frame:
            self._reference_frame = self._set_reference_frame()

        return self._reference_frame

    def _set_reference_frame(self):
        # The conversion is a rotation around the nominal pointing.
        # The rotation angle is the PA plus some detector-dependent angle
        rotation = self.orientation.pa + self.ANGLES[self.name] * u.deg

        # We can apply the rotation using the SkyOffsetFrame
        # machinery included in astropy.coordinates
        det_frame = SkyOffsetFrame(origin=self.orientation.pointing, rotation=rotation)
        det_frame.name = self.name

        return det_frame

    def coordinates(self, sky_coords):
        det_coords = sky_coords.transform_to(self.reference_frame, merge_attributes=False)
        det_coords.frame.name = self.reference_frame.name

        # empirical for pn, mos1 detx=-detx, dety=-dety
        if self.reference_frame.name in ["EPN", "EMOS1"]:
            det_coords.data.lon[()] = -1 * det_coords.lon
            det_coords.data.lat[()] = -1 * det_coords.lat
            det_coords.cache.clear()

        return det_coords


class Orientation:
    def __init__(self, ra_nom, dec_nom, pa):
        self.pointing = self._set_pointing_skycoord(ra_nom, dec_nom)
        self.pa = self._set_pa_deg(pa)

    @staticmethod
    def _set_pointing_skycoord(ra_nom, dec_nom):
        return SkyCoord(ra_nom, dec_nom, unit="deg", frame="fk5")

    @staticmethod
    def _set_pa_deg(pa):
        return pa * u.deg
