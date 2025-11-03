"""
Module defining energy bands for XMM-Newton EPIC cameras.
"""
from enum import Enum


class Eband(Enum):
    """
    Energy bands for XMM-Newton EPIC cameras.
    Each band is represented as a tuple of (min_energy, max_energy, mean_energy) in eV.

    Attributes
    ----------
    FULL:  Full energy band (500 eV to 8000 eV)
    SOFT:  Soft energy band (500 eV to 2000 eV)
    HARD:  Hard energy band (2000 eV to 8000 eV)
    UHRD:  Ultra High Resolution Detector band (7500 eV to 12000 eV)
    VHRD:  Very High Resolution Detector band (5000 eV to 8000 eV)
    XMM1:  XMM-Newton band 1 (200 eV to 500 eV)
    XMM2:  XMM-Newton band 2 (500 eV to 1000 eV)
    XMM3:  XMM-Newton band 3 (1000 eV to 2000 eV)
    XMM4:  XMM-Newton band 4 (2000 eV to 4500 eV)
    XMM5:  XMM-Newton band 5 (4500 eV to 12000 eV)
    XMM8:  XMM-Newton band 8 (200 eV to 12000 eV)

    Methods
    -------
    all(): 
        Returns a list of the main energy bands (FULL, SOFT, HARD).
    xmmall(): 
        Returns a list of the XMM-Newton specific energy bands (XMM1 to XMM5).
    """
    FULL = (500, 8000, 2000)
    SOFT = (500, 2000, 1000)
    HARD = (2000, 8000, 3500)
    UHRD = (7500, 12000, 8500)
    VHRD = (5000, 8000, 6000)

    XMM1 = (200, 500, 1000)
    XMM2 = (500, 1000, 1000)
    XMM3 = (1000, 2000, 1500)
    XMM4 = (2000, 4500, 2750)
    XMM5 = (4500, 12000, 6000)
    XMM8 = (200, 12000, 2000)

    def __init__(self, min, max, mean):
        self.min = min
        self.max = max
        self.mean = mean

    @classmethod
    def all(self):
        return [self.FULL, self.SOFT, self.HARD]

    @classmethod
    def xmmall(self):
        return [self.XMM1, self.XMM2, self.XMM3, self.XMM4, self.XMM5]
