from enum import Enum


class Eband(Enum):
    FULL = (500, 8000, 2000)
    SOFT = (500, 2000, 1000)
    HARD = (2000, 8000, 3500)
    UHRD = (7500, 12000, 8500)
    VHRD = (5000, 8000, 6000)

    XMM1 = (200, 500, 1000)
    XMM2 = (500, 1000, 1000)
    XMM3 = (1000, 2000, 1500)
    XMM4 = (2000, 4500, 2750)
    XMM5 = (4500, 1200, 6000)
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
