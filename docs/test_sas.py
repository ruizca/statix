import logging
from pathlib import Path

from rich.logging import RichHandler
from statix.exposure import Exposure
from statix.xmmsas import make_ccf


logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
data_path = Path(".", "data")
data_path = data_path.resolve()

event_list_path = data_path / "pnevt.fits"
att_path = data_path / "att.fits"

make_ccf(data_path)
xmmexp = Exposure(event_list_path, att_path, zsize=32)

# Run SAS emldetect algorithm
# This also creates all products needed 
# for running STATiX, except the data cube
srclist = xmmexp.detect_sources(method="emldetect", likemin=6)

# Create data cube
cube = xmmexp.cube

# Run STATiX algorithm
srclist = xmmexp.detect_sources()
