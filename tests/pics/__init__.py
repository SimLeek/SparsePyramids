from pathlib import Path
from PIL import Image
import numpy as np

larg = str(Path.joinpath(Path(__file__).parent, "larg.jpg"))
medium = str(Path.joinpath(Path(__file__).parent, "medium.jpg"))
smol = str(Path.joinpath(Path(__file__).parent, "smol.jpg"))

if __name__=='__main__':
    from displayarray import breakpoint_display

    breakpoint_display(larg)
    breakpoint_display(medium)
    breakpoint_display(smol)