import os
from .fft import *


import sys

from csrc.pic_codec import fast_idct



def inplace_idct(x, scale=1):
    fast_idct(x, scale)
