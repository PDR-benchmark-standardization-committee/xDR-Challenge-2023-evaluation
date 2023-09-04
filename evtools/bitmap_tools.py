#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from PIL import Image
import numpy as np

def load_bitmap_to_ndarray(filename):
    # load image as PIL Image Object
    image = Image.open(filename)
    # convert to ndarray and scaling 0-1
    array = np.array(image) / 255.0
    return array
