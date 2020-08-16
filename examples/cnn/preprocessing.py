from PIL import Image
import numpy as np


def grayscale(frame):
    return np.array(Image.fromarray(frame).convert('L'))
