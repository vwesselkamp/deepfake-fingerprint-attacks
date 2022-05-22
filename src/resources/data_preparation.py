"""Much of the preparation performed here is adopted from:
https://github.com/RUB-SysSec/GANDCTAnalysis/"""

import numpy as np
import os
from lib.GANDCTAnalysis.src.dataset import image_paths
from lib.GANDCTAnalysis.src.image_np import load_image, dct2


def load_images_from_directory(directory: str, greyscale=False):
    """ Loads pngs/jpegs from a given directory.
    :param directory: path to directory
    :return: a list of greyscale images as np.ndarray
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory {directory} does not exist')
    paths = image_paths(directory)  # gets path of all jpeg/pngs
    images = map(lambda d: load_image(d, grayscale=greyscale), paths)
    return list(images)


def get_name_from_path(path):
    """Extract name from directory name"""
    parts = path.split('/')
    return parts[-1]


def load_and_preprocess(image_path, mean, std):
    image = load_image(image_path)
    image = image.astype(np.float32)
    # dct2
    x = dct2(image)

    # log scale
    x = np.abs(x)
    x += 1e-13
    x = np.log(x)

    # remove mean + unit variance
    x = x - mean
    x = x / std

    return x
