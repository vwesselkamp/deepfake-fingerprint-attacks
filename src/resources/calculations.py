"""Much of the calculation performed here is adopted from:
https://github.com/RUB-SysSec/GANDCTAnalysis/"""

import numpy as np
from PIL import Image
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio

from lib.GANDCTAnalysis.src.math import welford, log_scale, welford_multidimensional, _welford_update
from lib.GANDCTAnalysis.src.image_np import load_image


def dct(image):
    """Performs more-dimensional DCT on image"""
    return fftpack.dctn(image, norm='ortho', axes=[0, 1])


def idct(image):
    """Performs inverse more-dimensional DCT on image"""
    return fftpack.idctn(image, norm='ortho', axes=[0, 1])


def reverse_transform(manipulated):
    # clip limits all array elements to be between 0 and 255
    # The manipulated DCT spectra not necessarily fall into 0 - 255.
    # reducing the DCT coefficient at 0,0 shifts all pixel values of the reverse transformed image
    restored = idct(manipulated)
    restored = np.clip(restored, 0, 255)
    return restored.astype('uint8')


def restore_from_dct(manipulated, greyscale=False):
    """Apply inverse DCT and convert back to a grayscale image"""
    restored = Image.fromarray(reverse_transform(manipulated))
    if greyscale:
        return restored.convert("L")
    else:
        return restored.convert("RGB")


def subtract_fingerprint(image, factor, fingerprint):
    """
    Receives two DCT spectras as input, subtracts one from the other.
    The spectras are not log-scaled, and they are not absolute, so the fingerprint as well as the image contain
    negative coefficients.
    """
    return image - factor * fingerprint


def reduce_fingerprint_proportionally(image, factor, fingerprint):
    """
    As the fingerprints are results from normalized and log-scaled images, we consider them to be a representation of
    proportionality of importance, not as absolute values. We therefore reduce the frequencies by the percentage given
    by the fingerprints, scaled by a factor.

    If the coefficient is positive, the absolute value of the frequency gets reduced, if negative it is raised.
    """
    return image * (1 - np.clip(factor * fingerprint, -1, 1))


def average_dct(images: list, log=True) -> np.ndarray:
    """Takes list of DCT-converted images and calculates the average of the DCT spectrum of images.
    By default returns is log-scaled"""
    # Calculating the mean by Welford has constant memory consumption,
    # Welford is used by Frank et al.
    if images[0].ndim == 2:
        average_dct_spectrum = welford(images)[0]
    elif images[0].ndim == 3:
        average_dct_spectrum = welford_multidimensional(images)[0]
    print("Calculated mean of image")
    if log:
        return log_scale(average_dct_spectrum)
    else:
        return average_dct_spectrum


def average_spatial(images):
    return welford([image.astype(np.float64) for image in images])[0]


def cosine_similarity_correlation(image_a, image_b):
    return np.mean(
        np.nan_to_num(np.divide(np.multiply(image_a, image_b), np.multiply(np.abs(image_a), np.abs(image_b)))))


def invert_log_scale(array, epsilon=1e-12):
    """Log scale the input array.
    """
    array = np.exp(array)
    array -= epsilon  # no zero in log
    return array


def welford_multidimensional_progress(sample):
    """
    Executes multidimensional welford on and saves means from all aggregates
    """
    aggregates = {}
    means_by_sample_size = list()
    for data in sample:
        # for each sample update each axis seperately
        mean = list()
        for i, d in enumerate(data):
            existing_aggregate = aggregates.get(i, (None, None, None))
            existing_aggregate = _welford_update(existing_aggregate, d)
            aggregates[i] = existing_aggregate
            mean.append(existing_aggregate[1])
        means_by_sample_size.append(np.asarray(mean))

    return means_by_sample_size


def load_image_and_compare(path, grayscale=False):
    """
    calulates PSNR = peak signal-to-noise ratio. The higher the better, about 30~50dB is good.
    """
    original = load_image(path[0], grayscale=grayscale)
    manipulated = load_image(path[1], grayscale=grayscale)

    psnr = peak_signal_noise_ratio(original, manipulated)
    return psnr


def fft(img):
    epsilon = 1e-8
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon
    return fshift


def remove_high_frequencies(img, factor, pattern=None):
    image = np.copy(img)
    x, y, _ = image.shape
    image[x - factor:, :, :] = 0
    image[:, y - factor:, :] = 0
    return image


def moving_frequencies(img, factor, pattern=None):
    image = np.copy(img)
    BAND_WIDTH = 10
    x, y, _ = image.shape
    image[x - factor - BAND_WIDTH:x - factor, :, :] = 0
    image[:, y - factor - BAND_WIDTH:y - factor, :] = 0
    return image


def get_psnr(originals, manipulated):
    psnrs = np.asarray([peak_signal_noise_ratio(original, man.clip(0, 255).astype(np.uint8))
                        for original, man in zip(originals, manipulated)])
    return np.mean(psnrs[np.isfinite(psnrs)])

