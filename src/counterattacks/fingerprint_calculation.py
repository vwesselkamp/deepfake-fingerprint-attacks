import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

from lib.GANDCTAnalysis.src.math import welford, log_scale, welford_multidimensional
from src.resources.calculations import invert_log_scale, dct
from src.resources.data_preparation import get_name_from_path, load_images_from_directory


SIZE = [128, 128, 3]


def calculate_fingerprint_from_mean(dataset_gan, dataset_real, greyscale=False, geometric=False, threshold=None):
    """ Calculates the fingeprint by subtracting the mean DCT of a GAN by the mean DCT of the underlying dataset"""
    mean_gan = load_images_and_get_mean_dct(dataset_gan, greyscale, geometric)
    mean_real = load_images_and_get_mean_dct(dataset_real, greyscale, geometric)

    # what Frank et al. do to calculate the absolute difference is
    # np.abs(log_scale(self.ref_mean) - log_scale(mean))
    fingerprint = mean_gan - mean_real

    # exponentiate the log-average to retrieve geometric mean
    if geometric:
        # normalize to [0, 1]
        fingerprint = invert_log_scale(fingerprint)
        fingerprint /= np.max(fingerprint)
        if threshold:
            assert 1 >= threshold >= 0
            fingerprint[fingerprint < threshold] = 0

    return fingerprint


def load_images_and_get_mean_dct(data_path, greyscale=False, geometric=False):
    # Welford has constant memory requirement while calculating the mean
    # give paths to welford and load an convert images there
    images = load_images_from_directory(data_path)
    images = map(dct, images)

    if geometric:
        images = map(log_scale, images)

    if greyscale:
        average_dct_spectrum = welford(images)[0]
    else:
        average_dct_spectrum = welford_multidimensional(images)[0]
    print(f"Mean of dataset {get_name_from_path(data_path)} is calculated")
    return average_dct_spectrum


def extract_weights(model_path):
    model = keras.models.load_model(model_path)
    w = model.trainable_weights[0].numpy().reshape(SIZE)
    return w


def gen_output(args):
    if args.output is not None:
        path = args.output
        os.makedirs(path, exist_ok=True)
    else:
        path = os.getcwd()
    if args.mode == "regression":
        return f"{path}/fingerprint_{args.mode}_{os.path.basename(args.MODEL)}"
    else:
        return f"{path}/fingerprint_{args.mode}_{os.path.basename(args.GAN_IMAGES)}_{os.path.basename(args.REAL_DATASET)}"


def save_fingerprint(fingerprint, output, mode):
    """Saves fingerprint as numpy array, and as log-scaled heatmap"""
    print(f"Saved at {output}")
    np.save(f"{output}.npy", fingerprint)

    # merge color channels
    fingerprint = np.mean(fingerprint, axis=2)

    # If it not already the geometric mean, log-scaled heatmap to visualize fingerprint
    if mode == "mean":
        fingerprint = log_scale(fingerprint)

    plt.matshow(fingerprint, cmap='inferno')
    plt.colorbar(pad=0.2)
    plt.savefig(f"{output}.pdf")


def main(args):
    output = gen_output(args)
    if args.mode == "mean":
        print(f"Calculate fingerprint for mean spectrum attack on {args.GAN_IMAGES}")
        fingerprint = calculate_fingerprint_from_mean(args.GAN_IMAGES, args.REAL_DATASET, args.grayscale, False)
    elif args.mode == "peak":
        print(f"Calculate fingerprint for peak extraction attack on {args.GAN_IMAGES}")
        fingerprint = calculate_fingerprint_from_mean(args.GAN_IMAGES, args.REAL_DATASET, args.grayscale, True, args.threshold)
    elif args.mode == "regression":
        print(f"Calculate fingerprint for regression attack on {args.MODEL}")
        fingerprint = extract_weights(args.MODEL)
    else:
        raise NotImplementedError("Specified non valid mode!")
    save_fingerprint(fingerprint, output, args.mode)

def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(help="Mode {mean|peak|regression}.", dest="mode")

    mean = commands.add_parser("mean")
    mean.add_argument("GAN_IMAGES", help="Path to GAN image dataset", type=str)
    mean.add_argument("REAL_DATASET", help="Path to dataset of real images", type=str)
    mean.add_argument("--grayscale", "-g", help=f"Calculate on grayscaled images.", action="store_true")
    mean.add_argument("--output", "-o",help=f"Output folder.", type=str)

    peak = commands.add_parser("peak")
    peak.add_argument("GAN_IMAGES", help="Path to GAN image dataset", type=str)
    peak.add_argument("REAL_DATASET", help="Path to dataset of real images", type=str)
    peak.add_argument("--grayscale", "-g", help=f"Calculate on grayscaled images.", action="store_true")
    peak.add_argument("--threshold", "-t", help=f"Apply threshold within [0,1].", type=float)
    peak.add_argument("--output", "-o", help=f"Output folder.", type=str)

    regression = commands.add_parser("regression")
    regression.add_argument("MODEL", help="Path to regression model", type=str)
    regression.add_argument("--output", "-o", help=f"Output folder.", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())

