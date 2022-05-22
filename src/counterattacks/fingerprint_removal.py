import argparse
import os
from pathlib import Path
import numpy as np
from lib.GANDCTAnalysis.src.dataset import image_paths
from lib.GANDCTAnalysis.src.image_np import load_image
from src.resources.calculations import subtract_fingerprint, reduce_fingerprint_proportionally, dct, restore_from_dct, \
    remove_high_frequencies

SIZE = [128, 128, 3]


def save_image(manipulated, name, output_path, greyscale=False):
    # Save reconstructed PNG image for both Frank and Yu et al.
    restored_image = restore_from_dct(manipulated, greyscale)
    restored_image.save(f"{output_path}/{name}.png")


def remove_fingerprint_and_save(paths, fingerprint, factor, output, proportional=False):
    for i, path in enumerate(paths):
        # load and manipulate image one after another
        # this will keep the memory requirement low
        image = load_image(path)
        image = dct(image)

        if not proportional:  # absolute fingerprint of mean spectra
            manipulated = subtract_fingerprint(image, factor, fingerprint)
        else:  # proportional fingerprint of regression weights
            manipulated = reduce_fingerprint_proportionally(image, factor, fingerprint)
        save_image(manipulated, Path(path).stem, output)
        print(f"\rRemoved fingerprint from {i+1:6d} of {len(paths)} images", end="")
    print()


def remove_bar(paths, output, width):
    for i, path in enumerate(paths):
        image = load_image(path)
        image = dct(image)

        image = remove_high_frequencies(image, width)

        # convert back and save
        save_image(image, Path(path).stem, output)
        print(f"\rRemoved bar from {i + 1:6d} of {len(paths)} images", end="")
    print()


def main(args):
    # Create output folder, if it doesn't already exist
    if args.output is not None:
        output = args.output
    else:
        path = Path(args.GAN_IMAGES)
        output = f"{path.parent}/{path.stem}_{args.mode}"
    os.makedirs(output, exist_ok=True)

    paths = image_paths(args.GAN_IMAGES)

    if args.mode == "mean":
        fingerprint = np.load(args.FINGERPRINT)
        print(f"Remove fingerprint directly from {args.GAN_IMAGES}")
        remove_fingerprint_and_save(paths, fingerprint, args.factor, output, proportional=False)
    elif args.mode == "peak":
        fingerprint = np.load(args.FINGERPRINT)
        print(f"Remove fingerprint proportionally from {args.GAN_IMAGES}")
        if args.threshold is not None:
            assert 1 >= args.threshold >= 0
            fingerprint[fingerprint < args.threshold] = 0
        remove_fingerprint_and_save(paths, fingerprint, args.factor, output, proportional=True)
    elif args.mode == "regression":
        fingerprint = np.load(args.FINGERPRINT)
        print(f"Remove fingerprint proportionally from {args.GAN_IMAGES}")
        remove_fingerprint_and_save(paths, fingerprint, args.factor, output, proportional=True)
    elif args.mode == "bar":
        print(f"Remove bars {args.GAN_IMAGES}")
        remove_bar(paths, output, args.width)
    else:
        raise NotImplementedError("Specified non valid mode!")


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(help="Mode {mean|peak|regression|bar}.", dest="mode")

    mean = commands.add_parser("mean")
    mean.add_argument("GAN_IMAGES", help="Folder of GAN image dataset to be manipulated.", type=str)
    mean.add_argument("FINGERPRINT", help=f".npy file which contains the precalculated fingerprint", type=str, default=1)
    mean.add_argument("--output", "-o",help=f"Output folder.", type=str)
    mean.add_argument("--factor", help=f"Factor by which to scale the fingerprint before removal", type=float, default=1)
    peak = commands.add_parser("peak")
    peak.add_argument("GAN_IMAGES", help="Folder of GAN image dataset to be manipulated.", type=str)
    peak.add_argument("FINGERPRINT", help=f".npy file which contains the precalculated fingerprint", type=str, default=1)
    peak.add_argument("--output", "-o",help=f"Output folder.", type=str)
    peak.add_argument("--factor", help=f"Factor by which to scale the fingerprint before removal", type=float, default=1)
    peak.add_argument("--threshold", help=f"Threshold, which to apply to fingerprint before removal", type=float)
    regression = commands.add_parser("regression")
    regression.add_argument("GAN_IMAGES", help="Folder of GAN image dataset to be manipulated.", type=str)
    regression.add_argument("FINGERPRINT", help=f".npy file which contains the precalculated fingerprint", type=str, default=1)
    regression.add_argument("--output", "-o",help=f"Output folder.", type=str)
    regression.add_argument("--factor", help=f"Factor by which to scale the fingerprint before removal", type=float, default=1)
    bar = commands.add_parser("bar")
    bar.add_argument("GAN_IMAGES", help="Folder of GAN image dataset to be manipulated.", type=str)
    bar.add_argument("--width", help=f"Width of bar to remove. Default is 10.", type=int, default=10)
    bar.add_argument("--output", "-o",help=f"Output folder.", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())

