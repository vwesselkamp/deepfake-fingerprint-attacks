"""This script for cropping LSUN was taken directly from: https://github.com/RUB-SysSec/GANDCTAnalysis/blob/master/crop_lsun.py
It has been slightly adapted to guarantee quadratic crops for non quadratic images.
The authers in turn adapted their script from: https://github.com/ningyu1991/GANFingerprints/"""
import argparse
import os

from PIL import Image
import numpy as np
from skimage.transform import resize

from concurrent.futures import ProcessPoolExecutor


def transform_image(stupid):
    file_path, directory, output = stupid
    if file_path.endswith("png") or file_path.endswith("jpeg") or file_path.endswith("jpg"):
        image = np.asarray(Image.open(f"{directory}/{file_path}"))

        if image.shape[0] != 128 or image.shape[1] != 128:
            x, y, _ = image.shape

            # center crop towards smaller side
            if x < y:
                y_center = y // 2
                crop = x // 2  # adapted for non quadratic images
                image = np.copy(image)
                image = image[:, y_center - crop:y_center + crop]

            elif x > y:
                x_center = x // 2
                crop = y // 2  # adapted for non quadratic images
                image = np.copy(image)
                image = image[x_center - crop:x_center + crop, :]

            try:
                # resize quadratic image to 128 pixel size
                image = resize(image.astype(np.float64), (128, 128))
            except Exception as e:
                print(e)
            image = np.clip(image, 0, 255.).astype(np.uint8)

        Image.fromarray(image).save(f"{output}/{file_path}")


def main(args):
    os.makedirs(args.OUTPUT, exist_ok=True)

    paths = os.listdir(args.DIRECTORY)
    packed = map(lambda p: (p, args.DIRECTORY, args.OUTPUT), paths)
    with ProcessPoolExecutor() as pool:
        list(pool.map(transform_image, packed))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("DIRECTORY", help="Source directory.", type=str)
    parser.add_argument("OUTPUT", help="Output directory.", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
