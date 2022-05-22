"""
Script to run innvestigate library on classifiers. Since the classifier by Frank et al. uses tf2, but the innvestigate
library is designed for tf < 1.15, we use a fork which only contains the LRP methods.
"""

import argparse
from pathlib import Path

from lib.GANDCTAnalysis.src.dataset import image_paths
from lib.innvestigate.src import innvestigate
import numpy as np
import matplotlib.pyplot as plt
from src.resources.data_preparation import load_and_preprocess
import tensorflow as tf

GAN_CLASS = 1
REAL_CLASS = 0


def investigate_image(i, path, mean, std, analyzer, model):
    image = load_and_preprocess(path, mean, std)
    # to replace batches
    image = np.expand_dims(image, axis=0)

    # argmax gets index of largest number in array of probabilites
    # this corresponds to the class
    prediction = tf.argmax(model(image), axis=1)

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(image)
    # to fix error message
    a = a['input_1']
    # Aggregate along color channels and normalize to [-1, 1]
    a /= np.max(np.abs(a))
    print(f"\rAnalyzed {i + 1:6d} images", end="")
    return prediction, a


def save_explanation(explanations, output_file):
    if len(explanations) != 0:
        explanation = np.mean(explanations, axis=0)[0]
        np.save(output_file, explanation)
        explanation = explanation.sum(axis=np.argmax(np.asarray(explanation.shape) == 3))

        plt.imshow(explanation, cmap="seismic", clim=(-1, 1))
        plt.colorbar(pad=0.2)
        plt.savefig(f"{output_file}.pdf")


def main(args):
    if args.MODE == 'multi':
        gan = args.GAN
        model = tf.keras.models.load_model(args.CLASSIFIER)
        # Strip softmax layer
        model = innvestigate.utils.model_wo_softmax(model)
    elif args.MODE == 'binary':
        gan = GAN_CLASS
        model = tf.keras.models.load_model(args.CLASSIFIER)
    else:
        print("Not a valid classifier")
        return

    # needed for normalization
    try:
        mean = np.load(f"{args.GAN_STATS}/mean.npy")
        std = np.sqrt(np.load(f"{args.GAN_STATS}/var.npy"))
    except FileNotFoundError as e:
        print(f"Expected two files:\n{args.GAN_STATS}/mean.npy\n{args.GAN_STATS}/var.npy")

    if args.output is not None:
        output_file = args.output
    else:
        path = Path(args.CLASSIFIER)
        output_file = f"{path.parent}/{path.stem}_explanation_{args.method}"
        if args.MODE == "multi":
            output_file += f"_{args.GAN}"

    paths = sorted(image_paths(args.DATASET))

    # Create analyzer
    analyzer = innvestigate.create_analyzer(args.method, model)

    correct = []
    incorrect = []
    for i, path in enumerate(paths):
        image = load_and_preprocess(path, mean, std)
        # to replace batches
        image = np.expand_dims(image, axis=0)

        raw_prediction = model(image)
        prediction = extract_prediction(args.MODE, raw_prediction)
        explanation = extract_explanation(analyzer, image)

        # differentiate between explanations for correct and incorrect classifications
        if prediction == gan:
            correct.append(explanation)
            print(f"\rAnalyzed {len(correct):6d} images", end="")
            # gathered all 1000 explanations
            if len(correct) == args.number:
                break
        else:
            incorrect.append(explanation)
    save_explanation(correct, output_file + '_correct')
    save_explanation(incorrect, output_file + '_incorrect')


def extract_explanation(analyzer, image):
    # Apply analyzer w.r.t. maximum activated output-neuron
    explanation = analyzer.analyze(image)
    # to fix error message
    explanation = explanation['input_1']
    # normalize to [-1, 1]
    explanation /= np.max(np.abs(explanation))
    return explanation


def extract_prediction(classifier, raw_prediction):
    prediction = -1
    if classifier == 'multi':
        # this corresponds to the class
        prediction = tf.argmax(raw_prediction, axis=1)
    elif classifier == 'binary':
        prediction = tf.math.round(raw_prediction)
        prediction = tf.cast(prediction, tf.uint8)
        prediction = tf.reshape(prediction, shape=(-1,))
    return prediction.numpy()


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(help="Mode {binary|multi}.", dest="mode")
    binary = commands.add_parser("binary")
    binary.add_argument("MODE", help="Model type {multi, binary}", type=str)
    binary.add_argument("CLASSIFIER", help="Path to classifier", type=str)
    binary.add_argument("DATASET", help="Directory of images as input to model.", type=str)
    binary.add_argument("GAN_STATS", help="Directory to mean and var of IMAGES, to use for conversion to DCT.", type=str)
    binary.add_argument("--method", help="Analysis method to use. Default is 'lrp.z'",
                        type=str, default='lrp.z')
    binary.add_argument("--number", help="Number of images to analyze.", type=int)
    binary.add_argument("--output", help="Output file.", type=int)

    multi = commands.add_parser("multi")
    multi.add_argument("GAN", help="Number which indicates the correct GAN class.", type=int)
    multi.add_argument("MODE", help="Model type {multi, binary}", type=str)
    multi.add_argument("CLASSIFIER", help="Path to classifier", type=str)
    multi.add_argument("DATASET", help="Directory of images as input to model.", type=str)
    multi.add_argument("GAN_STATS", help="Directory to mean and var of IMAGES, to use for conversion to DCT.", type=str)
    multi.add_argument("--method", help="Analysis method to use. Default is 'lrp.z'",
                        type=str, default='lrp.z')
    multi.add_argument("--number", help="Number of images to analyze.", type=int)
    multi.add_argument("--output", help="Output file.", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
