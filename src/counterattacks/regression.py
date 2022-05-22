import argparse

from lib.GANDCTAnalysis.classifier import train

# values chosen from paper and from experimentation
L1 = [0.01, 0.001, 0.0001, 0.00001]
L2 = [0.1, 0.01, 0.001, 0.0001, 0.00001]


def main(args):
    if args.MODEL == 'log1':
        regularization = L1
    elif args.MODEL == 'log2':
        regularization = L2
    else:
        TypeError(f"Model of type {args.MODEL} not possible")

    results = []
    for i, factor in enumerate(regularization):
        print(f"Training model with regularization factor {factor}.")
        if args.MODEL == 'log1':
            args.l1 = factor
        elif args.MODEL == 'log2':
            args.l2 = factor

        # pass onto code by Frank et al. in lib/GANDCTAnalysis
        model, eval_accuracy, model_dir = train(args)
        print(f"Model {i} with factor {factor} has accuracy - {eval_accuracy:.2%}")
        results.append({'model': model, 'acc': eval_accuracy, 'reg': factor})

    # model wih highest accuracy gets chosen
    best_model = max(results, key=lambda x: x['acc'])
    output = f'{args.output}/{args.MODEL}'

    best_model['model'].save(output, save_format="tf")
    print(f"Model with factor {best_model['reg']} and accuracy {best_model['acc']} performs best.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", "-s", help="Images to load.", type=int, default=None)

    parser.add_argument(
        "MODEL", help="Select model to train {log1, log2}.", type=str)
    parser.add_argument("TRAIN_DATASET", help="Dataset to load.", type=str)
    parser.add_argument("VAL_DATASET", help="Dataset to load.", type=str)
    parser.add_argument("TEST_DATASET", help="Dataset to load.", type=str)
    parser.add_argument("--debug", "-d", help="Debug mode.",
                       action="store_true")
    parser.add_argument(
        "--epochs", "-e", help=f"Epochs to train for.", type=int, default=50)
    parser.add_argument("--image_size",
                       help=f"Image size.", type=int, default=128)
    parser.add_argument("--early_stopping",
                       help=f"Early stopping criteria. Default: 5", type=int, default=5)
    parser.add_argument("--classes",
                       help=f"Classes.", type=int, default=2)
    parser.add_argument("--grayscale", "-g",
                       help=f"Train on grayscaled images.", action="store_true")
    parser.add_argument("--batch_size", "-b",
                       help=f"Batch size.", type=int, default=32)
    parser.add_argument("--l1",
                       help=f"L1 regularizer intensity. Default: 0.01", type=float, default=0.01)
    parser.add_argument("--l2",
                       help=f"L2 regularizer intensity. Default: 0.01", type=float, default=0.01)
    parser.add_argument("--output",
                       help=f"Output directory", type=str, default='final_models')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_args())
