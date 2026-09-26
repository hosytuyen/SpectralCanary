import argparse

from provenance_t2i.common.metrics.score_utils import emit_json_report
from provenance_t2i.methods.diagnosis.classifier import default_device, train_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the DIAGNOSIS poisoned-vs-clean verifier classifier.")
    parser.add_argument("--manifest", required=True, help="Path to the training manifest JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoint and metrics.")
    parser.add_argument("--architecture", default="resnet50", choices=["resnet50", "resnet18"], help="Backbone architecture.")
    parser.add_argument(
        "--weights",
        default="imagenet",
        choices=["imagenet", "none"],
        help="Backbone initialization. 'imagenet' may require cached torchvision weights.",
    )
    parser.add_argument("--image-resolution", type=int, default=224, help="Classifier input resolution.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument("--device", default=default_device(), help="Torch device.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    spec, metrics_payload = train_classifier(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        architecture=args.architecture,
        weights=args.weights,
        image_resolution=args.image_resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    result = {
        "classifier_spec": spec.to_dict(),
        "metrics": metrics_payload,
    }
    emit_json_report(result, args.output)


if __name__ == "__main__":
    main()
