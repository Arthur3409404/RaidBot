"""Command line entry point for Raid champion icon data tools."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from raid_bot.champion_recognition.data_collector import (
    AyumiLoveCollector,
    CollectorConfig,
    CropConfig,
    DataCollectionError,
    check4updates,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Raid Shadow Legends champion icon tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect one champion icon sample")
    collect_parser.add_argument("--url", required=True, help="AyumiLove champion page URL")
    collect_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    collect_parser.add_argument("--force", action="store_true", help="Overwrite cached and processed files")
    collect_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds before network requests",
    )
    collect_parser.add_argument("--crop-left", type=float, default=None)
    collect_parser.add_argument("--crop-top", type=float, default=None)
    collect_parser.add_argument("--crop-right", type=float, default=None)
    collect_parser.add_argument("--crop-bottom", type=float, default=None)

    ranking_parser = subparsers.add_parser(
        "collect-ranking",
        help="Collect champion icons from all champion links on a ranking page",
    )
    ranking_parser.add_argument("--url", required=True, help="AyumiLove ranking/list page URL")
    ranking_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    ranking_parser.add_argument("--force", action="store_true", help="Overwrite cached and processed files")
    ranking_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds before network requests",
    )
    ranking_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Collect only the first N discovered champion links",
    )
    ranking_parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch run when one champion page fails",
    )

    reprocess_parser = subparsers.add_parser(
        "reprocess-icons",
        help="Re-crop existing labels from cached raw images using the saved crop config",
    )
    reprocess_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    reprocess_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed icons and crop debug images",
    )

    update_parser = subparsers.add_parser(
        "check4updates",
        help="Check AyumiLove for champion links not yet present in the dataset",
    )
    update_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    update_parser.add_argument(
        "--url",
        default="https://ayumilove.net/raid-shadow-legends-list-of-champions-by-ranking/",
        help="AyumiLove ranking/list page URL",
    )

    crop_parser = subparsers.add_parser(
        "configure-crop",
        help="Interactively select and save the shared relative crop box",
    )
    crop_parser.add_argument("--url", required=True, help="AyumiLove champion page URL")
    crop_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    crop_parser.add_argument("--force", action="store_true", help="Overwrite crop config and outputs")
    crop_parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds before network requests",
    )

    train_parser = subparsers.add_parser("train-recognizer", help="Train the champion icon recognizer")
    add_recognizer_common_args(train_parser, include_train_folder=True)
    train_parser.add_argument("--force", action="store_true", help="Overwrite existing model artifacts")
    train_parser.add_argument("--epochs", type=int, default=10, help="Frozen-backbone classifier-head epochs")
    train_parser.add_argument("--fine-tune-epochs", type=int, default=0, help="Optional low-LR fine-tuning epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    train_parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument(
        "--benchmark-class-count",
        type=int,
        default=0,
        help="Limit training to a fixed number of champion classes for a quick benchmark run",
    )
    train_parser.add_argument(
        "--benchmark-seed",
        type=int,
        default=42,
        help="Random seed used to select benchmark classes",
    )
    train_parser.add_argument("--no-pretrained", action="store_true", help="Do not load pretrained timm weights")
    train_parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable inverse-frequency class-balanced training sampling",
    )

    prototype_parser = subparsers.add_parser(
        "build-prototypes",
        help="Build clean champion prototype embeddings from a trained recognizer",
    )
    add_recognizer_common_args(prototype_parser, include_train_folder=False)

    benchmark_train_parser = subparsers.add_parser(
        "benchmark-train-recognizer",
        help="Train on a small champion subset for a fast benchmark run",
    )
    add_recognizer_common_args(benchmark_train_parser, include_train_folder=True)
    benchmark_train_parser.add_argument("--force", action="store_true", help="Overwrite existing model artifacts")
    benchmark_train_parser.add_argument("--epochs", type=int, default=10, help="Frozen-backbone classifier-head epochs")
    benchmark_train_parser.add_argument("--fine-tune-epochs", type=int, default=0, help="Optional low-LR fine-tuning epochs")
    benchmark_train_parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    benchmark_train_parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate")
    benchmark_train_parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    benchmark_train_parser.add_argument("--unfreeze-last-blocks", type=int, default=1)
    benchmark_train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    benchmark_train_parser.add_argument("--benchmark-class-count", type=int, default=16)
    benchmark_train_parser.add_argument("--benchmark-seed", type=int, default=42)
    benchmark_train_parser.add_argument("--no-pretrained", action="store_true", help="Do not load pretrained timm weights")
    benchmark_train_parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable inverse-frequency class-balanced training sampling",
    )
    benchmark_train_parser.set_defaults(
        checkpoint_path="data/models/champion_recognition/checkpoints/benchmark_champion_icon_recognizer.pt",
        prototype_path="data/models/champion_recognition/checkpoints/benchmark_champion_prototypes.pt",
        log_dir="data/models/champion_recognition/logs/benchmark",
        plot_dir="data/models/champion_recognition/plots/benchmark",
        review_samples_dir="data/models/champion_recognition/review_samples/benchmark",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate-recognizer",
        help="Evaluate the recognizer against clean reference icons",
    )
    add_recognizer_common_args(evaluate_parser, include_train_folder=False)

    post_processing_parser = subparsers.add_parser(
        "post-processing-ai",
        help="Analyze trained weights and recommend confidence thresholds",
    )
    add_recognizer_common_args(post_processing_parser, include_train_folder=False)

    noise_parser = subparsers.add_parser(
        "noise-data",
        help="Create realistic augmented variants from processed champion icons",
    )
    noise_parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    noise_parser.add_argument(
        "--source-folder",
        default=None,
        help="Clean/source icon folder, default data/processed/icons",
    )
    noise_parser.add_argument(
        "--output-folder",
        default=None,
        help="Augmented icon folder, default data/processed/icons_noised",
    )
    noise_parser.add_argument(
        "--metadata-path",
        default=None,
        help="Metadata CSV path, default data/processed/augmentation_metadata.csv",
    )
    noise_parser.add_argument("--variants-per-icon", type=int, default=25)
    noise_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    noise_parser.add_argument(
        "--preview-grid",
        default=None,
        help="Optional path for an original-vs-augmented preview grid image",
    )
    noise_parser.add_argument("--preview-examples", type=int, default=8)
    noise_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing augmented images and metadata CSV",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict the champion for one icon")
    predict_parser.add_argument("--image", required=True, help="Path to a 48x64 champion icon")
    add_recognizer_common_args(predict_parser, include_train_folder=False)
    return parser


def add_recognizer_common_args(parser: argparse.ArgumentParser, *, include_train_folder: bool) -> None:
    """Add shared recognizer-only CLI options."""
    parser.add_argument("--data-dir", default="data", help="Dataset root directory")
    if include_train_folder:
        parser.add_argument(
            "--train-image-folder",
            default=None,
            help="Augmented/noisy training icon folder, default data/processed/icons_noised",
        )
    parser.add_argument(
        "--reference-icon-folder",
        default=None,
        help="Clean reference icon folder, default data/processed/icons",
    )
    parser.add_argument("--labels-csv", default=None, help="Labels CSV path")
    parser.add_argument("--backbone", default="convnext_tiny", help="timm backbone name")
    parser.add_argument("--image-size", type=int, default=224, help="Backbone input image size")
    parser.add_argument("--checkpoint-path", default="data/models/champion_recognition/checkpoints/champion_icon_recognizer.pt")
    parser.add_argument("--prototype-path", default="data/models/champion_recognition/checkpoints/champion_prototypes.pt")
    parser.add_argument("--log-dir", default="data/models/champion_recognition/logs")
    parser.add_argument("--plot-dir", default="data/models/champion_recognition/plots")
    parser.add_argument("--review-samples-dir", default="data/models/champion_recognition/review_samples")
    parser.add_argument("--min-similarity-threshold", type=float, default=0.65)
    parser.add_argument("--min-margin-threshold", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda")


def configure_logging() -> None:
    """Configure concise console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        crop_values = (args.crop_left, args.crop_top, args.crop_right, args.crop_bottom)
        if any(value is not None for value in crop_values) and not all(value is not None for value in crop_values):
            logging.getLogger(__name__).error("Pass all four crop values or none of them")
            return 1

        config = CollectorConfig(
            data_dir=Path(args.data_dir),
            request_delay_seconds=args.delay,
            crop=(
                None
                if args.crop_left is None
                else CropConfig(
                    left=args.crop_left,
                    top=args.crop_top,
                    right=args.crop_right,
                    bottom=args.crop_bottom,
                )
            ),
        )
        collector = AyumiLoveCollector(config)
        try:
            result = collector.collect(args.url, force=args.force)
        except (DataCollectionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(f"Collected {result.champion_name} -> {result.processed_image_path}")
        print(f"Processed image size: {result.image_width}x{result.image_height}")
        print(f"Crop debug image: {result.debug_image_path}")
        print(f"Labels CSV: {config.labels_path}")
        return 0

    if args.command == "configure-crop":
        config = CollectorConfig(
            data_dir=Path(args.data_dir),
            request_delay_seconds=args.delay,
        )
        collector = AyumiLoveCollector(config)
        try:
            result = collector.configure_crop_interactively(args.url, force=args.force)
        except (DataCollectionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(f"Saved crop config: {config.crop_config_path}")
        print(f"Selected crop box: {result.crop_box}")
        print(f"Processed image: {result.processed_image_path}")
        print(f"Crop debug image: {result.debug_image_path}")
        return 0

    if args.command == "collect-ranking":
        config = CollectorConfig(
            data_dir=Path(args.data_dir),
            request_delay_seconds=args.delay,
        )
        collector = AyumiLoveCollector(config)
        try:
            result = collector.collect_from_index(
                args.url,
                force=args.force,
                limit=args.limit,
                stop_on_error=args.stop_on_error,
            )
        except (DataCollectionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(f"Discovered champion links: {result.discovered_links}")
        print(f"Attempted: {result.attempted_links}")
        print(f"Successful: {result.successful}")
        print(f"Failed: {result.failed}")
        print(f"Labels CSV: {config.labels_path}")
        if result.errors:
            print("Failures:")
            for error in result.errors:
                print(f"  {error['url']}: {error['error']}")
        return 0

    if args.command == "reprocess-icons":
        config = CollectorConfig(data_dir=Path(args.data_dir))
        collector = AyumiLoveCollector(config)
        try:
            result = collector.reprocess_existing_icons(force=args.force)
        except (DataCollectionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(f"Attempted: {result.attempted_links}")
        print(f"Successful: {result.successful}")
        print(f"Failed: {result.failed}")
        print(f"Labels CSV: {config.labels_path}")
        if result.errors:
            print("Failures:")
            for error in result.errors:
                print(f"  {error['url']}: {error['error']}")
        return 0

    if args.command == "check4updates":
        try:
            check4updates(data_dir=Path(args.data_dir), index_url=args.url)
        except (DataCollectionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1
        return 0

    if args.command == "noise-data":
        try:
            from raid_bot.champion_recognition import ChampionRecognitionError, create_noise_data
        except ModuleNotFoundError as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        data_dir = Path(args.data_dir)
        processed_dir = data_dir / "processed"
        try:
            summary = create_noise_data(
                source_folder=Path(args.source_folder) if args.source_folder else processed_dir / "icons",
                output_folder=Path(args.output_folder) if args.output_folder else processed_dir / "icons_noised",
                metadata_path=Path(args.metadata_path)
                if args.metadata_path
                else processed_dir / "augmentation_metadata.csv",
                labels_csv_path=processed_dir / "labels.csv",
                variants_per_icon=args.variants_per_icon,
                random_seed=args.seed,
                preview_grid_path=Path(args.preview_grid) if args.preview_grid else None,
                preview_examples=args.preview_examples,
                overwrite=args.overwrite,
            )
        except (ChampionRecognitionError, ValueError) as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        print(f"Original icons found: {summary.original_icons_found}")
        print(f"Labels/champions found: {summary.labels_found}")
        print(f"Variants per icon: {summary.variants_per_icon}")
        print(f"Total augmented images created: {summary.total_augmented_images_created}")
        print(f"Output folder path: {summary.output_folder}")
        print(f"Metadata CSV: {summary.metadata_path}")
        if summary.preview_path:
            print(f"Preview grid: {summary.preview_path}")
        return 0

    if args.command in {
        "train-recognizer",
        "benchmark-train-recognizer",
        "build-prototypes",
        "evaluate-recognizer",
        "predict",
        "post-processing-ai",
    }:
        try:
            from raid_bot.champion_recognition import (
                ChampionRecognitionError,
                analyze_post_training,
                build_and_save_prototypes,
                evaluate_saved_recognizer,
                predict_champion,
                train_recognizer,
            )
        except ModuleNotFoundError as exc:
            logging.getLogger(__name__).error("%s", exc)
            return 1

        if args.command in {"train-recognizer", "benchmark-train-recognizer"}:
            config = build_recognition_config(args)
            if args.command == "benchmark-train-recognizer" and getattr(args, "benchmark_class_count", 0) <= 0:
                args.benchmark_class_count = 16
            try:
                result = train_recognizer(config, force=args.force)
            except (ChampionRecognitionError, ValueError) as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1

            print(f"Saved checkpoint: {result.checkpoint_path}")
            print(f"Saved prototypes: {result.prototype_path}")
            print(f"Saved logs: {result.log_path}")
            print(f"Images/classes: {result.num_train_images}/{result.num_classes}")
            print(f"Top-1 nearest-prototype accuracy: {result.evaluation['top1_accuracy']}")
            print(f"None rate: {result.evaluation['none_rate']}")
            if result.best_val_loss_checkpoint_path:
                print(f"Best val-loss checkpoint: {result.best_val_loss_checkpoint_path}")
                print(f"Best val-loss prototypes: {result.best_val_loss_prototype_path}")
            if result.best_val_accuracy_checkpoint_path:
                print(f"Best val-acc checkpoint: {result.best_val_accuracy_checkpoint_path}")
                print(f"Best val-acc prototypes: {result.best_val_accuracy_prototype_path}")
            return 0

        if args.command == "build-prototypes":
            config = build_recognition_config(args)
            try:
                prototypes = build_and_save_prototypes(config)
            except ChampionRecognitionError as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1

            print(f"Saved prototypes: {config.prototype_path}")
            print(f"Champion prototypes: {len(prototypes.labels)}")
            return 0

        if args.command == "evaluate-recognizer":
            config = build_recognition_config(args)
            try:
                result = evaluate_saved_recognizer(config)
            except ChampionRecognitionError as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1

            print(f"Saved evaluation: {config.log_dir / 'evaluation.json'}")
            print(f"Top-1 accuracy: {result['top1_accuracy']}")
            print(f"Accepted accuracy: {result['accepted_accuracy']}")
            print(f"None rate: {result['none_rate']}")
            print(f"Wrong accepted predictions: {result['wrong_accepted_predictions']}")
            return 0

        if args.command == "predict":
            config = build_recognition_config(args)
            try:
                prediction = predict_champion(Path(args.image), config)
            except ChampionRecognitionError as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1

            print("None" if prediction is None else prediction)
            return 0

        if args.command == "post-processing-ai":
            config = build_recognition_config(args)
            try:
                result = analyze_post_training(config)
            except ChampionRecognitionError as exc:
                logging.getLogger(__name__).error("%s", exc)
                return 1

            print(f"Saved post-processing report: {result.report_path}")
            print(f"Saved threshold sweep: {result.threshold_report_path}")
            print(f"Current top-1 accuracy: {result.current_top1_accuracy}")
            print(f"Current accepted accuracy: {result.current_accepted_accuracy}")
            print(f"Current none rate: {result.current_none_rate}")
            print(f"Recommended similarity threshold: {result.recommended_similarity_threshold}")
            print(f"Recommended margin threshold: {result.recommended_margin_threshold}")
            return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


def build_recognition_config(args: argparse.Namespace):
    """Build the prototype recognizer config from CLI args."""
    from raid_bot.champion_recognition import RecognitionConfig

    data_dir = Path(args.data_dir)
    processed_dir = data_dir / "processed"
    return RecognitionConfig.from_data_dir(
        data_dir,
        train_image_folder=Path(getattr(args, "train_image_folder", None) or processed_dir / "icons_noised"),
        reference_icon_folder=Path(getattr(args, "reference_icon_folder", None) or processed_dir / "icons"),
        labels_csv_path=Path(getattr(args, "labels_csv", None) or processed_dir / "labels.csv"),
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=getattr(args, "batch_size", 256),
        learning_rate=getattr(args, "learning_rate", 1e-3),
        fine_tune_learning_rate=getattr(args, "fine_tune_learning_rate", 1e-5),
        epochs=getattr(args, "epochs", 10),
        fine_tune_epochs=getattr(args, "fine_tune_epochs", 0),
        unfreeze_last_blocks=getattr(args, "unfreeze_last_blocks", 1),
        checkpoint_path=Path(args.checkpoint_path),
        prototype_path=Path(args.prototype_path),
        log_dir=Path(args.log_dir),
        plot_dir=Path(args.plot_dir),
        review_samples_dir=Path(args.review_samples_dir),
        random_seed=getattr(args, "seed", 42),
        benchmark_class_count=getattr(args, "benchmark_class_count", 0),
        benchmark_seed=getattr(args, "benchmark_seed", 42),
        min_similarity_threshold=args.min_similarity_threshold,
        min_margin_threshold=args.min_margin_threshold,
        pretrained=not getattr(args, "no_pretrained", False),
        use_balanced_sampling=not getattr(args, "no_balanced_sampling", False),
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
