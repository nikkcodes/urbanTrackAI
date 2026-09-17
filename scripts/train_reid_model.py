"""Fine-tune OSNet x0.25 on the AI City Re-ID dataset using TorchReID.

Usage:
    python scripts/train_reid_model.py --epochs 25
    python scripts/train_reid_model.py --epochs 2  # validation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

# Allow running from repository root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torchreid

DEFAULT_DATASET_DIR = "data/reid_dataset"
DEFAULT_OUTPUT_DIR = "models/reid"
DEFAULT_MODEL_NAME = "osnet_x0_25"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128


def format_duration(seconds: float) -> str:
    """Format duration in seconds as HH:MM:SS."""
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--:--"
    total_sec = int(seconds)
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def validate_dataset(dataset_dir: Path) -> dict:
    """Validate that the Re-ID dataset exists and contains valid data."""
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Error: Dataset manifest not found at '{manifest_path}'. "
            "Please run scripts/build_reid_dataset.py first."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    total_ids = manifest.get("total_identities", 0)
    total_imgs = manifest.get("total_images", 0)
    if total_ids <= 0 or total_imgs <= 0:
        raise SystemExit(
            f"Error: Dataset in '{dataset_dir}' has zero identities ({total_ids}) "
            f"or zero images ({total_imgs})."
        )

    return manifest


class AICityReIDDataset(torchreid.data.ImageDataset):
    """Custom TorchReID ImageDataset for AI City vehicle crops."""

    def __init__(self, root: str = DEFAULT_DATASET_DIR, **kwargs) -> None:
        root_path = Path(root)
        train_dir = root_path / "train"
        val_dir = root_path / "validation"

        if not train_dir.is_dir():
            raise RuntimeError(f"Train directory not found: {train_dir}")

        train_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        val_folders = (
            sorted([d for d in val_dir.iterdir() if d.is_dir()])
            if val_dir.is_dir()
            else []
        )

        all_identities = sorted(
            list(set([d.name for d in train_folders] + [d.name for d in val_folders]))
        )
        pid_map = {name: idx for idx, name in enumerate(all_identities)}

        # Map camera names to camera IDs
        cam_names = set()
        for f in train_folders + val_folders:
            parts = f.name.split("_")
            if len(parts) >= 3:
                cam_names.add(f"{parts[0]}_{parts[1]}_{parts[2]}")
            else:
                cam_names.add("CAM_000")
        cam_map = {name: idx for idx, name in enumerate(sorted(list(cam_names)))}

        train_data: list[tuple[str, int, int]] = []
        query_data: list[tuple[str, int, int]] = []
        gallery_data: list[tuple[str, int, int]] = []

        if val_folders:
            # When validation split exists, use train for training and validation for test
            for folder in train_folders:
                pid = pid_map[folder.name]
                cam_prefix = "_".join(folder.name.split("_")[:3])
                camid = cam_map.get(cam_prefix, 0)
                for img_path in sorted(folder.glob("*.jpg")):
                    train_data.append((str(img_path), pid, camid))

            for folder in val_folders:
                pid = pid_map[folder.name]
                cam_prefix = "_".join(folder.name.split("_")[:3])
                camid = cam_map.get(cam_prefix, 0)
                imgs = sorted(folder.glob("*.jpg"))
                if len(imgs) >= 2:
                    query_data.append((str(imgs[0]), pid, camid * 2))
                    for img in imgs[1:]:
                        gallery_data.append((str(img), pid, camid * 2 + 1))
                elif len(imgs) == 1:
                    query_data.append((str(imgs[0]), pid, camid * 2))
                    gallery_data.append((str(imgs[0]), pid, camid * 2 + 1))
        else:
            # Hold out test crops from train tracks for evaluation
            for folder in train_folders:
                pid = pid_map[folder.name]
                cam_prefix = "_".join(folder.name.split("_")[:3])
                camid = cam_map.get(cam_prefix, 0)
                imgs = sorted(folder.glob("*.jpg"))
                if len(imgs) >= 3:
                    query_data.append((str(imgs[-1]), pid, camid * 2))
                    gallery_data.append((str(imgs[-2]), pid, camid * 2 + 1))
                    for img in imgs[:-2]:
                        train_data.append((str(img), pid, camid))
                elif len(imgs) == 2:
                    query_data.append((str(imgs[-1]), pid, camid * 2))
                    gallery_data.append((str(imgs[-2]), pid, camid * 2 + 1))
                    for img in imgs:
                        train_data.append((str(img), pid, camid))
                elif len(imgs) == 1:
                    train_data.append((str(imgs[0]), pid, camid))
                    query_data.append((str(imgs[0]), pid, camid * 2))
                    gallery_data.append((str(imgs[0]), pid, camid * 2 + 1))

        super().__init__(train_data, query_data, gallery_data, **kwargs)


class AMPEngine(torchreid.engine.ImageSoftmaxEngine):
    """TorchReID ImageSoftmaxEngine with Automatic Mixed Precision (AMP)."""

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
        use_amp=True,
        **kwargs,
    ) -> None:
        super().__init__(
            datamanager,
            model,
            optimizer,
            scheduler=scheduler,
            use_gpu=use_gpu,
            **kwargs,
        )
        self.use_amp = use_amp and self.use_gpu
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.last_epoch_loss = 0.0

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if self.use_amp:
            try:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(imgs)
                    loss = self.compute_loss(self.criterion, outputs, pids)
            except Exception:
                with torch.cuda.amp.autocast():
                    outputs = self.model(imgs)
                    loss = self.compute_loss(self.criterion, outputs, pids)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(imgs)
            loss = self.compute_loss(self.criterion, outputs, pids)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": torchreid.metrics.accuracy(outputs, pids)[0].item(),
        }
        return loss_summary

    def train_epoch(self) -> float:
        """Run one training epoch and return average loss."""
        losses = torchreid.utils.MetricMeter()
        self.set_model_mode("train")

        for self.batch_idx, data in enumerate(self.train_loader):
            loss_summary = self.forward_backward(data)
            losses.update(loss_summary)

        self.update_lr()
        avg_loss = float(losses.meters["loss"].avg)
        self.last_epoch_loss = avg_loss
        return avg_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune OSNet x0.25 on AI City Re-ID dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs. Default: 25.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training and testing. Default: 32.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of DataLoader workers. Default: 4.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        help="Learning rate for Adam optimizer. Default: 0.0003.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing weights and outputs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device to train on. Default: auto (CUDA if available).",
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Path to Re-ID dataset. Default: data/reid_dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save checkpoints and metrics. Default: models/reid.",
    )

    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate dataset
    validate_dataset(dataset_dir)

    # 2. Check output overwrite protection
    weights_path = output_dir / "osnet_x0_25_aicity.pth"
    best_weights_path = output_dir / "osnet_x0_25_aicity_best.pth"
    if not args.overwrite and not args.resume:
        if weights_path.is_file() or best_weights_path.is_file():
            raise SystemExit(
                f"Error: Model weights already exist in '{output_dir}'. "
                "Use --overwrite to overwrite existing files."
            )

    # 3. Device & AMP selection
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device_str = "cpu"
        else:
            device_str = "cuda"
    elif args.device == "cpu":
        device_str = "cpu"
    else:  # "auto"
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    use_gpu = device_str == "cuda"
    use_amp = use_gpu  # Automatically enable AMP when CUDA is available

    print(f"Device selected: {device_str} (AMP enabled: {use_amp})")

    # 4. Export training_config.yaml
    config_data = {
        "model": DEFAULT_MODEL_NAME,
        "optimizer": "adam",
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "workers": args.workers,
        "dataset_path": str(dataset_dir).replace("\\", "/"),
        "device": device_str,
        "amp_enabled": use_amp,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
    }
    config_file = output_dir / "training_config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

    # 5. Register dataset and build ImageDataManager
    dataset_name = "aicity_reid"
    try:
        torchreid.data.register_image_dataset(dataset_name, AICityReIDDataset)
    except ValueError:
        pass  # Already registered

    datamanager = torchreid.data.ImageDataManager(
        root=str(dataset_dir),
        sources=dataset_name,
        targets=dataset_name,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        batch_size_train=args.batch_size,
        batch_size_test=args.batch_size,
        workers=args.workers,
        use_gpu=use_gpu,
    )

    # 6. Build model
    model = torchreid.models.build_model(
        name=DEFAULT_MODEL_NAME,
        num_classes=datamanager.num_train_pids,
        loss="softmax",
        pretrained=True,
    )
    if use_gpu:
        model = model.cuda()

    # 7. Optimizer & LR Scheduler
    optimizer = torchreid.optim.build_optimizer(model, optim="adam", lr=args.lr)
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="single_step", stepsize=20
    )

    # 8. Resume if requested
    start_epoch = 0
    best_rank1 = 0.0
    best_map = 0.0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise SystemExit(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device_str)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
            if "optimizer" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                except Exception:
                    pass
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_rank1 = checkpoint.get("rank1", 0.0)
            best_map = checkpoint.get("mAP", 0.0)
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed from checkpoint: {resume_path} at epoch {start_epoch}")

    # 9. Initialize Engine
    engine = AMPEngine(
        datamanager=datamanager,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        use_gpu=use_gpu,
        use_amp=use_amp,
    )

    # 10. Training Loop
    history_records: list[dict] = []
    total_start_time = time.time()

    print("\nStarting fine-tuning OSNet on AI City Re-ID dataset...")
    for epoch in range(start_epoch, args.epochs):
        engine.epoch = epoch
        epoch_start = time.time()

        # Run one epoch of training
        loss = engine.train_epoch()

        # Evaluate on query & gallery
        rank1, map_val = engine._evaluate(
            dataset_name=dataset_name,
            query_loader=datamanager.test_loader[dataset_name]["query"],
            gallery_loader=datamanager.test_loader[dataset_name]["gallery"],
            dist_metric="euclidean",
            normalize_feature=False,
            visrank=False,
        )

        epoch_time = time.time() - epoch_start
        lr = engine.get_current_lr()

        # Format console progress
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"Loss: {loss:.4f}")
        print(f"Rank-1: {rank1 * 100:.1f}%")
        print(f"mAP: {map_val * 100:.1f}%")
        print(f"LR: {lr:.6f}")
        print(f"Time: {format_duration(epoch_time)}")

        is_best = rank1 >= best_rank1
        if is_best:
            best_rank1 = rank1
            best_map = map_val

        # Save checkpoint
        checkpoint_state = {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "rank1": float(rank1),
            "mAP": float(map_val),
            "optimizer": optimizer.state_dict(),
            "num_classes": datamanager.num_train_pids,
            "model_name": DEFAULT_MODEL_NAME,
        }
        torch.save(checkpoint_state, weights_path)
        if is_best:
            torch.save(checkpoint_state, best_weights_path)

        # Record history
        history_records.append(
            {
                "epoch": epoch + 1,
                "train_loss": round(float(loss), 4),
                "learning_rate": round(float(lr), 6),
                "rank1_accuracy": round(float(rank1), 4),
                "mAP": round(float(map_val), 4),
            }
        )

    total_training_time = time.time() - total_start_time

    # 11. Write training_history.csv
    history_file = output_dir / "training_history.csv"
    with history_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "learning_rate", "rank1_accuracy", "mAP"],
        )
        writer.writeheader()
        for row in history_records:
            writer.writerow(row)

    # 12. Write training_metrics.json
    metrics_data = {
        "model_name": DEFAULT_MODEL_NAME,
        "dataset": "AI City Challenge 2022 Track 1",
        "epochs_completed": args.epochs,
        "batch_size": args.batch_size,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "device": device_str,
        "mixed_precision": use_amp,
        "best_rank1_accuracy": round(float(best_rank1), 4),
        "best_map": round(float(best_map), 4),
        "training_time_seconds": round(float(total_training_time), 2),
        "weights_file": "models/reid/osnet_x0_25_aicity_best.pth",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    metrics_file = output_dir / "training_metrics.json"
    with metrics_file.open("w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # 13. Completion Summary
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Total training time: {format_duration(total_training_time)}")
    print(f"Best Rank-1: {best_rank1 * 100:.1f}%")
    print(f"Best mAP: {best_map * 100:.1f}%")
    print(f"Saved checkpoint path: {best_weights_path}")
    print(f"History written to: {history_file}")
    print(f"Metrics written to: {metrics_file}")
    print(f"Config written to: {config_file}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
