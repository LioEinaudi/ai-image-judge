from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_ID = "Rajarshi-Roy-research/Defactify_Image_Dataset"
SOURCE_NAMES = {
    0: "Real",
    1: "SD21",
    2: "SDXL",
    3: "SD3",
    4: "DALLE3",
    5: "Midjourney",
}


class RandomJpegCompression:
    def __init__(self, probability: float = 0.2, quality_min: int = 45, quality_max: int = 95):
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.probability:
            return image
        buffer = io.BytesIO()
        quality = random.randint(self.quality_min, self.quality_max)
        image.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class HuggingFaceImageDataset(Dataset):
    def __init__(
        self,
        split,
        transform,
        task: str,
        max_samples: int | None = None,
    ):
        if max_samples:
            split = split.select(range(min(max_samples, len(split))))
        self.split = split
        self.transform = transform
        self.task = task

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.split[index]
        image = item["Image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
        label_a = int(item["Label_A"])
        label_b = int(item["Label_B"])
        label = label_a if self.task == "binary" else label_b
        return {
            "image": self.transform(image),
            "label": torch.tensor(label, dtype=torch.long),
            "label_a": torch.tensor(label_a, dtype=torch.long),
            "label_b": torch.tensor(label_b, dtype=torch.long),
        }


def build_transforms(image_size: int, train: bool, jpeg_prob: float) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                RandomJpegCompression(probability=jpeg_prob),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if model_name == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def labels_from_split(split, task: str, max_samples: int | None) -> list[int]:
    total = min(max_samples, len(split)) if max_samples else len(split)
    column = "Label_A" if task == "binary" else "Label_B"
    return [int(split[i][column]) for i in range(total)]


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean()


def make_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    per_class = 1.0 / counts.clamp_min(1.0)
    sample_weights = [float(per_class[label]) for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.long()
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32, device=scores.device)
    pos_rank_sum = ranks[positives].sum()
    auc = (pos_rank_sum - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return float(auc)


def fpr_at_tpr(scores: torch.Tensor, labels: torch.Tensor, target_tpr: float = 0.95) -> float:
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return float("nan")
    thresholds = torch.unique(scores).sort(descending=True).values
    best_fpr = 1.0
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int((pred & positives).sum())
        fp = int((pred & negatives).sum())
        tpr = tp / pos_count
        if tpr >= target_tpr:
            best_fpr = min(best_fpr, fp / neg_count)
    return float(best_fpr)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_b: torch.Tensor,
    task: str,
    loss: float,
) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean())
    result: dict[str, Any] = {"loss": loss, "accuracy": accuracy}

    if task == "binary":
        probs = torch.softmax(logits, dim=1)[:, 1]
        tp = int(((predictions == 1) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        result.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fpr": fp / max(fp + tn, 1),
                "fnr": fn / max(fn + tp, 1),
                "auroc": binary_auroc(probs, labels),
                "fpr_at_95_tpr": fpr_at_tpr(probs, labels, 0.95),
                "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
                "by_source": {},
            }
        )
        for source_id, source_name in SOURCE_NAMES.items():
            mask = label_b == source_id
            if int(mask.sum()) == 0:
                continue
            result["by_source"][source_name] = {
                "count": int(mask.sum()),
                "accuracy": float((predictions[mask] == labels[mask]).float().mean()),
                "mean_ai_probability": float(probs[mask].mean()),
            }
    else:
        result["by_source"] = {}
        for source_id, source_name in SOURCE_NAMES.items():
            mask = labels == source_id
            if int(mask.sum()) == 0:
                continue
            result["by_source"][source_name] = {
                "count": int(mask.sum()),
                "accuracy": float((predictions[mask] == labels[mask]).float().mean()),
            }
    return result


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    task: str,
    amp: bool,
    desc: str,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    seen = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_label_b: list[torch.Tensor] = []
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        label_b = batch["label_b"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = images.size(0)
        total_loss += float(loss.detach()) * batch_size
        seen += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        all_label_b.append(label_b.detach().cpu())

    logits_cpu = torch.cat(all_logits)
    labels_cpu = torch.cat(all_labels)
    label_b_cpu = torch.cat(all_label_b)
    return compute_metrics(logits_cpu, labels_cpu, label_b_cpu, task, total_loss / max(seen, 1))


def save_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "by_source"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key != "by_source"}
            writer.writerow(flat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AI image detector on Defactify.")
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--output-dir", default="runs/defactify_binary")
    parser.add_argument("--task", choices=["binary", "source"], default="binary")
    parser.add_argument(
        "--model",
        choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"],
        default="efficientnet_b0",
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-augment-prob", type=float, default=0.25)
    parser.add_argument("--no-balance-sampler", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--cache-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    num_classes = 2 if args.task == "binary" else 6

    print(f"Loading dataset: {args.dataset_id}")
    dataset = load_dataset(args.dataset_id, cache_dir=args.cache_dir)
    train_labels = labels_from_split(dataset["train"], args.task, args.max_train_samples)
    val_split_name = "validation" if "validation" in dataset else "test"

    train_ds = HuggingFaceImageDataset(
        dataset["train"],
        build_transforms(args.image_size, train=True, jpeg_prob=args.jpeg_augment_prob),
        task=args.task,
        max_samples=args.max_train_samples,
    )
    val_ds = HuggingFaceImageDataset(
        dataset[val_split_name],
        build_transforms(args.image_size, train=False, jpeg_prob=0.0),
        task=args.task,
        max_samples=args.max_val_samples,
    )

    sampler = None
    shuffle = True
    if not args.no_balance_sampler:
        sampler = make_sampler(train_labels, num_classes)
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained).to(device)
    weights = None
    if not args.no_class_weights:
        weights = class_weights(train_labels, num_classes).to(device)
        print(f"Class weights: {[round(float(x), 4) for x in weights.cpu()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    config = vars(args) | {
        "device": str(device),
        "num_classes": num_classes,
        "source_names": SOURCE_NAMES,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, args.task, amp, f"train {epoch}"
        )
        scheduler.step()
        val_metrics = run_epoch(
            model, val_loader, criterion, None, device, args.task, amp, f"val {epoch}"
        )

        score = val_metrics.get("auroc", val_metrics["accuracy"])
        if isinstance(score, float) and math.isnan(score):
            score = val_metrics["accuracy"]
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_metrics.items() if k != "by_source"},
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "by_source"},
        }
        history.append(row)
        save_metrics(output_dir / "metrics.csv", history)
        (output_dir / "latest_metrics.json").write_text(
            json.dumps({"train": train_metrics, "validation": val_metrics}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(json.dumps({"train": train_metrics, "validation": val_metrics}, ensure_ascii=False, indent=2))

        checkpoint = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "validation": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if score > best_score:
            best_score = float(score)
            torch.save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path} ({score:.4f})")

    if args.eval_test and "test" in dataset:
        test_ds = HuggingFaceImageDataset(
            dataset["test"],
            build_transforms(args.image_size, train=False, jpeg_prob=0.0),
            task=args.task,
            max_samples=args.max_test_samples,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(
            model, test_loader, criterion, None, device, args.task, amp, "test"
        )
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nTest metrics")
        print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
