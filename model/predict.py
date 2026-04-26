from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageFile

try:
    from .train_defactify import SOURCE_NAMES, build_model, build_transforms
except ImportError:
    from train_defactify import SOURCE_NAMES, build_model, build_transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained Defactify checkpoint on one image.")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--checkpoint", default="runs/defactify_binary/best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)
    if not image_path.is_file():
        raise SystemExit(
            f"Image file not found: {image_path}\n"
            "请把 path\\to\\image.jpg 换成真实图片路径，例如：\n"
            "python -m model.predict .\\samples\\test.jpg --checkpoint runs\\defactify_binary\\best.pt"
        )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    task = config.get("task", "binary")
    num_classes = int(config.get("num_classes", 2 if task == "binary" else 6))
    image_size = int(config.get("image_size", 224))
    model_name = config.get("model", "efficientnet_b0")

    device = torch.device(args.device)
    model = build_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    transform = build_transforms(image_size, train=False, jpeg_prob=0.0)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    if task == "binary":
        result = {
            "task": "binary",
            "label": "AI-Generated" if int(probs.argmax()) == 1 else "Real",
            "ai_probability": float(probs[1]),
            "real_probability": float(probs[0]),
        }
    else:
        result = {
            "task": "source",
            "label": SOURCE_NAMES.get(int(probs.argmax()), str(int(probs.argmax()))),
            "probabilities": {
                SOURCE_NAMES.get(index, str(index)): float(value)
                for index, value in enumerate(probs.tolist())
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
