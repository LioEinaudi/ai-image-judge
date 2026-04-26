# AI Image Judge

AI Image Judge is a local image-forensics MVP for checking whether an image may be AI-generated or AI-edited.

The project combines provenance checks, metadata inspection, weak pixel forensics, and an optional trained classifier. It is designed to report evidence and uncertainty instead of pretending that image detection is perfectly reliable.

## Current Capabilities

- C2PA / Content Credentials inspection through `c2patool` when installed.
- EXIF / XMP / text metadata inspection through `exiftool` when installed, plus built-in basic JPEG and PNG parsing.
- File-level evidence: SHA256, image type, dimensions, and container structure.
- Pixel-forensics heuristics: high-frequency energy, edge density, residual distribution, and JPEG grid artifacts.
- Optional machine-learning classifier trained on the Defactify image dataset.
- Evidence-based verdicts with confidence and warnings.

With the trained Defactify baseline, the system can recognize many AI images produced by common model families covered by the dataset, such as Stable Diffusion variants, DALL-E 3, and Midjourney. It currently cannot reliably identify GPT-image-2 images.

## Important Limits

This is not a universal AI-image detector.

Missing C2PA, EXIF, or camera metadata does not prove that an image is AI-generated. Screenshots, WeChat forwarding, social-media uploads, recompression, and normal editing can remove or rewrite metadata.

The trained model score is only a probability-like signal under the training distribution. It should be combined with C2PA, metadata, source-chain evidence, and human review.

## Run the Web App

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Start the local server:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8765
```

Optional environment variables:

```powershell
$env:HOST="127.0.0.1"
$env:PORT="8765"
python app.py
```

If `runs/defactify_binary/best.pt` exists, the web app and `/api/analyze` automatically load it and include `model.ai_probability` in the report. You can also specify a checkpoint manually:

```powershell
$env:MODEL_CHECKPOINT="D:\project\ai-image-judge\runs\defactify_binary\best.pt"
python app.py
```

## Optional External Tools

The app works without these tools, but the evidence is much stronger when they are installed.

### c2patool

Used to read and verify C2PA / Content Credentials manifests.

```bash
c2patool image.jpg --json
```

### exiftool

Used to read fuller EXIF, XMP, and MakerNotes metadata.

```bash
exiftool -j -n image.jpg
```

## Verdict Labels

- `confirmed_ai_generated`: a valid C2PA manifest and AI-tool provenance were found.
- `likely_ai_generated`: metadata or the trained classifier strongly suggests AI generation.
- `possibly_ai_generated`: source evidence is missing and multiple weak signals are suspicious.
- `possibly_ai_edited`: C2PA/JUMBF-like structure was found but cannot be fully verified.
- `likely_camera_originated`: camera-origin metadata and low model risk point toward a camera or phone source.
- `inconclusive`: evidence is insufficient.

## API

Health check:

```bash
curl http://127.0.0.1:8765/api/health
```

Analyze an image:

```bash
curl -F "image=@image.jpg" http://127.0.0.1:8765/api/analyze
```

The response includes:

- `sha256`
- `image_type`
- `dimensions`
- `verdict`
- `metadata`
- `model`

`metadata.visual_forensics` contains weak pixel-forensics features. `model.ai_probability`, when available, comes from the trained Defactify baseline and should not be treated as absolute truth.

## Model Files and Git

Model checkpoints are not committed to git. The `runs/` directory is ignored by `.gitignore` because checkpoints and training outputs can become large and environment-specific.

To use the model, train it locally and keep the checkpoint at:

```text
runs/defactify_binary/best.pt
```

or set:

```powershell
$env:MODEL_CHECKPOINT="path\to\best.pt"
```

## Train the Defactify Baseline

This project includes a training script for the Hugging Face dataset:

```text
Rajarshi-Roy-research/Defactify_Image_Dataset
```

Dataset fields:

- `Image`: image data.
- `Caption`: caption or prompt text.
- `Label_A`: binary label, `0 = Real`, `1 = AI-Generated`.
- `Label_B`: source label, `0 = Real`, `1 = SD21`, `2 = SDXL`, `3 = SD3`, `4 = DALLE3`, `5 = Midjourney`.

The full dataset download is about 7.5GB.

### Smoke Test

```powershell
python -m model.train_defactify `
  --max-train-samples 200 `
  --max-val-samples 100 `
  --epochs 1 `
  --batch-size 8 `
  --model resnet18 `
  --output-dir runs/smoke
```

### Binary Classifier Training

Recommended when using an NVIDIA GPU:

```powershell
python -m model.train_defactify `
  --task binary `
  --model efficientnet_b0 `
  --pretrained `
  --epochs 8 `
  --batch-size 32 `
  --lr 3e-4 `
  --eval-test `
  --output-dir runs/defactify_binary
```

Smaller CPU-friendly run:

```powershell
python -m model.train_defactify `
  --task binary `
  --model resnet18 `
  --epochs 3 `
  --batch-size 8 `
  --max-train-samples 5000 `
  --max-val-samples 1000 `
  --output-dir runs/defactify_cpu_baseline
```

Training outputs:

```text
runs/defactify_binary/
  best.pt
  last.pt
  config.json
  metrics.csv
  latest_metrics.json
  test_metrics.json
```

### Source Classifier Training

To train a six-class source classifier:

```powershell
python -m model.train_defactify `
  --task source `
  --model efficientnet_b0 `
  --pretrained `
  --epochs 8 `
  --batch-size 32 `
  --eval-test `
  --output-dir runs/defactify_source
```

### Single-Image Prediction

After training:

```powershell
python -m model.predict "path\to\image.jpg" --checkpoint runs\defactify_binary\best.pt
```

Example output:

```json
{
  "task": "binary",
  "label": "AI-Generated",
  "ai_probability": 0.83,
  "real_probability": 0.17
}
```

## Training Notes

The Defactify dataset is useful for a first baseline, but it should not be treated as a complete real-world benchmark. It mainly covers SD21, SDXL, SD3, DALL-E 3, and Midjourney. Images from GPT-image-2, Doubao, Flux, local inpainting, AI upscaling, screenshots, or WeChat recompression may still be misclassified.

The training script reports:

- Accuracy
- Precision / Recall / F1
- AUROC
- FPR / FNR
- FPR@95%TPR
- Per-source accuracy and average AI probability based on `Label_B`

Pay special attention to `by_source` in `latest_metrics.json`. If accuracy is low for one source, the model is not generalizing well to that generator.

For a stronger product, add more data:

- Real and AI images after WeChat recompression.
- Screenshots.
- Original phone and camera photos.
- Photoshop / Lightroom edited real photos.
- Doubao, Flux, Ideogram, GPT-image-2, and other newer generators.
- Image-to-image, inpainting, outpainting, and super-resolution examples.

The product should continue to present model output as probabilistic evidence, not as a final truth label.
