from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_MODEL_CHECKPOINT = ROOT / "runs" / "defactify_binary" / "best.pt"
MODEL_CACHE: dict[str, Any] = {}

AI_TOOL_PATTERNS = [
    "openai",
    "chatgpt",
    "dall-e",
    "dalle",
    "midjourney",
    "stable diffusion",
    "sdxl",
    "firefly",
    "ideogram",
    "leonardo",
    "flux",
    "comfyui",
    "automatic1111",
    "invokeai",
    "novelai",
]

CAMERA_HINT_TAGS = ("make", "model", "lensmodel", "datetimeoriginal")
TEXT_TAG_LIMIT = 80


def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def public_error(message: str, status: int = 400) -> tuple[int, dict[str, Any]]:
    return status, {"error": message}


def configured_model_path() -> Path:
    value = os.environ.get("MODEL_CHECKPOINT")
    return Path(value).expanduser().resolve() if value else DEFAULT_MODEL_CHECKPOINT


def sniff_image_type(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_printable_strings(data: bytes) -> list[str]:
    strings = re.findall(rb"[\x20-\x7e]{4,}", data)
    decoded = []
    seen = set()
    for item in strings:
        text = item.decode("latin-1", errors="ignore").strip()
        if text and text not in seen:
            seen.add(text)
            decoded.append(text)
        if len(decoded) >= TEXT_TAG_LIMIT:
            break
    return decoded


def scan_keywords(texts: list[str]) -> list[str]:
    found: list[str] = []
    body = "\n".join(texts).lower()
    for pattern in AI_TOOL_PATTERNS:
        if pattern in body:
            found.append(pattern)
    return found


def parse_jpeg_dimensions(data: bytes) -> dict[str, int] | None:
    if not data.startswith(b"\xff\xd8"):
        return None
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset : offset + 2])[0]
        segment_start = offset + 2
        segment_end = segment_start + length - 2
        if segment_end > len(data):
            break
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            segment = data[segment_start:segment_end]
            if len(segment) >= 5:
                height, width = struct.unpack(">HH", segment[1:5])
                return {"width": width, "height": height}
        offset = segment_end
    return None


def parse_png_dimensions(data: bytes) -> dict[str, int] | None:
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 24:
        return None
    width, height = struct.unpack(">II", data[16:24])
    return {"width": width, "height": height}


def parse_gif_dimensions(data: bytes) -> dict[str, int] | None:
    if not (data.startswith(b"GIF87a") or data.startswith(b"GIF89a")) or len(data) < 10:
        return None
    width, height = struct.unpack("<HH", data[6:10])
    return {"width": width, "height": height}


def parse_webp_dimensions(data: bytes) -> dict[str, int] | None:
    if not (data.startswith(b"RIFF") and data[8:12] == b"WEBP") or len(data) < 30:
        return None
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return {"width": width, "height": height}
    if chunk == b"VP8 " and len(data) >= 30:
        marker = data.find(b"\x9d\x01\x2a", 20)
        if marker != -1 and marker + 7 <= len(data):
            width, height = struct.unpack("<HH", data[marker + 3 : marker + 7])
            return {"width": width & 0x3FFF, "height": height & 0x3FFF}
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return {"width": width, "height": height}
    return None


def parse_dimensions(data: bytes, image_type: str) -> dict[str, int] | None:
    if image_type == "jpeg":
        return parse_jpeg_dimensions(data)
    if image_type == "png":
        return parse_png_dimensions(data)
    if image_type == "gif":
        return parse_gif_dimensions(data)
    if image_type == "webp":
        return parse_webp_dimensions(data)
    return None


def parse_jpeg_segments(data: bytes) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not data.startswith(b"\xff\xd8"):
        return segments
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker == 0xDA:
            break
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        length = struct.unpack(">H", data[offset : offset + 2])[0]
        start = offset + 2
        end = start + length - 2
        if length < 2 or end > len(data):
            break
        payload = data[start:end]
        name = f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else f"0x{marker:02X}"
        segments.append({"marker": marker, "name": name, "payload": payload})
        offset = end
    return segments


def read_tiff_value(
    tiff: bytes, endian: str, field_type: int, count: int, value_offset: int
) -> str | int | None:
    type_sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4}
    size = type_sizes.get(field_type)
    if not size:
        return None
    total = size * count
    raw = value_offset.to_bytes(4, "little" if endian == "<" else "big")
    if total <= 4:
        data = raw[:total]
    else:
        if value_offset < 0 or value_offset + total > len(tiff):
            return None
        data = tiff[value_offset : value_offset + total]
    try:
        if field_type == 2:
            return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        if field_type == 3 and count == 1:
            return struct.unpack(endian + "H", data[:2])[0]
        if field_type in (4, 9) and count == 1:
            return struct.unpack(endian + ("I" if field_type == 4 else "i"), data[:4])[0]
        if field_type == 7:
            return data.hex()
    except struct.error:
        return None
    return None


def parse_ifd(tiff: bytes, endian: str, offset: int) -> tuple[dict[int, Any], int | None]:
    tags: dict[int, Any] = {}
    if offset < 0 or offset + 2 > len(tiff):
        return tags, None
    count = struct.unpack(endian + "H", tiff[offset : offset + 2])[0]
    cursor = offset + 2
    for _ in range(count):
        if cursor + 12 > len(tiff):
            break
        tag, field_type, item_count, value_offset = struct.unpack(
            endian + "HHII", tiff[cursor : cursor + 12]
        )
        tags[tag] = read_tiff_value(tiff, endian, field_type, item_count, value_offset)
        cursor += 12
    next_ifd = None
    if cursor + 4 <= len(tiff):
        next_ifd = struct.unpack(endian + "I", tiff[cursor : cursor + 4])[0]
    return tags, next_ifd


def parse_exif_payload(payload: bytes) -> dict[str, Any]:
    if not payload.startswith(b"Exif\x00\x00"):
        return {}
    tiff = payload[6:]
    if len(tiff) < 8:
        return {}
    byte_order = tiff[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return {}
    magic = struct.unpack(endian + "H", tiff[2:4])[0]
    if magic != 42:
        return {}
    first_ifd = struct.unpack(endian + "I", tiff[4:8])[0]
    tags, _ = parse_ifd(tiff, endian, first_ifd)
    exif_ifd_offset = tags.get(0x8769)
    exif_tags: dict[int, Any] = {}
    if isinstance(exif_ifd_offset, int):
        exif_tags, _ = parse_ifd(tiff, endian, exif_ifd_offset)
    mapping = {
        0x010E: "image_description",
        0x010F: "make",
        0x0110: "model",
        0x0131: "software",
        0x0132: "datetime",
        0x013B: "artist",
        0x8298: "copyright",
    }
    exif_mapping = {
        0x9003: "datetime_original",
        0x9004: "datetime_digitized",
        0xA434: "lens_model",
    }
    result: dict[str, Any] = {}
    for tag, name in mapping.items():
        value = tags.get(tag)
        if value not in (None, ""):
            result[name] = value
    for tag, name in exif_mapping.items():
        value = exif_tags.get(tag)
        if value not in (None, ""):
            result[name] = value
    return result


def parse_jpeg_metadata(data: bytes) -> dict[str, Any]:
    exif: dict[str, Any] = {}
    xmp_blocks: list[str] = []
    text_samples: list[str] = []
    c2pa_hints: list[str] = []
    segments = parse_jpeg_segments(data)
    for segment in segments:
        payload = segment["payload"]
        if segment["name"] == "APP1" and payload.startswith(b"Exif\x00\x00"):
            exif.update(parse_exif_payload(payload))
        if b"http://ns.adobe.com/xap/1.0/" in payload or b"<x:xmpmeta" in payload:
            xmp = payload.split(b"\x00", 1)[-1].decode("utf-8", errors="replace")
            xmp_blocks.append(xmp[:4000])
        if b"c2pa" in payload.lower() or b"content authenticity" in payload.lower():
            c2pa_hints.append(f"{segment['name']} contains C2PA/JUMBF-like data")
        text_samples.extend(extract_printable_strings(payload))
    return {
        "exif": exif,
        "xmp": xmp_blocks[:3],
        "text_samples": text_samples[:TEXT_TAG_LIMIT],
        "c2pa_hints": c2pa_hints,
        "container": {
            "jpeg_segments": [
                {"name": item["name"], "bytes": len(item["payload"])} for item in segments[:40]
            ]
        },
    }


def parse_png_chunks(data: bytes) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return chunks
    offset = 8
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8].decode("latin-1", errors="replace")
        start = offset + 8
        end = start + length
        if end + 4 > len(data):
            break
        payload = data[start:end]
        chunks.append({"type": chunk_type, "payload": payload})
        offset = end + 4
        if chunk_type == "IEND":
            break
    return chunks


def parse_png_metadata(data: bytes) -> dict[str, Any]:
    chunks = parse_png_chunks(data)
    text: dict[str, str] = {}
    xmp_blocks: list[str] = []
    c2pa_hints: list[str] = []
    for chunk in chunks:
        kind = chunk["type"]
        payload = chunk["payload"]
        lowered = payload.lower()
        if kind == "tEXt" and b"\x00" in payload:
            key, value = payload.split(b"\x00", 1)
            text[key.decode("latin-1", errors="replace")] = value.decode(
                "utf-8", errors="replace"
            )[:1000]
        elif kind == "zTXt" and b"\x00" in payload:
            key, rest = payload.split(b"\x00", 1)
            if len(rest) > 1:
                try:
                    value = zlib.decompress(rest[1:]).decode("utf-8", errors="replace")
                    text[key.decode("latin-1", errors="replace")] = value[:1000]
                except zlib.error:
                    pass
        elif kind == "iTXt" and b"\x00" in payload:
            parts = payload.split(b"\x00", 5)
            if len(parts) >= 6:
                key = parts[0].decode("latin-1", errors="replace")
                value = parts[5].decode("utf-8", errors="replace")
                text[key] = value[:1000]
                if "<x:xmpmeta" in value:
                    xmp_blocks.append(value[:4000])
        if kind.lower() in ("caBX".lower(), "c2pa") or b"c2pa" in lowered:
            c2pa_hints.append(f"PNG chunk {kind} contains C2PA-like data")
    return {
        "exif": {},
        "xmp": xmp_blocks[:3],
        "text": text,
        "text_samples": list(text.values())[:TEXT_TAG_LIMIT],
        "c2pa_hints": c2pa_hints,
        "container": {
            "png_chunks": [{"type": item["type"], "bytes": len(item["payload"])} for item in chunks]
        },
    }


def parse_embedded_metadata(data: bytes, image_type: str) -> dict[str, Any]:
    if image_type == "jpeg":
        return parse_jpeg_metadata(data)
    if image_type == "png":
        return parse_png_metadata(data)
    return {
        "exif": {},
        "xmp": [],
        "text_samples": extract_printable_strings(data),
        "c2pa_hints": [],
        "container": {},
    }


def run_tool(command: list[str], timeout: int = 8) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False}
    except subprocess.TimeoutExpired:
        return {"available": True, "timed_out": True}
    return {
        "available": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[:20000],
        "stderr": completed.stderr[:4000],
    }


def parse_json_output(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def run_external_checks(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "c2patool": {"available": bool(shutil.which("c2patool"))},
        "exiftool": {"available": bool(shutil.which("exiftool"))},
    }
    if result["c2patool"]["available"]:
        c2pa = run_tool(["c2patool", str(path), "--json"])
        parsed = parse_json_output(c2pa.get("stdout", ""))
        c2pa["json"] = parsed
        result["c2patool"] = c2pa
    if result["exiftool"]["available"]:
        exif = run_tool(["exiftool", "-j", "-n", str(path)])
        parsed = parse_json_output(exif.get("stdout", ""))
        exif["json"] = parsed[0] if isinstance(parsed, list) and parsed else parsed
        result["exiftool"] = exif
    return result


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(flatten_text(v) for v in value)
    return str(value)


def collect_ai_tool_hits(metadata: dict[str, Any], external: dict[str, Any]) -> list[str]:
    text_parts = [
        flatten_text(metadata.get("exif", {})),
        flatten_text(metadata.get("xmp", [])),
        flatten_text(metadata.get("text", {})),
        flatten_text(metadata.get("text_samples", [])),
    ]
    for tool_name in ("c2patool", "exiftool"):
        tool = external.get(tool_name, {})
        text_parts.append(flatten_text(tool.get("json", "")))
        text_parts.append(tool.get("stdout", ""))
    return scan_keywords(text_parts)


def has_external_c2pa_manifest(external: dict[str, Any]) -> bool:
    c2pa = external.get("c2patool", {})
    if not c2pa.get("available") or c2pa.get("exit_code") not in (0, None):
        return False
    text = flatten_text(c2pa.get("json")) or c2pa.get("stdout", "")
    lowered = text.lower()
    return any(word in lowered for word in ("claim", "manifest", "c2pa", "ingredients"))


def has_camera_origin(metadata: dict[str, Any], external: dict[str, Any]) -> bool:
    exif = dict(metadata.get("exif", {}))
    exiftool_json = external.get("exiftool", {}).get("json")
    if isinstance(exiftool_json, dict):
        for key, value in exiftool_json.items():
            exif[key.lower()] = value
    return any(key in exif and exif[key] for key in CAMERA_HINT_TAGS)


def weak_image_heuristics(
    image_type: str,
    dimensions: dict[str, int] | None,
    metadata: dict[str, Any],
    visual: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    evidence: list[str] = []
    exif = metadata.get("exif", {})
    if not exif:
        score += 0.12
        evidence.append("未发现基础 EXIF；这常见于 AI 导出、截图、微信转发或社交平台重压缩")
    if dimensions:
        width = dimensions["width"]
        height = dimensions["height"]
        if width == height and width in {512, 768, 896, 1024, 1152, 1536, 2048}:
            score += 0.14
            evidence.append(f"图片为常见生成模型方图尺寸 {width}x{height}；仅作为弱信号")
        if width % 64 == 0 and height % 64 == 0 and max(width, height) <= 2048:
            score += 0.06
            evidence.append("宽高均为 64 的倍数；这在生成图中常见，但不能单独说明问题")
    if image_type in ("png", "webp") and not exif:
        score += 0.06
        evidence.append(f"{image_type.upper()} 文件缺少相机来源信息；这更像导出图而不是相机原图")

    if visual.get("status") == "available":
        visual_score = float(visual.get("score", 0))
        score += visual_score
        evidence.extend(visual.get("evidence", []))
    elif visual.get("status") == "unavailable":
        evidence.append("未能进行像素启发式分析：" + str(visual.get("reason", "未知原因")))

    return clamp(score, 0.0, 0.72), evidence


def visual_forensics(data: bytes) -> dict[str, Any]:
    try:
        import numpy as np
        from PIL import Image, ImageOps
    except ImportError as exc:
        return {"status": "unavailable", "reason": f"缺少依赖：{exc.name}"}

    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        original_mode = image.mode
        has_alpha = original_mode in ("RGBA", "LA") or (
            original_mode == "P" and "transparency" in image.info
        )
        image = image.convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        arr = np.asarray(image).astype("float32") / 255.0
    except Exception as exc:
        return {"status": "unavailable", "reason": f"图片解码失败：{exc}"}

    if arr.ndim != 3 or arr.shape[0] < 64 or arr.shape[1] < 64:
        return {"status": "unavailable", "reason": "图片太小，像素统计没有意义"}

    gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    grad = np.sqrt(gx[:-1, :] ** 2 + gy[:, :-1] ** 2)
    edge_density = float(np.mean(grad > 0.08))
    mean_gradient = float(np.mean(grad))

    center = gray[1:-1, 1:-1]
    lap = np.abs(
        -4 * center
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    residual_mean = float(np.mean(lap))
    residual_std = float(np.std(lap))

    fft_source = gray
    if max(gray.shape) > 256:
        step_y = max(1, gray.shape[0] // 256)
        step_x = max(1, gray.shape[1] // 256)
        fft_source = gray[::step_y, ::step_x][:256, :256]
    fft_centered = fft_source - float(np.mean(fft_source))
    power = np.abs(np.fft.fftshift(np.fft.fft2(fft_centered))) ** 2
    h, w = power.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    max_radius = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    total_power = float(np.sum(power) + 1e-8)
    high_freq_ratio = float(np.sum(power[radius > max_radius * 0.42]) / total_power)
    mid_freq_ratio = float(
        np.sum(power[(radius > max_radius * 0.18) & (radius <= max_radius * 0.42)])
        / total_power
    )

    block = 16
    usable_h = (lap.shape[0] // block) * block
    usable_w = (lap.shape[1] // block) * block
    block_cv = None
    if usable_h >= block * 2 and usable_w >= block * 2:
        blocks = lap[:usable_h, :usable_w].reshape(
            usable_h // block, block, usable_w // block, block
        )
        block_means = blocks.mean(axis=(1, 3))
        block_cv = float(np.std(block_means) / (np.mean(block_means) + 1e-8))

    jpeg_grid_ratio = None
    if gray.shape[0] > 64 and gray.shape[1] > 64:
        horizontal = np.abs(np.diff(gray, axis=0))
        vertical = np.abs(np.diff(gray, axis=1))
        h_idx = np.arange(horizontal.shape[0])
        v_idx = np.arange(vertical.shape[1])
        h_grid = horizontal[h_idx % 8 == 7, :].mean() if np.any(h_idx % 8 == 7) else 0
        h_other = horizontal[h_idx % 8 != 7, :].mean() if np.any(h_idx % 8 != 7) else 1e-8
        v_grid = vertical[:, v_idx % 8 == 7].mean() if np.any(v_idx % 8 == 7) else 0
        v_other = vertical[:, v_idx % 8 != 7].mean() if np.any(v_idx % 8 != 7) else 1e-8
        jpeg_grid_ratio = float(((h_grid / (h_other + 1e-8)) + (v_grid / (v_other + 1e-8))) / 2)

    features = {
        "edge_density": round(edge_density, 4),
        "mean_gradient": round(mean_gradient, 4),
        "residual_mean": round(residual_mean, 4),
        "residual_std": round(residual_std, 4),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "mid_freq_ratio": round(mid_freq_ratio, 4),
        "block_residual_cv": round(block_cv, 4) if block_cv is not None else None,
        "jpeg_grid_ratio": round(jpeg_grid_ratio, 4) if jpeg_grid_ratio is not None else None,
        "has_alpha": has_alpha,
        "decoded_mode": original_mode,
    }

    score = 0.0
    evidence: list[str] = []
    if high_freq_ratio < 0.13 and edge_density < 0.18:
        score += 0.12
        evidence.append("像素统计显示高频细节和边缘密度偏低；这可能来自生成图、磨皮、压缩或缩放")
    if block_cv is not None and block_cv < 0.55 and residual_mean < 0.08:
        score += 0.10
        evidence.append("局部残差分布较均匀；部分生成图会出现这种弱纹理特征")
    if mid_freq_ratio > 0.42 and residual_std < 0.10:
        score += 0.08
        evidence.append("频域能量集中在中频且残差波动较低；这是弱可疑信号")
    if jpeg_grid_ratio is not None and jpeg_grid_ratio > 1.22:
        score += 0.04
        evidence.append("检测到 8x8 网格压缩痕迹；更像经过 JPEG/社交平台二次处理")
    if has_alpha:
        score += 0.04
        evidence.append("图片带透明通道；这更常见于设计/导出素材，不常见于相机原图")

    return {
        "status": "available",
        "score": round(clamp(score, 0.0, 0.28), 2),
        "features": features,
        "evidence": evidence,
    }


def load_ml_model() -> dict[str, Any]:
    checkpoint_path = configured_model_path()
    if not checkpoint_path.is_file():
        return {
            "status": "not_configured",
            "checkpoint": str(checkpoint_path),
            "message": "未找到训练好的 checkpoint；当前只使用元数据和像素启发式证据。",
        }

    cache_key = str(checkpoint_path)
    mtime = checkpoint_path.stat().st_mtime
    cached = MODEL_CACHE.get(cache_key)
    if cached and cached.get("mtime") == mtime:
        return cached["payload"]

    try:
        import torch
        from PIL import ImageFile

        from model.train_defactify import SOURCE_NAMES, build_model, build_transforms
    except Exception as exc:
        return {
            "status": "unavailable",
            "checkpoint": str(checkpoint_path),
            "message": f"模型依赖加载失败：{exc}",
        }

    try:
        device_name = os.environ.get("MODEL_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        device = torch.device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]
        task = config.get("task", "binary")
        num_classes = int(config.get("num_classes", 2 if task == "binary" else 6))
        image_size = int(config.get("image_size", 224))
        model_name = config.get("model", "efficientnet_b0")
        model = build_model(model_name, num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        transform = build_transforms(image_size, train=False, jpeg_prob=0.0)
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        payload = {
            "status": "loaded",
            "checkpoint": str(checkpoint_path),
            "device": str(device),
            "task": task,
            "num_classes": num_classes,
            "image_size": image_size,
            "model_name": model_name,
            "model": model,
            "transform": transform,
            "source_names": SOURCE_NAMES,
            "torch": torch,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "checkpoint": str(checkpoint_path),
            "message": f"模型加载失败：{exc}",
        }

    MODEL_CACHE.clear()
    MODEL_CACHE[cache_key] = {"mtime": mtime, "payload": payload}
    return payload


def ml_inference(data: bytes) -> dict[str, Any]:
    loaded = load_ml_model()
    if loaded.get("status") != "loaded":
        return loaded

    try:
        from PIL import Image

        torch = loaded["torch"]
        image = Image.open(io.BytesIO(data)).convert("RGB")
        tensor = loaded["transform"](image).unsqueeze(0).to(loaded["device"])
        with torch.no_grad():
            logits = loaded["model"](tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    except Exception as exc:
        return {
            "status": "unavailable",
            "checkpoint": loaded.get("checkpoint"),
            "message": f"模型推理失败：{exc}",
        }

    if loaded["task"] == "binary":
        ai_probability = float(probs[1])
        real_probability = float(probs[0])
        return {
            "status": "available",
            "task": "binary",
            "checkpoint": loaded["checkpoint"],
            "device": loaded["device"],
            "model_name": loaded["model_name"],
            "label": "AI-Generated" if ai_probability >= real_probability else "Real",
            "ai_probability": ai_probability,
            "real_probability": real_probability,
            "warning": "这是 Defactify 数据集分布下的模型分数，不是绝对概率。",
        }

    probabilities = {
        loaded["source_names"].get(index, str(index)): float(value)
        for index, value in enumerate(probs.tolist())
    }
    best_index = int(probs.argmax())
    return {
        "status": "available",
        "task": "source",
        "checkpoint": loaded["checkpoint"],
        "device": loaded["device"],
        "model_name": loaded["model_name"],
        "label": loaded["source_names"].get(best_index, str(best_index)),
        "probabilities": probabilities,
        "warning": "这是来源分类模型分数；未覆盖的生成器可能被归到相近类别。",
    }


def choose_verdict(
    metadata: dict[str, Any],
    external: dict[str, Any],
    image_type: str,
    dimensions: dict[str, int] | None,
    visual: dict[str, Any],
    ml: dict[str, Any],
) -> dict[str, Any]:
    evidence: list[str] = []
    warnings = [
        "该结果是证据链分析，不是司法级结论；元数据可以被删除、伪造或在转发时丢失。"
    ]
    ai_hits = collect_ai_tool_hits(metadata, external)
    c2pa_manifest = has_external_c2pa_manifest(external)
    embedded_c2pa = metadata.get("c2pa_hints", [])
    camera_origin = has_camera_origin(metadata, external)
    heuristic_score, heuristic_evidence = weak_image_heuristics(
        image_type, dimensions, metadata, visual
    )

    if c2pa_manifest:
        evidence.append("c2patool 检测到 C2PA/Content Credentials manifest")
    elif embedded_c2pa:
        evidence.extend(embedded_c2pa)
    else:
        evidence.append("未检测到可验证的 C2PA 来源凭证")

    if not external.get("c2patool", {}).get("available"):
        evidence.append("本机未安装 c2patool；无法验证 GPT/DALL-E 等图片可能携带的 C2PA 签名")
    if not external.get("exiftool", {}).get("available"):
        evidence.append("本机未安装 exiftool；只能读取一部分基础 EXIF/XMP")

    if ai_hits:
        evidence.append("元数据或来源凭证中出现 AI 工具痕迹：" + ", ".join(sorted(set(ai_hits))))
    if camera_origin:
        evidence.append("检测到相机/镜头/拍摄时间等来源字段，倾向真实拍摄来源，但这些字段可被编辑")
    if ml.get("status") == "available" and ml.get("task") == "binary":
        ai_probability = float(ml.get("ai_probability", 0.0))
        evidence.append(f"训练模型输出 AI 分数：{ai_probability:.3f}")
    elif ml.get("status") == "available" and ml.get("task") == "source":
        evidence.append(f"训练模型判断来源更接近：{ml.get('label')}")
    elif ml.get("status") in {"not_configured", "unavailable"}:
        evidence.append(str(ml.get("message", "训练模型未启用")))
    evidence.extend(heuristic_evidence)

    if c2pa_manifest and ai_hits:
        return {
            "label": "confirmed_ai_generated",
            "title": "基本可确认 AI 生成或 AI 编辑",
            "confidence": 0.98,
            "evidence": evidence,
            "warnings": warnings,
        }
    if ai_hits:
        return {
            "label": "likely_ai_generated",
            "title": "很可能包含 AI 生成痕迹",
            "confidence": 0.82,
            "evidence": evidence,
            "warnings": warnings,
        }
    if ml.get("status") == "available" and ml.get("task") == "binary":
        ai_probability = float(ml.get("ai_probability", 0.0))
        if ai_probability >= 0.85:
            return {
                "label": "likely_ai_generated",
                "title": "模型强烈倾向 AI 生成",
                "confidence": round(min(0.92, ai_probability), 2),
                "evidence": evidence,
                "warnings": warnings
                + [
                    "模型只在 Defactify 数据集分布上训练；豆包、Flux、微信压缩、局部重绘等场景仍可能错判。"
                ],
            }
        if ai_probability >= 0.65:
            return {
                "label": "possibly_ai_generated",
                "title": "模型倾向 AI，但证据不足以确认",
                "confidence": round(min(0.76, ai_probability), 2),
                "evidence": evidence,
                "warnings": warnings
                + ["模型分数只是概率证据；建议结合 C2PA、EXIF 和来源链一起判断。"],
            }
        if ai_probability <= 0.15 and camera_origin:
            return {
                "label": "likely_camera_originated",
                "title": "模型和元数据都更倾向真实拍摄",
                "confidence": round(min(0.86, 1 - ai_probability), 2),
                "evidence": evidence,
                "warnings": warnings,
            }
    if embedded_c2pa:
        return {
            "label": "possibly_ai_edited",
            "title": "发现来源凭证结构，但需外部工具验证",
            "confidence": 0.62,
            "evidence": evidence,
            "warnings": warnings + ["建议安装 c2patool 后重新检测，以验证签名和 manifest 内容。"],
        }
    if camera_origin and heuristic_score < 0.15:
        return {
            "label": "likely_camera_originated",
            "title": "更倾向相机/手机拍摄来源",
            "confidence": 0.66,
            "evidence": evidence,
            "warnings": warnings,
        }
    if not camera_origin and heuristic_score >= 0.32:
        return {
            "label": "possibly_ai_generated",
            "title": "可能是 AI 生成或平台重导出的图片",
            "confidence": round(min(0.68, 0.54 + heuristic_score * 0.22), 2),
            "evidence": evidence,
            "warnings": warnings
            + [
                "这是弱启发式判断，不是确认结论；微信转发、截图、压缩和修图都可能产生类似信号。"
            ],
        }
    if heuristic_score >= 0.18:
        return {
            "label": "inconclusive",
            "title": "无法确认，但存在弱可疑信号",
            "confidence": round(0.45 + min(heuristic_score, 0.35), 2),
            "evidence": evidence,
            "warnings": warnings
            + ["当前没有训练好的分类模型；弱信号不能作为 AI 生成结论。"],
        }
    return {
        "label": "inconclusive",
        "title": "证据不足，无法判断",
        "confidence": 0.5,
        "evidence": evidence,
        "warnings": warnings
        + ["未接入训练好的图像分类器时，不能仅凭缺失元数据判断是否为 AI 图片。"],
    }


def summarize_metadata(
    metadata: dict[str, Any], external: dict[str, Any], visual: dict[str, Any]
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "embedded_exif": metadata.get("exif", {}),
        "embedded_text": metadata.get("text", {}),
        "xmp_blocks": metadata.get("xmp", []),
        "container": metadata.get("container", {}),
        "c2pa_hints": metadata.get("c2pa_hints", []),
        "visual_forensics": visual,
        "tools": {
            "c2patool": {
                "available": external.get("c2patool", {}).get("available", False),
                "exit_code": external.get("c2patool", {}).get("exit_code"),
                "parsed": external.get("c2patool", {}).get("json"),
                "stderr": external.get("c2patool", {}).get("stderr", ""),
            },
            "exiftool": {
                "available": external.get("exiftool", {}).get("available", False),
                "exit_code": external.get("exiftool", {}).get("exit_code"),
                "parsed": external.get("exiftool", {}).get("json"),
                "stderr": external.get("exiftool", {}).get("stderr", ""),
            },
        },
    }
    return summary


def analyze_image(data: bytes, filename: str) -> dict[str, Any]:
    started = now_ms()
    image_type = sniff_image_type(data)
    if image_type == "unknown":
        raise ValueError("暂不支持该文件类型；请上传 JPEG、PNG、GIF 或 WebP 图片。")
    dimensions = parse_dimensions(data, image_type)
    metadata = parse_embedded_metadata(data, image_type)
    visual = visual_forensics(data)
    ml = ml_inference(data)
    with tempfile.NamedTemporaryFile(
        suffix=f".{image_type}", prefix="ai-image-judge-", delete=False
    ) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        external = run_external_checks(tmp_path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    verdict = choose_verdict(metadata, external, image_type, dimensions, visual, ml)
    return {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "sha256": sha256_hex(data),
        "bytes": len(data),
        "image_type": image_type,
        "dimensions": dimensions,
        "verdict": verdict,
        "metadata": summarize_metadata(metadata, external, visual),
        "model": ml,
        "elapsed_ms": now_ms() - started,
    }


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes] | None:
    match = re.search(r"boundary=(?P<boundary>[^;]+)", content_type)
    if not match:
        return None
    boundary = match.group("boundary").strip().strip('"').encode("utf-8")
    marker = b"--" + boundary
    for part in body.split(marker):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not header_blob:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        if 'name="image"' not in headers:
            continue
        filename_match = re.search(r'filename="([^"]*)"', headers)
        filename = unquote(filename_match.group(1)) if filename_match else "upload"
        if content.endswith(b"\r\n"):
            content = content[:-2]
        return filename, content
    return None


class AiImageJudgeHandler(BaseHTTPRequestHandler):
    server_version = "AiImageJudge/0.1"

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_get_or_head(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            path = "/index.html"
        if path == "/api/health":
            payload = {
                "ok": True,
                "tools": {
                    "c2patool": bool(shutil.which("c2patool")),
                    "exiftool": bool(shutil.which("exiftool")),
                },
                "model": {
                    "checkpoint": str(configured_model_path()),
                    "exists": configured_model_path().is_file(),
                },
            }
            data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if send_body:
                self.wfile.write(data)
            return
        requested = (STATIC_DIR / path.lstrip("/")).resolve()
        if not str(requested).startswith(str(STATIC_DIR.resolve())) or not requested.is_file():
            self.send_error(404)
            return
        content_type = "text/plain; charset=utf-8"
        if requested.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif requested.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif requested.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        data = requested.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def do_GET(self) -> None:
        self.serve_get_or_head(send_body=True)

    def do_HEAD(self) -> None:
        self.serve_get_or_head(send_body=False)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self.send_json(*public_error("没有收到上传内容。"))
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_json(*public_error("图片过大；当前限制为 25MB。", 413))
            return
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        parsed = parse_multipart(body, content_type)
        if not parsed:
            self.send_json(*public_error("请使用字段名 image 上传图片文件。"))
            return
        filename, data = parsed
        if not data:
            self.send_json(*public_error("上传文件为空。"))
            return
        try:
            result = analyze_image(data, filename)
        except ValueError as exc:
            self.send_json(*public_error(str(exc)))
            return
        except Exception as exc:  # Keep server alive and show a clean API error.
            self.send_json(500, {"error": f"分析失败：{exc}"})
            return
        self.send_json(200, result)

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    server = ThreadingHTTPServer((host, port), AiImageJudgeHandler)
    print(f"AI Image Judge running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
