#!/usr/bin/env python3
"""Сверка /vision с эталоном torchvision (критерий 3 из ЗАДАНИЕ.md).

Запуск (нужны torch/torchvision из .venv экспорта + httpx/pyjwt):
    .venv/bin/python scripts/check_scores.py [image.jpg ...]

Для каждой картинки: эталон = веса варианта + weights.transforms() (CPU torch),
сравниваются top-1 class_id и score (допуск грейдера ~0.02), печатаются оба top-5.
"""
from __future__ import annotations

import os
import sys
import time

import httpx
import jwt
import torch
import torchvision.models as models
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gateway"))

VARIANTS = {
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def read_env(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    cv_model = read_env("CV_MODEL")
    base = os.environ.get("BASE_URL", "http://localhost:8080")
    secret = read_env("JWT_SECRET")
    token = jwt.encode({"sub": "check", "exp": int(time.time()) + 3600}, secret, algorithm="HS256")
    images = sys.argv[1:] or [os.path.join(ROOT, "sample.jpg")]

    ctor, weights = VARIANTS[cv_model]
    model = ctor(weights=weights).eval()
    transforms = weights.transforms()
    labels = [l.strip() for l in open(os.path.join(ROOT, "gateway", "imagenet_classes.txt"), encoding="utf-8")]

    failed = 0
    with httpx.Client(timeout=60.0) as client:
        for path in images:
            img = Image.open(path)
            with torch.no_grad():
                logits = model(transforms(img.convert("RGB")).unsqueeze(0))[0]
            ref_probs = torch.softmax(logits, dim=0)
            ref_top = torch.argsort(ref_probs, descending=True)[:5]

            r = client.post(
                f"{base}/vision",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (os.path.basename(path), open(path, "rb").read(), "image/jpeg")},
            )
            if r.status_code != 200:
                print(f"FAIL {path}: gateway вернул {r.status_code}: {r.text[:200]}")
                failed += 1
                continue
            preds = r.json()["predictions"]
            got_top1, ref_top1 = preds[0], int(ref_top[0])
            diff = abs(got_top1["score"] - float(ref_probs[ref_top1]))
            ok = got_top1["class_id"] == ref_top1 and diff <= 0.02
            print(f"{'ok' if ok else 'FAIL'} {path}: top1 class={got_top1['class_id']} "
                  f"score={got_top1['score']:.4f} | эталон class={ref_top1} "
                  f"score={float(ref_probs[ref_top1]):.4f} | diff={diff:.4f}")
            print("  gateway top-5:", [(p["class_id"], round(p["score"], 4)) for p in preds])
            print("  эталон  top-5:", [(int(i), round(float(ref_probs[i]), 4)) for i in ref_top])
            if not ok:
                failed += 1
    if failed:
        print(f"\nПРОВАЛЕНО: {failed}")
        return 1
    print("\nСКОРЫ СОВПАДАЮТ С ЭТАЛОНОМ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
