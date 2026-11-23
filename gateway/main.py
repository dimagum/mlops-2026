"""Real-time inference gateway: Triton (CV) + vLLM (chat) за JWT-авторизацией.

Контракт ручек — API_CONTRACT.md в корне этого каталога.
"""
from __future__ import annotations

import io
import json
import os
import time

import anyio.to_thread
import httpx
import jwt
import numpy as np
import tritonclient.http as tritonhttp
import tritonclient.http.aio as triton_aio
from PIL import Image
from fastapi import FastAPI, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_fastapi_instrumentator import Instrumentator

from pydantic_settings import BaseSettings
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class Settings(BaseSettings):
    jwt_secret: str
    cv_model: str
    llm_model: str = "Qwen2.5-1.5B-Instruct"
    triton_url: str = "http://triton:8000"
    vllm_url: str = "http://vllm:8000/v1"
    student_name: str
    student_email: str
    gpu_name: str

    triton_metrics_url: str
    vllm_metrics_url: str
    node_metrics_url: str


settings = Settings()

bearer = HTTPBearer(auto_error=False)

def require_auth(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> dict:
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return jwt.decode(creds.credentials, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/info")
def info(payload: dict = Depends(require_auth)) -> dict:
    return {
        "student": {"name" : settings.student_name, "email" : settings.student_email},
        "models": {"vision": settings.cv_model, "llm": settings.llm_model},
        "gpu": settings.gpu_name,
        "version": "1.0"
    }


@app.get("/triton/health")
async def triton_health_check() -> dict:

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            ready_resp = await client.get(f"{settings.triton_url}/v2/health/ready")

            if ready_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Triton server is not ready",
                )

            return {"backend": "triton", "ready": True, "model": settings.cv_model}

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to Triton server: {str(e)}",
            )

@app.get("/vllm/health")
async def vllm_health_check():

    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            ready_resp = await client.get(f"{settings.vllm_url}/models")

            if ready_resp.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="vLLM server is not ready",
                )

            models = [m["id"] for m in ready_resp.json()["data"]]

            return {"backend": "vllm", "ready": True, "models": models}

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to vLLM server: {str(e)}",
            )

Image.MAX_IMAGE_PIXELS = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PREPROCESS_PARAMS = {
    "resnet50": (256, Image.BILINEAR),
    "efficientnet_b0": (256, Image.BICUBIC),
    "mobilenet_v3_large": (232, Image.BILINEAR),
    "convnext_tiny": (236, Image.BILINEAR),
}

with open(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagenet_classes.txt"),
    encoding="utf-8",
) as f:
    IMAGENET_LABELS = [line.strip() for line in f]


def preprocess(img: Image.Image) -> np.ndarray:
    resize_size, resample = PREPROCESS_PARAMS[settings.cv_model]
    img = img.convert("RGB")

    w, h = img.size
    if w <= h:
        new_w, new_h = resize_size, int(resize_size * h / w)
    else:
        new_w, new_h = int(resize_size * w / h), resize_size
    img = img.resize((new_w, new_h), resample)
    left = int(round((new_w - 224) / 2.0))
    top = int(round((new_h - 224) / 2.0))
    img = img.crop((left, top, left + 224, top + 224))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


_triton_client: triton_aio.InferenceServerClient | None = None


def get_triton_client() -> triton_aio.InferenceServerClient:
    global _triton_client
    if _triton_client is None:
        url = settings.triton_url.split("://", 1)[-1]
        try:
            _triton_client = triton_aio.InferenceServerClient(url, concurrency=32)
        except TypeError:
            _triton_client = triton_aio.InferenceServerClient(url, conn_limit=32)
    return _triton_client


def _prepare_input(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ValueError("not an image") from e
    return preprocess(img)


@app.post("/vision")
async def vision(file: UploadFile = File(...), payload: dict = Depends(require_auth)) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        x = await anyio.to_thread.run_sync(_prepare_input, data)
    except ValueError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    try:
        inp = tritonhttp.InferInput("INPUT__0", list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        out = tritonhttp.InferRequestedOutput("OUTPUT__0")
        t0 = time.perf_counter()
        result = await get_triton_client().infer(settings.cv_model, inputs=[inp], outputs=[out])
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logits = result.as_numpy("OUTPUT__0")[0]
    except Exception:
        raise HTTPException(status_code=502, detail="Triton inference failed")

    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    top5 = np.argsort(-probs)[:5]
    return {
        "model": settings.cv_model,
        "predictions": [
            {"class_id": int(i), "label": IMAGENET_LABELS[int(i)], "score": float(probs[i])}
            for i in top5
        ],
        "latency_ms": float(latency_ms),
    }


@app.post("/chat")
async def chat(request: Request, payload: dict = Depends(require_auth)):
    try:
        body = await request.json()

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(body, dict) or not body.get("messages"):
        raise HTTPException(status_code=400, detail="Field 'messages' is required")

    body["model"] = settings.llm_model
    stream = bool(body.get("stream", False))

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))
    try:
        upstream = await client.send(
            client.build_request("POST", f"{settings.vllm_url}/chat/completions", json=body),
            stream=True,
        )
    except httpx.HTTPError:
        await client.aclose()
        raise HTTPException(status_code=502, detail="vLLM is unavailable")

    if upstream.status_code >= 400:
        detail = (await upstream.aread()).decode(errors="replace")[:500]
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=400 if upstream.status_code < 500 else 502,
            detail=f"vLLM rejected the request: {detail}",
        )

    if not stream:
        try:
            content = json.loads((await upstream.aread()).decode())
        except Exception:
            raise HTTPException(status_code=502, detail="Invalid response from vLLM")
        finally:
            await upstream.aclose()
            await client.aclose()
        return JSONResponse(content=content)

    async def event_stream():
        saw_done = False
        try:
            async for line in upstream.aiter_lines():
                if not line:
                    continue
                if line.startswith("data:") and line[5:].strip() == "[DONE]":
                    saw_done = True
                yield line + "\n\n"
        except httpx.HTTPError:
            yield 'data: {"error": {"message": "vLLM stream interrupted"}}\n\n'
            return
        finally:
            await upstream.aclose()
            await client.aclose()
        if not saw_done:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _proxy_metrics(url: str) -> Response:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)

    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Metrics backend is unavailable")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Metrics backend returned {resp.status_code}")

    return Response(content=resp.content, media_type=resp.headers.get("content-type", "text/plain"))


@app.get("/metrics/triton")
async def metrics_triton() -> Response:
    return await _proxy_metrics(settings.triton_metrics_url)


@app.get("/metrics/vllm")
async def metrics_vllm() -> Response:
    return await _proxy_metrics(settings.vllm_metrics_url)


@app.get("/metrics/node")
async def metrics_node() -> Response:
    return await _proxy_metrics(settings.node_metrics_url)
