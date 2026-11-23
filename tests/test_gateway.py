"""Локальные тесты контракта gateway без Docker.

Mock-бэкенды (Triton, vLLM, метрики) поднимаются в процессе на случайном порту,
gateway тестируется через fastapi.TestClient. Препроцессинг сверяется с официальным
weights.transforms() torchvision (нужен torch/torchvision в окружении).

Запуск:  .venv/bin/python tests/test_gateway.py
Зависимости: fastapi, uvicorn, httpx, pyjwt, pydantic-settings,
prometheus-fastapi-instrumentator, python-multipart, tritonclient[http], numpy, pillow.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "gateway"))

import jwt as pyjwt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from PIL import Image

SECRET = "test-secret"
LLM = "Qwen2.5-1.5B-Instruct"

mock = FastAPI()


@mock.get("/v2/health/ready")
def triton_ready():
    return JSONResponse({}, status_code=200)


@mock.post("/v2/models/{model}/infer")
async def triton_infer(model: str, request: Request):
    # tritonclient может слать JSON + бинарный тензорный хвост — читаем только JSON-часть
    raw = (await request.body()).decode("latin-1")
    json.JSONDecoder().raw_decode(raw)
    logits = [0.0] * 1000
    for idx, val in [(207, 10.0), (208, 5.0), (220, 3.0), (222, 2.0), (852, 1.0)]:
        logits[idx] = val
    return {
        "model_name": model,
        "model_version": "1",
        "outputs": [
            {"name": "OUTPUT__0", "datatype": "FP32", "shape": [1, 1000], "data": logits}
        ],
    }


@mock.get("/v1/models")
def vllm_models():
    return {"data": [{"id": LLM}]}


@mock.post("/v1/chat/completions")
async def vllm_chat(request: Request):
    body = await request.json()
    msgs = body.get("messages")
    if not msgs:
        return JSONResponse({"error": "messages required"}, status_code=400)
    if any(len(str(m.get("content", ""))) > 10000 for m in msgs):
        return JSONResponse({"error": "context length exceeded"}, status_code=400)
    if body.get("stream"):
        async def gen():
            for piece in ["PO", "NG"]:
                yield "data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "PONG"}, "finish_reason": "stop"}
        ],
    }


@mock.get("/metrics")
def metrics():
    return PlainTextResponse("fake_metric_total 1\n")


def start_mock() -> str:
    config = uvicorn.Config(mock, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(0.02)
    else:
        raise RuntimeError("mock server did not start")
    port = server.servers[0].sockets[0].getsockname()[1]
    return f"http://127.0.0.1:{port}"


BASE = start_mock()
os.environ.update(
    {
        "JWT_SECRET": SECRET,
        "CV_MODEL": "convnext_tiny",
        "LLM_MODEL": LLM,
        "STUDENT_NAME": "Test Student",
        "STUDENT_EMAIL": "test@edu.hse.ru",
        "GPU_NAME": "Test GPU",
        "TRITON_URL": BASE,
        "VLLM_URL": BASE + "/v1",
        "TRITON_METRICS_URL": BASE + "/metrics",
        "VLLM_METRICS_URL": BASE + "/metrics",
        "NODE_METRICS_URL": BASE + "/metrics",
    }
)

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

DEAD = "http://127.0.0.1:1"


def token(secret: str = SECRET, exp_delta: int = 3600, alg: str = "HS256") -> str:
    payload = {"sub": "test", "exp": int(time.time()) + exp_delta}
    return pyjwt.encode(payload, secret if alg != "none" else "", algorithm=alg)


def auth(t: str) -> dict:
    return {"Authorization": f"Bearer {t}"}


def png_bytes(size=(300, 200), mode="RGB") -> bytes:
    rng = np.random.default_rng(42)
    channels = len(mode)
    shape = (size[1], size[0]) if channels == 1 else (size[1], size[0], channels)
    arr = rng.integers(0, 255, shape, dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG" if mode == "CMYK" else "PNG")
    return buf.getvalue()


CHECKS = 0


def check(cond: bool, msg: str) -> None:
    global CHECKS
    if not cond:
        raise AssertionError(msg)
    CHECKS += 1
    print(f"  ok: {msg}")


def main_tests() -> None:
    with TestClient(main.app) as c:
        t = token()

        print("== health / info / auth ==")
        r = c.get("/health")
        check(r.status_code == 200 and r.json() == {"status": "ok"}, "/health")

        r = c.get("/info", headers=auth(t))
        check(r.status_code == 200, "/info с валидным токеном → 200")
        body = r.json()
        check(
            body["student"]["email"] == "test@edu.hse.ru"
            and body["models"]["vision"] == "convnext_tiny"
            and body["models"]["llm"] == LLM
            and body["gpu"] == "Test GPU",
            "/info схема",
        )

        for name, headers in [
            ("без токена", {}),
            ("мусор", auth("not-a-jwt")),
            ("чужой секрет", auth(token(secret="another-secret"))),
            ("alg=none", auth(token(alg="none"))),
            ("просрочен", auth(token(exp_delta=-3600))),
        ]:
            r = c.get("/info", headers=headers)
            check(r.status_code == 401, f"/info {name} → 401")
            r = c.post("/vision", headers=headers, files={"file": ("a.png", png_bytes(), "image/png")})
            check(r.status_code == 401, f"/vision {name} → 401")
            r = c.post("/chat", headers=headers, json={"messages": [{"role": "user", "content": "hi"}]})
            check(r.status_code == 401, f"/chat {name} → 401")

        print("== backend health ==")
        r = c.get("/triton/health")
        check(
            r.status_code == 200 and r.json() == {"backend": "triton", "ready": True, "model": "convnext_tiny"},
            "/triton/health → 200",
        )
        r = c.get("/vllm/health")
        check(
            r.status_code == 200 and r.json() == {"backend": "vllm", "ready": True, "models": [LLM]},
            "/vllm/health → 200",
        )

        print("== /vision ==")
        r = c.post("/vision", headers=auth(t))
        check(r.status_code == 422, "/vision без поля file → 422")
        r = c.post("/vision", headers=auth(t), files={"file": ("a.png", b"", "image/png")})
        check(r.status_code == 400, "/vision пустой файл → 400")
        r = c.post("/vision", headers=auth(t), files={"file": ("a.txt", b"this is not an image", "text/plain")})
        check(r.status_code == 400, "/vision не картинка → 400")

        r = c.post("/vision", headers=auth(t), files={"file": ("a.png", png_bytes(), "image/png")})
        check(r.status_code == 200, "/vision валидная картинка → 200")
        body = r.json()
        preds = body["predictions"]
        check(body["model"] == "convnext_tiny", "/vision model")
        check(len(preds) == 5, "/vision ровно top-5")
        check([p["class_id"] for p in preds] == [207, 208, 220, 222, 852], "/vision порядок top-5")
        check(preds[0]["label"] == "golden retriever", "/vision class_id 207 = golden retriever")
        scores = [p["score"] for p in preds]
        check(scores == sorted(scores, reverse=True), "/vision скоры по убыванию")
        check(all(0.0 <= s <= 1.0 for s in scores) and sum(scores) <= 1.0 + 1e-6, "/vision скоры — softmax")
        check(isinstance(body["latency_ms"], float) and body["latency_ms"] >= 0, "/vision latency_ms")

        for mode in ["RGBA", "L", "CMYK"]:
            r = c.post("/vision", headers=auth(t), files={"file": ("a.png", png_bytes(mode=mode), "image/png")})
            check(r.status_code == 200, f"/vision режим {mode} → 200")

        print("== /chat ==")
        r = c.post(
            "/chat", headers={**auth(t), "Content-Type": "application/json"}, content=b"{broken json"
        )
        check(r.status_code == 400, "/chat битый JSON → 400")
        r = c.post("/chat", headers=auth(t), json={"max_tokens": 8})
        check(r.status_code == 400, "/chat без messages → 400")
        r = c.post("/chat", headers=auth(t), json={"messages": []})
        check(r.status_code == 400, "/chat пустой messages → 400")

        r = c.post(
            "/chat",
            headers=auth(t),
            json={"messages": [{"role": "user", "content": "Respond with exactly one word: PONG"}],
                  "max_tokens": 16, "temperature": 0.0, "stream": True},
        )
        check(r.status_code == 200, "/chat stream → 200")
        check(r.headers["content-type"].startswith("text/event-stream"), "/chat content-type SSE")
        text = r.text
        check(text.count("data: {") >= 2, "/chat ≥ 2 чанка")
        check(text.endswith("data: [DONE]\n\n"), "/chat финал data: [DONE]")
        content = "".join(
            json.loads(line[5:].strip())["choices"][0]["delta"]["content"]
            for line in text.splitlines()
            if line.startswith("data: {")
        )
        check(content == "PONG", "/chat контент собирается из delta")

        r = c.post(
            "/chat",
            headers=auth(t),
            json={"messages": [{"role": "user", "content": "x" * 20000}], "stream": True},
        )
        check(r.status_code == 400, "/chat промпт длиннее контекста → 400 (не 200/5xx)")

        r = c.post(
            "/chat",
            headers=auth(t),
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        check(
            r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "PONG",
            "/chat stream=false → JSON",
        )

        print("== метрики ==")
        for path in ["/metrics/triton", "/metrics/vllm", "/metrics/node"]:
            r = c.get(path)
            check(r.status_code == 200 and r.text == "fake_metric_total 1\n", f"{path} сырой проброс")
        r = c.get("/metrics")
        check(r.status_code == 200 and "http_requests_total" in r.text, "/metrics gateway")

        print("== недоступные бэкенды ==")
        saved = (main.settings.triton_url, main.settings.vllm_url,
                 main.settings.triton_metrics_url, main.settings.vllm_metrics_url,
                 main.settings.node_metrics_url)
        try:
            main.settings.triton_url = DEAD
            main._triton_client = None
            r = c.post("/vision", headers=auth(t), files={"file": ("a.png", png_bytes(), "image/png")})
            check(r.status_code == 502, "/vision Triton недоступен → 502")
            r = c.get("/triton/health")
            check(r.status_code == 503, "/triton/health недоступен → 503")

            main.settings.vllm_url = DEAD + "/v1"
            r = c.post("/chat", headers=auth(t), json={"messages": [{"role": "user", "content": "hi"}], "stream": True})
            check(r.status_code == 502, "/chat vLLM недоступен → 502")
            r = c.get("/vllm/health")
            check(r.status_code == 503, "/vllm/health недоступен → 503")

            main.settings.triton_metrics_url = DEAD + "/metrics"
            main.settings.vllm_metrics_url = DEAD + "/metrics"
            main.settings.node_metrics_url = DEAD + "/metrics"
            for path in ["/metrics/triton", "/metrics/vllm", "/metrics/node"]:
                r = c.get(path)
                check(r.status_code == 502, f"{path} недоступен → 502")
        finally:
            (main.settings.triton_url, main.settings.vllm_url,
             main.settings.triton_metrics_url, main.settings.vllm_metrics_url,
             main.settings.node_metrics_url) = saved
            main._triton_client = None


def preprocess_tests() -> None:
    print("== препроцессинг vs weights.transforms() ==")
    try:
        from torchvision.models import ConvNeXt_Tiny_Weights
    except ImportError:
        print("  SKIP: torchvision не установлен")
        return
    transforms = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    rng = np.random.default_rng(7)
    cases = [((640, 480), "RGB"), ((100, 300), "RGB"), ((224, 224), "RGB"),
             ((50, 50), "RGB"), ((1000, 700), "RGB"), ((300, 200), "RGBA"),
             ((300, 200), "L"), ((300, 200), "CMYK")]
    for size, mode in cases:
        channels = len(mode)
        shape = (size[1], size[0]) if channels == 1 else (size[1], size[0], channels)
        arr = rng.integers(0, 255, shape, dtype=np.uint8)
        img = Image.fromarray(arr, mode=mode)
        ref = transforms(img.convert("RGB")).numpy()
        mine = main.preprocess(img)[0]
        np.testing.assert_allclose(mine, ref, atol=1e-5, rtol=1e-5)
        check(True, f"препроцессинг {size} {mode} совпадает с эталоном")


if __name__ == "__main__":
    main_tests()
    preprocess_tests()
    print(f"\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ: {CHECKS}")
