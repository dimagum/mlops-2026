#!/usr/bin/env python3
"""Нагрузочная самопроверка стека (критерии 4, 5, 7, 9, 10 из ЗАДАНИЕ.md).

Запуск (нужны httpx и pyjwt, есть в .venv):
    .venv/bin/python scripts/load_test.py [--base-url http://localhost:8080]
        [--concurrency 20] [--duration 20] [--image sample.jpg]

Воркеры шлют микс /vision + /chat (stream). Считаются: доля успешных (>= 0.95),
отсутствие 5xx, p95 латентности /vision (< 400 мс), TTFT /chat (< 2000 мс).
До и после снимаются метрики Triton/vLLM: проверяется рост счётчиков на число
запросов и dynamic batching (delta nv_inference_count / nv_inference_exec_count >= 1.5).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
import sys
import time

import httpx
import jwt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_env(key: str) -> str:
    if os.environ.get(key):
        return os.environ[key]
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip()
    return ""


def metric_sum(text: str, name: str, label_filter: str = "") -> float:
    total = 0.0
    for m in re.finditer(rf"^{re.escape(name)}\{{([^}}]*)\}}\s+([0-9.eE+-]+)", text, re.M):
        if label_filter and label_filter not in m.group(1):
            continue
        total += float(m.group(2))
    return total


async def fetch_metrics(client: httpx.AsyncClient, base: str, cv_model: str, llm: str) -> dict:
    triton = (await client.get(f"{base}/metrics/triton")).text
    vllm = (await client.get(f"{base}/metrics/vllm")).text
    return {
        "infer_count": metric_sum(triton, "nv_inference_count", f'model="{cv_model}"'),
        "exec_count": metric_sum(triton, "nv_inference_exec_count", f'model="{cv_model}"'),
        "gpu_mem": metric_sum(triton, "nv_gpu_memory_total_bytes"),
        "vllm_success": metric_sum(vllm, "vllm:request_success_total", f'model_name="{llm}"'),
    }


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8080")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--image", default=os.path.join(ROOT, "sample.jpg"))
    args = p.parse_args()

    secret = read_env("JWT_SECRET")
    cv_model = read_env("CV_MODEL")
    llm = read_env("LLM_MODEL") or "Qwen2.5-1.5B-Instruct"
    token = jwt.encode({"sub": "load-test", "exp": int(time.time()) + 3600}, secret, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    with open(args.image, "rb") as f:
        image_bytes = f.read()

    results: list[dict] = []
    end = time.monotonic() + args.duration

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=args.concurrency + 10),
    ) as client:
        before = await fetch_metrics(client, args.base_url, cv_model, llm)

        async def do_vision() -> dict:
            t0 = time.monotonic()
            r = await client.post(
                f"{args.base_url}/vision", headers=headers,
                files={"file": ("img.jpg", image_bytes, "image/jpeg")},
            )
            dt = (time.monotonic() - t0) * 1000
            ok = r.status_code == 200 and len(r.json().get("predictions", [])) == 5
            return {"kind": "vision", "status": r.status_code, "ms": dt, "ok": ok}

        async def do_chat() -> dict:
            t0 = time.monotonic()
            ttft = None
            chunks = 0
            done = False
            status = 0
            try:
                async with client.stream(
                    "POST", f"{args.base_url}/chat", headers=headers,
                    json={"messages": [{"role": "user", "content": "What is 37 + 48? Answer with the number only."}],
                          "max_tokens": 32, "temperature": 0.0, "stream": True},
                ) as r:
                    status = r.status_code
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        if ttft is None:
                            ttft = (time.monotonic() - t0) * 1000
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                        else:
                            chunks += 1
            except httpx.HTTPError:
                pass
            ok = status == 200 and chunks >= 2 and done
            return {"kind": "chat", "status": status, "ms": ttft or -1, "ok": ok}

        async def worker(i: int) -> None:
            n = 0
            while time.monotonic() < end:
                res = await (do_vision() if (i + n) % 2 == 0 else do_chat())
                results.append(res)
                n += 1

        await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
        await asyncio.sleep(1.0)  # метрики отстают на scrape-интервал
        after = await fetch_metrics(client, args.base_url, cv_model, llm)

    vision = [r for r in results if r["kind"] == "vision"]
    chat = [r for r in results if r["kind"] == "chat"]
    ok_count = sum(r["ok"] for r in results)
    s5xx = sum(r["status"] >= 500 for r in results)
    v_ms = sorted(r["ms"] for r in vision if r["ok"])
    c_ttft = sorted(r["ms"] for r in chat if r["ok"] and r["ms"] >= 0)

    def p95(xs: list[float]) -> float:
        return xs[min(len(xs) - 1, int(len(xs) * 0.95))] if xs else -1

    d_count = after["infer_count"] - before["infer_count"]
    d_exec = after["exec_count"] - before["exec_count"]
    batch_ratio = d_count / d_exec if d_exec > 0 else 0.0
    d_vllm = after["vllm_success"] - before["vllm_success"]

    print(f"запросов: {len(results)} (vision {len(vision)}, chat {len(chat)})")
    print(f"успешных: {ok_count}/{len(results)} = {ok_count / max(1, len(results)):.3f}  (нужно >= 0.95)")
    print(f"5xx: {s5xx}  (нужно 0)")
    v_p50 = statistics.median(v_ms) if v_ms else -1
    c_p50 = statistics.median(c_ttft) if c_ttft else -1
    print(f"vision p50/p95/max мс: {v_p50:.1f} / {p95(v_ms):.1f} / {max(v_ms) if v_ms else -1:.1f}  (p95 < 400)")
    print(f"chat TTFT p50/p95/max мс: {c_p50:.1f} / {p95(c_ttft):.1f} / {max(c_ttft) if c_ttft else -1:.1f}  (< 2000)")
    print(f"triton: delta inference_count={d_count:.0f} (vision-запросов {len(vision)}), delta exec_count={d_exec:.0f}")
    print(f"dynamic batching: {batch_ratio:.2f} картинок/запуск  (нужно >= 1.5)")
    print(f"vllm: delta request_success_total={d_vllm:.0f} (chat-запросов успешных {sum(r['ok'] for r in chat)})")
    print(f"gpu-метрики Triton: nv_gpu_memory_total_bytes={after['gpu_mem']:.0f}")

    checks = [
        (ok_count / max(1, len(results)) >= 0.95, "доля успешных >= 0.95"),
        (s5xx == 0, "нет 5xx"),
        (0 <= p95(v_ms) < 400, "vision p95 < 400 мс"),
        (0 <= p95(c_ttft) < 2000 and (max(c_ttft) if c_ttft else 1e9) < 2000, "chat TTFT < 2000 мс"),
        (d_count >= len(vision) * 0.95, "счётчик Triton вырос на число /vision"),
        (batch_ratio >= 1.5, "dynamic batching >= 1.5"),
        (d_vllm >= sum(r["ok"] for r in chat) * 0.95, "счётчик vLLM вырос на число /chat"),
        (after["gpu_mem"] > 0, "есть GPU-метрики Triton"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    for ok, msg in checks:
        print(f"  {'ok' if ok else 'FAIL'}: {msg}")
    if failed:
        print(f"\nПРОВАЛЕНО: {len(failed)} проверок")
        return 1
    print("\nНАГРУЗОЧНАЯ САМОПРОВЕРКА ПРОЙДЕНА")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
