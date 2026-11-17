"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from pydantic_settings import BaseSettings
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials 
import jwt 
import httpx 

app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# TODO: реализуйте остальные ручки контракта (см. API_CONTRACT.md).
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
