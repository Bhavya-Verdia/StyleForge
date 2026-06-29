"""
main.py — FastAPI inference server with SSE token streaming.

Environment variables:
    USE_MOCK=true           Run without GPU (realistic fake streaming for local dev)
    BASE_MODEL_ID           Override default Mistral model
    ADAPTER_DIR             Path to QLoRA adapter weights
    HF_TOKEN                HuggingFace token for gated models
    EVAL_RESULTS_PATH       Path to eval_results.json
    LOSS_LOG_PATH           Path to training_loss.csv

Start:
    # Local Mac dev (no GPU needed):
    USE_MOCK=true uvicorn main:app --reload --port 8000

    # Production (GPU server / Colab):
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import ast
import csv
import json
import logging
import os
import random
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _CUDA_AVAILABLE = False
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model_loader import ModelLoader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
USE_MOCK          = os.getenv("USE_MOCK", "false").lower() == "true"
EVAL_RESULTS_PATH = os.getenv("EVAL_RESULTS_PATH", "../eval_results.json")
LOSS_LOG_PATH     = os.getenv("LOSS_LOG_PATH",     "../training_loss.csv")

model_loader: Optional[ModelLoader] = None

# ─────────────────────────────────────────────────────────────
# Mock responses (realistic coding outputs for local dev)
# ─────────────────────────────────────────────────────────────
MOCK_BASE = """def fibonacci(n):
    # Simple recursive implementation
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)"""

MOCK_FINETUNED = '''def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using dynamic programming.

    Uses an iterative approach for O(n) time and O(1) space complexity,
    avoiding the exponential cost of naive recursion.

    Args:
        n: The position in the Fibonacci sequence (0-indexed). Must be >= 0.

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: If n is a negative integer.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError(f"Expected a non-negative integer, got {n!r}")
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


if __name__ == "__main__":
    print([fibonacci(i) for i in range(11)])
    # → [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]'''

MOCK_RESPONSES = {"base": MOCK_BASE, "finetuned": MOCK_FINETUNED}


# ─────────────────────────────────────────────────────────────
# Lifespan: load models at startup
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_loader
    if not USE_MOCK:
        try:
            model_loader = ModelLoader()
            model_loader.load_all()
        except RuntimeError as exc:
            logger.warning("[STARTUP] %s", exc)
            logger.warning("[STARTUP] Falling back to mock mode. Set USE_MOCK=true explicitly.")
        except Exception as exc:
            logger.error("[STARTUP] Failed to load models: %s", exc, exc_info=True)
    else:
        logger.info("[STARTUP] ✓ Running in MOCK mode (no GPU required)")
    yield
    if model_loader:
        model_loader.cleanup()


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM Fine-Tuning Comparison API",
    description=(
        "Side-by-side comparison of base Mistral-7B vs QLoRA fine-tuned model on Python coding."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ─────────────────────────────────────────────────────────────
# Request schema
# ─────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:      str   = Field(..., min_length=1, max_length=4096,
                               description="The user prompt / instruction")
    max_tokens:  int   = Field(default=512, ge=64, le=2048)
    temperature: float = Field(default=0.7,  ge=0.1, le=1.5)

class ValidateRequest(BaseModel):
    code: str = Field(..., description="The python code to validate")


# ─────────────────────────────────────────────────────────────
# Streaming helpers
# ─────────────────────────────────────────────────────────────
async def _mock_stream(text: str, base_delay: float = 0.035) -> AsyncGenerator[str, None]:
    """Word-by-word mock stream that mimics realistic token cadence."""
    words = text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'token': chunk})}\n\n"
        jitter = random.uniform(0.8, 1.4)
        await asyncio.sleep(base_delay * jitter)
    yield f"data: {json.dumps({'done': True})}\n\n"


async def _real_stream(
    model, tokenizer, prompt_text: str, max_tokens: int, temperature: float
) -> AsyncGenerator[str, None]:
    """Stream tokens from MLX model."""
    from mlx_lm import stream_generate

    try:
        # stream_generate blocks, so ideally we run it in a thread if it blocks the event loop, 
        # but for simplicity we will run it directly or via to_thread.
        # mlx_lm generation is fast enough that yielding directly works for local testing.
        for response in stream_generate(
            model, 
            tokenizer, 
            prompt_text, 
            max_tokens=max_tokens,
        ):
            if response.text:
                yield f"data: {json.dumps({'token': response.text})}\n\n"
                await asyncio.sleep(0)  # yield back to event loop
    except Exception as e:
        logger.error("Generation error: %s", e, exc_info=True)

    yield f"data: {json.dumps({'done': True})}\n\n"


def _sse_response(generator: AsyncGenerator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/generate/base", summary="Stream tokens from the base (unmodified) model")
@limiter.limit("5/minute")
async def generate_base(request: Request, req: GenerateRequest):
    if USE_MOCK or model_loader is None:
        return _sse_response(_mock_stream(MOCK_RESPONSES["base"]))

    if model_loader.base_model is None:
        raise HTTPException(status_code=503, detail="Base model not loaded — check server logs.")

    # Base model expects standard Mistral Instruct chat template
    conversation = [{"role": "user", "content": req.prompt}]
    formatted_prompt = model_loader.base_tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    return _sse_response(
        _real_stream(
            model_loader.base_model, model_loader.base_tokenizer,
            formatted_prompt, req.max_tokens, req.temperature,
        )
    )


@app.post("/generate/finetuned", summary="Stream tokens from the QLoRA fine-tuned model")
@limiter.limit("5/minute")
async def generate_finetuned(request: Request, req: GenerateRequest):
    if USE_MOCK or model_loader is None:
        return _sse_response(_mock_stream(MOCK_RESPONSES["finetuned"], base_delay=0.028))

    if model_loader.finetuned_model is None:
        raise HTTPException(
            status_code=503,
            detail="Fine-tuned model not loaded — run train.py first and point ADAPTER_DIR to the output.",
        )

    # Finetuned model was trained using raw Alpaca format in data_prep.py
    formatted_prompt = f"### Instruction:\n{req.prompt}\n\n### Response:\n"

    return _sse_response(
        _real_stream(
            model_loader.finetuned_model, model_loader.finetuned_tokenizer,
            formatted_prompt, req.max_tokens, req.temperature,
        )
    )


@app.get("/adapters", summary="List available adapters in the server directory")
async def list_adapters():
    """Scan the server directory for folders containing MLX or PEFT adapter files."""
    adapters = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check for MLX or PEFT safetensors
            if os.path.exists(os.path.join(item_path, "adapters.safetensors")) or \
               os.path.exists(os.path.join(item_path, "adapter_model.safetensors")):
                adapters.append(item)
    return {"adapters": adapters, "active": model_loader is not None and model_loader.finetuned_model is not None}

class LoadAdapterRequest(BaseModel):
    adapter_name: str

@app.post("/adapter/load", summary="Dynamically load a specific adapter")
async def load_adapter(req: LoadAdapterRequest):
    if USE_MOCK or model_loader is None:
        return {"status": "ok", "mock_mode": True}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(base_dir, req.adapter_name)
    adapter_path = os.path.realpath(adapter_path)
    if not adapter_path.startswith(os.path.realpath(base_dir)):
        raise HTTPException(status_code=400, detail="Invalid adapter path.")

    if not os.path.exists(adapter_path):
        raise HTTPException(status_code=404, detail="Adapter directory not found.")
    
    try:
        model_loader.load_finetuned_model(adapter_path=adapter_path)
        return {"status": "ok", "message": f"Successfully loaded adapter {req.adapter_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate", summary="Check if code is syntactically valid Python")
@limiter.limit("20/minute")
async def validate_code(request: Request, req: ValidateRequest):
    code = req.code.strip()
    if not code:
        return {"valid": False, "error": "Empty code provided"}
    
    # Smarter heuristic to extract python code blocks
    import re
    match = re.search(r"```(?:python|py)?\n(.*?)\n```", code, re.DOTALL)
    if match:
        code = match.group(1).strip()
    elif "```" in code:
        lines = code.split("```")
        if len(lines) >= 3:
            code = lines[1].strip()
            if code.startswith("python\n") or code.startswith("py\n"):
                code = code.split("\n", 1)[-1].strip()

    try:
        ast.parse(code)
        return {"valid": True}
    except SyntaxError as e:
        return {
            "valid": False, 
            "error": str(e.msg),
            "errorLine": e.lineno
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/metrics", summary="Return ROUGE / perplexity evaluation results")
async def get_metrics():
    for path in (EVAL_RESULTS_PATH, "./eval_results.json", "../eval_results.json"):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

    # Plausible sample data so the frontend renders without running eval
    return {
        "results": [
            {
                "model": "base",
                "rouge1": 0.312, "rouge2": 0.187, "rougeL": 0.298,
                "perplexity": 18.4, "avg_response_length": 64.2, "num_samples": 50,
            },
            {
                "model": "finetuned",
                "rouge1": 0.489, "rouge2": 0.321, "rougeL": 0.463,
                "perplexity": 9.7,  "avg_response_length": 112.8, "num_samples": 50,
            },
        ],
        "eval_samples": 50,
        "note": "Sample data — run eval.py to generate real metrics",
    }


@app.get("/training-loss", summary="Return training loss curve as JSON array")
async def get_training_loss():
    for path in (LOSS_LOG_PATH, "./training_loss.csv", "../training_loss.csv"):
        if os.path.exists(path):
            rows = []
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        rows.append({
                            "step":          int(row["step"]),
                            "loss":          float(row["loss"])          if row.get("loss")          else None,
                            "learning_rate": float(row["learning_rate"]) if row.get("learning_rate") else None,
                            "epoch":         float(row["epoch"])         if row.get("epoch")         else None,
                        })
                    except (ValueError, KeyError):
                        continue
            return {"data": rows}

    # Generate a realistic simulated loss curve
    rng = random.Random(42)
    data, loss = [], 2.85
    for step in range(10, 301, 10):
        loss = max(0.28, loss - rng.uniform(0.04, 0.11) + rng.uniform(-0.015, 0.015))
        data.append({
            "step":  step,
            "loss":  round(loss, 4),
            "epoch": round(step / 100, 2),
        })
    return {"data": data, "note": "Simulated data — run train.py to generate real loss curve"}


@app.get("/health", summary="Server health check")
async def health():
    return {
        "status":        "ok",
        "mock_mode":     USE_MOCK or model_loader is None,
        "cuda_available": _CUDA_AVAILABLE,
        "models_loaded": (
            model_loader is not None and model_loader.base_model is not None
        ),
    }

# ─────────────────────────────────────────────────────────────
# Static Frontend Serving (For Single-Container Production)
# ─────────────────────────────────────────────────────────────
frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")

if os.path.isdir(frontend_dist):
    logger.info(f"Serving frontend static files from: {frontend_dist}")
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend(full_path: str):
        # Serve index.html for all unknown routes to support React Router
        path_to_file = os.path.join(frontend_dist, full_path)
        if full_path != "" and os.path.isfile(path_to_file):
            return FileResponse(path_to_file)
        return FileResponse(os.path.join(frontend_dist, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
