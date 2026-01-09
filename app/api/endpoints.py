import asyncio
from fastapi import APIRouter, Request, Form, BackgroundTasks, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

from app.core.dependencies import get_engine

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")
logger = structlog.get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@router.post("/v1/train-knowledge")
@limiter.limit("5/minute")
async def train(
    request: Request,
    background_tasks: BackgroundTasks,
    text_data: str = Form(...),
):
    """
    Industry Grade: Training happens in background.
    Server does not freeze.
    """
    engine = get_engine()
    background_tasks.add_task(engine.learn_new_data, text_data)

    logger.info("training_started", text_length=len(text_data))

    return {
        "status": "Training started",
        "message": "You can continue chatting while I learn.",
    }


# Legacy endpoint (deprecated)
@router.post("/train-knowledge")
async def train_legacy(
    request: Request,
    background_tasks: BackgroundTasks,
    text_data: str = Form(...),
):
    """Legacy endpoint - redirects to v1"""
    engine = get_engine()
    background_tasks.add_task(engine.learn_new_data, text_data)
    return {
        "status": "Training started",
        "message": "You can continue chatting while I learn.",
    }


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the Investor Dashboard"""
    engine = get_engine()
    return templates.TemplateResponse(
        "index.html", {"request": request, "history": engine.memory.history}
    )


async def async_query_generator(engine, query: str):
    """Async wrapper for the synchronous query generator with true streaming."""
    import queue
    import threading

    chunk_queue = queue.Queue()
    done_sentinel = object()

    def producer():
        """Run the sync generator in a thread and put chunks in queue."""
        try:
            for chunk in engine.run_query_generator(query):
                chunk_queue.put(chunk)
        finally:
            chunk_queue.put(done_sentinel)

    # Start producer thread
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # Yield chunks as they arrive (preserving the slow typing effect)
    while True:
        try:
            # Non-blocking check with small timeout to stay async-friendly
            chunk = await asyncio.get_event_loop().run_in_executor(
                None, lambda: chunk_queue.get(timeout=0.1)
            )
            if chunk is done_sentinel:
                break
            yield chunk
        except queue.Empty:
            # Queue empty, wait a bit and check again
            await asyncio.sleep(0.01)


@router.post("/v1/chat")
@limiter.limit("30/minute")
async def chat_v1(
    request: Request, query: str = Form(...), mode: str = Form("assistant")
):
    """API Endpoint for interaction (v1 with rate limiting)"""
    engine = get_engine()
    engine.mode = mode

    logger.info("chat_request", query_length=len(query), mode=mode)

    return StreamingResponse(
        async_query_generator(engine, query), media_type="application/x-ndjson"
    )


# Legacy endpoint
@router.post("/chat")
async def chat(request: Request, query: str = Form(...), mode: str = Form("assistant")):
    """Legacy chat endpoint"""
    engine = get_engine()
    engine.mode = mode
    return StreamingResponse(
        async_query_generator(engine, query), media_type="application/x-ndjson"
    )


@router.get("/v1/health")
async def health_v1():
    """Health check with detailed stats"""
    engine = get_engine()
    memory_stats = engine.memory.get_stats()

    return {
        "status": "active",
        "engine": "PIL-VAE Hybrid",
        "version": "1.0.0",
        "memory": memory_stats,
    }


@router.get("/health")
async def health():
    return {"status": "active", "engine": "PIL-VAE Hybrid"}


@router.get("/v1/stats")
async def stats():
    """Get detailed engine statistics"""
    engine = get_engine()
    return {
        "memory": engine.memory.get_stats(),
        "mode": engine.mode,
        "vae_trained": engine.memory_embeddings is not None,
    }


@router.get("/v1/training-status")
async def training_status():
    """Get detailed training status with source breakdown"""
    engine = get_engine()

    # Get source breakdown from database
    conn = engine.memory._get_connection()
    cursor = conn.execute(
        "SELECT source, COUNT(*) as count FROM knowledge GROUP BY source"
    )
    source_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

    return {
        "status": "ready",
        "total_knowledge": engine.memory.get_stats()["knowledge_entries"],
        "vae_trained": engine.memory_embeddings is not None,
        "source_breakdown": source_breakdown,
        "user_trained_count": source_breakdown.get("user_training", 0),
    }
