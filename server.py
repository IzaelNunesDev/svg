"""
FastAPI Server Entry Point: Vector AI Studio (Anim8 Architecture)
Orchestrates modular API routers and static web studio assets.
"""

import logging
import os
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Ensure project root is in sys.path
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from api.routes import system_router, vectorize_router, export_router

app = FastAPI(
    title="Vector AI Studio - Anim8 Engine",
    description="Motor de vetorização de imagem e vídeo com Bézier cúbico e flipbook temporal.",
    version="2.0.0"
)

# CORS configuration (allow_credentials must not be combined with wildcard origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    """Disables caching for live web studio frontend assets."""
    response = await call_next(request)
    if request.url.path.endswith((".js", ".css", ".html")) or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# Register modular API routes
app.include_router(system_router)
app.include_router(vectorize_router)
app.include_router(export_router)

# Mount static web studio frontend
static_dir = os.path.join(_PROJECT_DIR, "static")
os.makedirs(static_dir, exist_ok=True)


@app.get("/")
async def serve_root():
    return FileResponse(os.path.join(static_dir, "index.html"))


app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Vector AI Studio - Anim8 Architecture Server v2.0")
    print(" Modular Engine: Hybrid Cel-Shading + Perceptual Palette")
    print(" Running on http://localhost:8000")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
