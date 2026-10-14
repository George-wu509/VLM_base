"""Local service: VLM_CHECKPOINT=/path/model.pt uvicorn serving.api:create_app --factory.

One worker owns one model, requests are serialized to avoid concurrent VRAM
spikes. Uploaded image bytes are used directly; no client-supplied file paths.
"""
import io
import os
import threading
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from training.checkpoint import model_from_checkpoint
from training.config import device_for
from inference.image_processor import process_image
from inference.generate import generate_text
from inference.grounding_parser import parse_grounding

def create_app(checkpoint=None, device="auto"):
    lock = threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        path = checkpoint or os.environ.get("VLM_CHECKPOINT")
        if not path:
            raise RuntimeError("Set VLM_CHECKPOINT to a trained checkpoint or export model.pt")
        app.state.model, app.state.tokenizer, _ = model_from_checkpoint(path)
        app.state.model.to(device_for(os.environ.get("VLM_DEVICE", device))).eval()
        yield

    app = FastAPI(title="simple-vlm", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "device": str(next(app.state.model.parameters()).device),
                "spatial_tokens": app.state.model.cfg.spatial_tokens}

    @app.post("/generate")
    def generate(image: UploadFile = File(...), prompt: str = Form("caption en", max_length=4096),
                 max_new_tokens: int = Form(64, ge=1, le=256), temperature: float = Form(0.0, ge=0, le=2)):
        content = image.file.read(10*1024*1024+1)
        if len(content) > 10*1024*1024:
            raise HTTPException(413, "Image upload exceeds 10 MiB")
        try:
            with Image.open(io.BytesIO(content)) as source:
                pixels = process_image(source, app.state.model.cfg.vit_img_size)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "Invalid image") from exc
        try:
            with lock:
                text = generate_text(app.state.model, app.state.tokenizer, prompt, pixels,
                                     max_new_tokens=max_new_tokens, temperature=temperature)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {"text": text, "objects": parse_grounding(text)}
    return app
