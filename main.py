"""OmniCall AI HTTP API."""

import logging
import os
from pathlib import Path
from typing import Any

import sentry_sdk
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from agent import list_support_logs, process_support_call  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("omnicall.api")

app = FastAPI(
    title="OmniCall AI",
    description="Autonomous customer support agent API",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    """Customer transcript submitted by the voice or chat client."""

    text: str = Field(..., min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned


class SupportResponse(BaseModel):
    """Result of one autonomous support turn."""

    intent_analysis: dict[str, Any]
    database_status: str
    message: str
    audio_base64: str


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the console where a transcript can be heard as speech."""
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/api/support/logs")
def recent_support_calls(limit: int = 10) -> dict[str, list[dict[str, Any]]]:
    """Return recent calls stored in MongoDB Atlas."""
    safe_limit = min(max(limit, 1), 20)
    return {"calls": list_support_logs(safe_limit)}


@app.post("/api/support", response_model=SupportResponse)
def create_support_call(body: SupportRequest) -> dict:
    """Accept a transcript, run the support agent, and return JSON."""
    try:
        return process_support_call(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Support call processing failed")
        sentry_sdk.capture_exception(exc)
        raise HTTPException(status_code=500, detail="Failed to process support call") from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
    )
