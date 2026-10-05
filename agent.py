"""Autonomous support-call processing for OmniCall AI."""

import base64
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
import sentry_sdk
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

logger = logging.getLogger("omnicall.agent")

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    send_default_pii=True,
)

AGENT_NAME = "OmniCall"
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
_GEMMA_PROMPT = (
    "A parent messaged a teacher who is in an evening class and cannot stop to read. "
    "Reply with one JSON object and no markdown. "
    "Keys: intent (billing for fees, technical_support for the class link or website, "
    "account for the parent login, cancellation for leaving the class, "
    "order_status for worksheets or materials, or general_inquiry), "
    "priority (low, medium, high, or critical), urgency (same as priority), "
    "summary (one sentence about what the parent needs), "
    "resolution_summary (two short sentences the teacher can send back to the parent, written to be spoken aloud), "
    "confidence (number from 0 to 1).\n\nParent message:\n"
)

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
mongo_client = MongoClient(os.getenv("MONGO_URI"))

# Matilda — a premade voice available on the free ElevenLabs plan.
# The older Rachel voice (21m00Tcm4TlvDq8ikWAM) is a library voice and returns 402.
DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "XrExE9yKIg1WjnnlVkGX")
TTS_MODEL_ID = "eleven_flash_v2_5"

_db = mongo_client[os.getenv("MONGO_DB_NAME", "omnicall_ai")]
support_logs = _db["support_logs"]

_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "billing": ("bill", "charge", "fee", "fees", "payment", "refund", "paid"),
    "technical_support": ("link", "not working", "cannot join", "can't join", "website", "down", "crash"),
    "account": ("password", "portal", "login", "locked out", "email"),
    "cancellation": ("cancel", "leave the class", "enrollment", "withdraw"),
    "order_status": ("worksheet", "material", "tracking", "arrive", "delivery"),
}

_URGENCY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "critical": ("urgent", "asap", "immediately", "emergency", "outage", "can't access", "cannot access"),
    "high": ("refund", "cancel", "broken", "frustrated", "angry", "charged twice", "not working"),
    "medium": ("help", "issue", "problem", "question", "how do i"),
}

_RESOLUTIONS: dict[str, str] = {
    "billing": (
        "I see the class fee was charged twice. I will check the payment and start the refund."
    ),
    "technical_support": (
        "The class link is not working, so this is {priority} priority. "
        "I will send a fresh link as soon as class allows."
    ),
    "account": (
        "You are locked out of the parent portal. I will reset the login after this class."
    ),
    "cancellation": (
        "I have your request to leave the class. I will confirm it before the next fee is charged."
    ),
    "order_status": (
        "The class materials have not arrived. I will check the tracking and update you."
    ),
    "general_inquiry": (
        "I have your question. I will reply properly as soon as this class ends."
    ),
}


def process_support_call(transcript_text: str) -> dict[str, Any]:
    """Analyze a support transcript, log it, and synthesize a spoken resolution.

    Intent and urgency come from a Gemma-compatible API when ``GEMMA_API_URL``
    is set. Otherwise a local stand-in performs the same extraction so the
    hackathon flow still runs without a model server.
    """
    transcript = (transcript_text or "").strip()
    if not transcript:
        raise ValueError("transcript_text must not be empty")

    sentry_sdk.get_current_scope().set_conversation_id(str(uuid.uuid4()))
    with sentry_sdk.start_span(op="gen_ai.invoke_agent", name=f"invoke_agent {AGENT_NAME}") as span:
        span.set_data("gen_ai.operation.name", "invoke_agent")
        span.set_data("gen_ai.agent.name", AGENT_NAME)
        span.set_data("gen_ai.pipeline.name", "omnicall-support")
        analysis = _extract_intent(transcript)
        database_status = _log_support_call(transcript, analysis)
        audio_bytes = _synthesize_resolution(analysis["resolution_summary"])

    return {
        "intent_analysis": analysis,
        "database_status": database_status,
        "message": "Support call processed successfully",
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }


def _extract_intent(transcript_text: str) -> dict[str, Any]:
    """Ask hosted Gemma, then a custom endpoint, then the local stand-in."""
    if os.getenv("GEMINI_API_KEY"):
        try:
            return _call_hosted_gemma(transcript_text)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            logger.exception("Hosted Gemma request failed; using local reasoning")
            sentry_sdk.capture_exception(exc)

    api_url = os.getenv("GEMMA_API_URL")
    if api_url:
        try:
            return _call_gemma_api(api_url, transcript_text)
        except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
            logger.exception("Gemma API request failed; using local reasoning")
            sentry_sdk.capture_exception(exc)

    return _simulate_gemma_reasoning(transcript_text)


def _call_hosted_gemma(transcript_text: str) -> dict[str, Any]:
    """Classify a transcript with Google-hosted Gemma. No local model download."""
    model = os.getenv("GEMMA_MODEL", GEMMA_MODEL)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = _GEMMA_PROMPT + transcript_text
    with sentry_sdk.start_span(op="gen_ai.chat", name=f"chat {model}") as span:
        span.set_data("gen_ai.operation.name", "chat")
        span.set_data("gen_ai.provider.name", "google")
        span.set_data("gen_ai.request.model", model)
        span.set_data("gen_ai.agent.name", AGENT_NAME)
        span.set_data("gen_ai.system_instructions", _GEMMA_PROMPT.strip())
        span.set_data(
            "gen_ai.input.messages",
            json.dumps([{"role": "user", "parts": [{"type": "text", "content": transcript_text}]}]),
        )
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2},
            },
            timeout=float(os.getenv("GEMMA_TIMEOUT_SECONDS", "60")),
        )
        response.raise_for_status()
        payload = response.json()
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        usage = payload.get("usageMetadata") or {}
        span.set_data("gen_ai.response.model", model)
        span.set_data(
            "gen_ai.output.messages",
            json.dumps([{"role": "assistant", "parts": [{"type": "text", "content": text}]}]),
        )
        if usage.get("promptTokenCount") is not None:
            span.set_data("gen_ai.usage.input_tokens", int(usage["promptTokenCount"]))
        if usage.get("candidatesTokenCount") is not None:
            span.set_data("gen_ai.usage.output_tokens", int(usage["candidatesTokenCount"]))

    parsed = _parse_model_json(text)
    analysis = _analysis_from_payload(parsed, transcript_text)
    analysis["source"] = model
    return analysis


def _call_gemma_api(api_url: str, transcript_text: str) -> dict[str, Any]:
    """Call an OpenAI-style endpoint that fronts Gemma, such as a remote Ollama server."""
    model = os.getenv("GEMMA_MODEL", "gemma")
    response = requests.post(
        api_url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": _GEMMA_PROMPT},
                {"role": "user", "content": transcript_text},
            ],
            "stream": False,
        },
        headers=_gemma_headers(),
        timeout=float(os.getenv("GEMMA_TIMEOUT_SECONDS", "60")),
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    parsed = _parse_model_json(content) if isinstance(content, str) else content
    analysis = _analysis_from_payload(parsed, transcript_text)
    analysis["source"] = model
    return analysis


def _parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        fence = text.rfind("```")
        if fence != -1:
            text = text[:fence]
    start = text.find("{")
    if start == -1:
        raise ValueError("Gemma did not return JSON")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(parsed, dict):
        raise ValueError("Gemma JSON was not an object")
    return parsed


def _gemma_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("GEMMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _analysis_from_payload(payload: dict[str, Any], transcript_text: str) -> dict[str, Any]:
    """Normalize either a flat JSON body or a chat-completion envelope."""
    if "choices" in payload:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            payload = {"summary": content, "resolution_summary": content}
        elif isinstance(content, dict):
            payload = content

    priority = _normalize_priority(str(payload.get("priority") or payload.get("urgency") or "medium"))
    intent = str(payload.get("intent") or "general_inquiry").strip().lower().replace(" ", "_")
    summary = str(payload.get("summary") or _clip(transcript_text))
    resolution = str(payload.get("resolution_summary") or _resolution_for(intent, priority))
    try:
        confidence = float(payload.get("confidence", 0.75))
    except (TypeError, ValueError):
        confidence = 0.75
    return {
        "intent": intent,
        "priority": priority,
        "urgency": priority,
        "summary": summary,
        "resolution_summary": resolution,
        "confidence": confidence,
    }


def _simulate_gemma_reasoning(transcript_text: str) -> dict[str, Any]:
    """Lightweight stand-in for Gemma open-weight intent and urgency extraction."""
    lowered = transcript_text.lower()
    intent, intent_score = _best_match(lowered, _INTENT_KEYWORDS, default="general_inquiry")
    priority, urgency_score = _best_match(lowered, _URGENCY_KEYWORDS, default="low")
    if priority == "low" and intent in {"cancellation", "billing"}:
        priority = "medium"

    confidence = min(0.95, 0.4 + (0.12 * intent_score) + (0.08 * urgency_score))
    return {
        "intent": intent,
        "priority": priority,
        "urgency": priority,
        "summary": _clip(transcript_text),
        "resolution_summary": _resolution_for(intent, priority),
        "confidence": round(confidence, 2),
        "source": "gemma-simulation",
    }


def _best_match(
    text: str,
    table: dict[str, tuple[str, ...]],
    default: str,
) -> tuple[str, int]:
    best_label = default
    best_score = 0
    for label, keywords in table.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_label = label
            best_score = score
    return best_label, best_score


def _normalize_priority(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "p1": "critical",
        "urgent": "critical",
        "p2": "high",
        "p3": "medium",
        "normal": "medium",
        "p4": "low",
    }
    normalized = aliases.get(lowered, lowered)
    if normalized not in {"critical", "high", "medium", "low"}:
        return "medium"
    return normalized


def _resolution_for(intent: str, priority: str) -> str:
    template = _RESOLUTIONS.get(intent, _RESOLUTIONS["general_inquiry"])
    return template.format(priority=priority)


def _clip(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _log_support_call(transcript_text: str, analysis: dict[str, Any]) -> str:
    """Insert the query and Gemma analysis into the support_logs collection."""
    document = {
        "transcript_text": transcript_text,
        "analysis": analysis,
        "created_at": datetime.now(timezone.utc),
    }
    with sentry_sdk.start_span(op="gen_ai.execute_tool", name="execute_tool save_support_log") as span:
        span.set_data("gen_ai.operation.name", "execute_tool")
        span.set_data("gen_ai.tool.name", "save_support_log")
        span.set_data("gen_ai.agent.name", AGENT_NAME)
        try:
            result = support_logs.insert_one(document)
        except PyMongoError as exc:
            span.set_status("internal_error")
            logger.exception("Failed to write support log")
            sentry_sdk.capture_exception(exc)
            return "failed"
    return f"logged:{result.inserted_id}"


def list_support_logs(limit: int = 10) -> list[dict[str, Any]]:
    """Return the newest support calls stored in Atlas."""
    try:
        documents = support_logs.find().sort("created_at", -1).limit(limit)
    except PyMongoError as exc:
        logger.exception("Failed to read support logs")
        sentry_sdk.capture_exception(exc)
        return []

    items: list[dict[str, Any]] = []
    for document in documents:
        analysis = document.get("analysis") or {}
        created_at = document.get("created_at")
        items.append(
            {
                "id": str(document.get("_id")),
                "transcript_text": document.get("transcript_text", ""),
                "intent": analysis.get("intent"),
                "priority": analysis.get("priority"),
                "resolution_summary": analysis.get("resolution_summary"),
                "source": analysis.get("source"),
                "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else None,
            }
        )
    return items


def _synthesize_resolution(resolution_summary: str) -> bytes:
    """Convert the resolution summary to speech with ElevenLabs Flash."""
    with sentry_sdk.start_span(op="gen_ai.execute_tool", name="execute_tool elevenlabs_tts") as span:
        span.set_data("gen_ai.operation.name", "execute_tool")
        span.set_data("gen_ai.tool.name", "elevenlabs_tts")
        span.set_data("gen_ai.agent.name", AGENT_NAME)
        span.set_data("gen_ai.request.model", TTS_MODEL_ID)
        audio = client.text_to_speech.convert(
            text=resolution_summary,
            voice_id=DEFAULT_VOICE_ID,
            model_id=TTS_MODEL_ID,
        )
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio)
        return b"".join(audio)
