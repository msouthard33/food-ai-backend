"""LLM provider wrapper — vision + embedding + text completions.

Supports OpenAI for embeddings and vision (gpt-4o-mini).
Never logs image bytes. Never stores photos to disk.
"""

import json
import logging
import time

import openai

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Tier-label mapping (D9 confidence framework)
# ---------------------------------------------------------------------------
TIER_THRESHOLDS = [
    (0.85, "Well-established"),
    (0.55, "Some evidence"),
    (0.0, "AI estimate"),
]


def confidence_to_tier(confidence: float) -> str:
    for threshold, label in TIER_THRESHOLDS:
        if confidence >= threshold:
            return label
    return "AI estimate"


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class LLMProvider:
    """Unified LLM provider for vision, embeddings, and text."""

    EMBEDDING_MODEL = "text-embedding-3-small"
    VISION_MODEL = "gpt-4o-mini"

    def __init__(self):
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — vision and embedding calls will fail")
        self._client = openai.AsyncOpenAI(api_key=api_key) if api_key else None

    # ------------------------------------------------------------------
    # Vision: meal photo analysis
    # ------------------------------------------------------------------

    async def analyze_meal_photo(self, image_base64: str) -> list[dict]:
        """Analyze a meal photo and return structured food identification.

        Uses gpt-4o-mini vision (cheapest vision tier).
        Returns: [{food: str, portion: str, confidence: float, tier_label: str}]
        On failure: [{food: "Photo analysis unavailable", confidence: 0,
                      tier_label: "AI estimate", error: "vision_provider_unavailable"}]

        NEVER logs image bytes. NEVER stores photos to disk.
        """
        if not self._client:
            logger.error("Vision call failed: no OpenAI client (missing API key)")
            return [self._vision_fallback("vision_provider_unavailable")]

        # Validate base64 is not empty / obviously broken
        if not image_base64 or len(image_base64) < 100:
            return [self._vision_fallback("invalid_image_data")]

        prompt = (
            "You are a food identification assistant for a dietary tracking app. "
            "Analyze this meal photo and identify every distinct food item visible.\n\n"
            "Return ONLY valid JSON — no markdown fences, no explanation.\n"
            'Format: [{"food": "<food name>", "portion": "<estimated portion>", '
            '"confidence": <0.0-1.0>}]\n\n'
            "Rules:\n"
            "- Be specific: 'grilled chicken breast' not just 'chicken'\n"
            "- Estimate portions in common units (cup, oz, piece, slice, tablespoon)\n"
            "- Confidence: 0.9+ for clearly visible items, 0.5-0.8 for partially visible, "
            "<0.5 for guesses\n"
            "- Include sauces, dressings, and condiments as separate items\n"
            '- If the image is not food, return [{"food": "Not a food image", '
            '"portion": "N/A", "confidence": 0.0}]'
        )

        t0 = time.time()
        try:
            response = await self._client.chat.completions.create(
                model=self.VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "low",  # cheapest tier
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            raw_text = response.choices[0].message.content or ""
            # Log model info (never log image bytes)
            logger.info(
                "Vision call: model=%s, tokens_prompt=%s, tokens_completion=%s, elapsed_ms=%d",
                self.VISION_MODEL,
                response.usage.prompt_tokens if response.usage else "?",
                response.usage.completion_tokens if response.usage else "?",
                elapsed_ms,
            )

            # Cost estimation (gpt-4o-mini vision: ~$0.00015/1K input tokens for low detail)
            if response.usage:
                input_cost = (response.usage.prompt_tokens / 1000) * 0.00015
                output_cost = (response.usage.completion_tokens / 1000) * 0.0006
                total_cost = input_cost + output_cost
                logger.info(
                    "Vision cost estimate: $%.6f (input=$%.6f, output=$%.6f)",
                    total_cost,
                    input_cost,
                    output_cost,
                )
                if total_cost > 0.01:
                    logger.warning(
                        "COST GATE: photo analysis cost $%.4f exceeds $0.01 threshold", total_cost
                    )

            # Parse JSON from response
            foods = self._parse_vision_response(raw_text)

            # Add tier labels
            for item in foods:
                item["tier_label"] = confidence_to_tier(item.get("confidence", 0.0))

            return foods

        except openai.APIError as e:
            logger.error("Vision API error: %s", str(e)[:200])
            return [self._vision_fallback("vision_api_error")]
        except Exception as e:
            logger.error("Vision unexpected error: %s", str(e)[:200])
            return [self._vision_fallback("vision_provider_unavailable")]

    def _parse_vision_response(self, raw_text: str) -> list[dict]:
        """Parse the vision model's JSON response."""
        # Strip markdown fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # Validate structure
                result = []
                for item in parsed:
                    result.append(
                        {
                            "food": str(item.get("food", "Unknown")),
                            "portion": str(item.get("portion", "Unknown")),
                            "confidence": float(item.get("confidence", 0.5)),
                        }
                    )
                return result
            elif isinstance(parsed, dict) and "foods" in parsed:
                return self._parse_vision_response(json.dumps(parsed["foods"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse vision response as JSON: %s", text[:200])

        # Fallback: try to extract food names from text
        return [
            {
                "food": "Parse error — raw response available",
                "portion": "N/A",
                "confidence": 0.0,
            }
        ]

    @staticmethod
    def _vision_fallback(error_code: str) -> dict:
        return {
            "food": "Photo analysis unavailable",
            "portion": "N/A",
            "confidence": 0.0,
            "tier_label": "AI estimate",
            "error": error_code,
        }

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        if not self._client:
            logger.warning("Embedding call skipped: no OpenAI client")
            return []

        response = await self._client.embeddings.create(
            model=self.EMBEDDING_MODEL,
            input=[text],
        )
        logger.info("Embedding call: model=%s, text_len=%d", self.EMBEDDING_MODEL, len(text))
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str], batch_size: int = 2000) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        import asyncio

        if not self._client:
            logger.warning("Batch embedding call skipped: no OpenAI client")
            return [[] for _ in texts]

        all_embs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = await self._client.embeddings.create(model=self.EMBEDDING_MODEL, input=chunk)
            all_embs.extend([item.embedding for item in resp.data])
            if i + batch_size < len(texts):
                await asyncio.sleep(0.3)

        logger.info("Batch embedding: model=%s, count=%d", self.EMBEDDING_MODEL, len(texts))
        return all_embs


# ---------------------------------------------------------------------------
# Module-level embedding helpers (W2-1 / W2-1b) — used by semantic_search.py
# ---------------------------------------------------------------------------
# These describe the OFFLINE fallback dimensions. The OpenAI path uses its own
# OPENAI_EMBEDDING_* constants and writes to a separate sidecar table
# (food_embeddings_oai). Ported from pillar3-wip; the offline embedder is used
# only by benchmarks/tests today — the shipped /search/semantic endpoint uses
# lexical fallback until embeddings are populated (a gated follow-up).
EMBEDDING_MODEL_ID = "offline-trigramhash-v1"
EMBEDDING_DIM = 384

OPENAI_EMBEDDING_MODEL_ID = "openai-text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536


def _embed_text_offline(text: str, dim: int = 384) -> list[float]:
    """Deterministic offline embedding fallback (char-trigram hash + L2 norm).

    Model identity logged as 'offline-trigramhash-v1'. Dim is 384.

    NOTE: uses a stable hash (not the process-randomized builtin ``hash``) so
    vectors are reproducible across processes — required because embeddings are
    built in one process and queried in another.
    """
    import hashlib
    import math

    text = (text or "").lower().strip()
    vec = [0.0] * dim
    if not text:
        return vec
    padded = f"  {text}  "
    grams = [padded[i : i + 3] for i in range(len(padded) - 2)]
    words = [w for w in "".join(c if c.isalnum() else " " for c in text).split() if w]
    for tok in grams + words:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _embed_text_openai(text: str) -> list[float]:
    """Live OpenAI text-embedding-3-small call (dim=1536). Raises on any failure.

    Reads the API key from env var OPENAI_API_KEY (or the app settings fallback).
    Short timeout (15s). Caller is responsible for catching exceptions to fall
    back to the offline path.
    """
    import os

    import httpx

    api_key = os.environ.get("OPENAI_API_KEY") or getattr(settings, "openai_api_key", "") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not configured")
    text = (text or "").strip()
    if not text:
        return [0.0] * OPENAI_EMBEDDING_DIM
    resp = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": "text-embedding-3-small", "input": text},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()
    vec = data["data"][0]["embedding"]
    if len(vec) != OPENAI_EMBEDDING_DIM:
        raise RuntimeError(f"unexpected embedding dim {len(vec)} from OpenAI")
    return vec


def embed_text(text: str, dim: int = 384, provider: str = "openai") -> list[float]:
    """Embed text using the requested provider, with automatic fallback.

    provider="openai" (default): call OpenAI text-embedding-3-small (1536-dim).
        On ANY exception (missing key, network, rate limit) fall back to the
        offline trigram-hash embedder. Note: the return dim in fallback mode is
        ``dim`` (default 384), NOT 1536 — callers that commingle vectors must
        check the length.
    provider="offline": use the deterministic trigram-hash embedder (dim arg
        honored; default 384). Used by tests and the offline benchmark path.
    """
    import sys

    if provider == "offline":
        return _embed_text_offline(text, dim=dim)
    try:
        return _embed_text_openai(text)
    except Exception as e:  # noqa: BLE001
        print(
            f"[embed_text] OpenAI embedder failed ({type(e).__name__}): falling back to offline",
            file=sys.stderr,
        )
        return _embed_text_offline(text, dim=dim)


def get_llm_provider() -> LLMProvider:
    """Factory returning the configured LLM provider (single provider on main)."""
    return LLMProvider()
