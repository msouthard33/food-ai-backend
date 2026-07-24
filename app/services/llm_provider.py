"""LLM provider wrapper — vision + embedding + text completions.

Supports OpenAI for embeddings and vision (gpt-4o-mini).
Never logs image bytes. Never stores photos to disk.
"""

import base64
import json
import logging
import time
from typing import Any

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
            "Format: [{\"food\": \"<food name>\", \"portion\": \"<estimated portion>\", "
            "\"confidence\": <0.0-1.0>}]\n\n"
            "Rules:\n"
            "- Be specific: 'grilled chicken breast' not just 'chicken'\n"
            "- Estimate portions in common units (cup, oz, piece, slice, tablespoon)\n"
            "- Confidence: 0.9+ for clearly visible items, 0.5-0.8 for partially visible, "
            "<0.5 for guesses\n"
            "- Include sauces, dressings, and condiments as separate items\n"
            "- If the image is not food, return [{\"food\": \"Not a food image\", "
            "\"portion\": \"N/A\", \"confidence\": 0.0}]"
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
                logger.info("Vision cost estimate: $%.6f (input=$%.6f, output=$%.6f)",
                            total_cost, input_cost, output_cost)
                if total_cost > 0.01:
                    logger.warning("COST GATE: photo analysis cost $%.4f exceeds $0.01 threshold",
                                   total_cost)

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
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # Validate structure
                result = []
                for item in parsed:
                    result.append({
                        "food": str(item.get("food", "Unknown")),
                        "portion": str(item.get("portion", "Unknown")),
                        "confidence": float(item.get("confidence", 0.5)),
                    })
                return result
            elif isinstance(parsed, dict) and "foods" in parsed:
                return self._parse_vision_response(json.dumps(parsed["foods"]))
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse vision response as JSON: %s", text[:200])

        # Fallback: try to extract food names from text
        return [{"food": "Parse error — raw response available", "portion": "N/A", "confidence": 0.0}]

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
            resp = await self._client.embeddings.create(
                model=self.EMBEDDING_MODEL, input=chunk
            )
            all_embs.extend([item.embedding for item in resp.data])
            if i + batch_size < len(texts):
                await asyncio.sleep(0.3)

        logger.info("Batch embedding: model=%s, count=%d", self.EMBEDDING_MODEL, len(texts))
        return all_embs
