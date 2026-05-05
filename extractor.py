"""
Vision-LLM extraction of structured metadata from architectural drawing title blocks.
"""

import base64
import json
import logging
import re
import time
from io import BytesIO
import requests
from PIL import Image
from pydantic import BaseModel, field_validator

logger = logging.getLogger("title_block_extractor")

BASE_URL = "http://localhost:11434"
MODEL = "qwen3.5:9b"
NUM_CTX = 4096
TEMPERATURE = 0
TIMEOUT = 120
FALLBACK = True

_PROMPT_SHARED = """You are an expert Architectural Data Analyst specializing in CAD drawings and architectural documentation. You possess strong OCR capabilities and understand various languages found in technical drawings (German, French, English).

Your task is to Analyze the provided architectural drawing image. Identify the Title Block (usually bottom-right or right-side) and extract the following metadata fields.

Extraction rules:
{scope_rule}
2. Plan No: Look strictly for fields labelled exactly: "Plan N°", "Plannummer", "Plan Nr.", "Dateibezeichnung", "Zeichnung Nr.", or "File No.". If none of these labels are visible, return null. Do NOT infer a plan number from unlabelled strings.
3. Plan Name: Look for fields labelled "Bezeichnung", "Plantitel", "Beschrieb", or "Planname". If no label is found, identify the plan name as a descriptive text string in the title block — typically larger font, centred, and describing the drawing content (e.g. "Südfassade Detail"). Do NOT pick up project names, addresses, or firm names.
4. Scale: Extract all distinct drawing scales (e.g., 1:50, 1:100). Return a list of strings, e.g., ["1:50", "1:100"].
5. Date: Extract all drawing dates.
   - Format each date as dd.mm.yyyy. Always expand 2-digit years to 4 digits. A year of '13' on an architectural drawing is almost certainly 2013, not 1913.
   - If only month and year are present format as mm.yyyy.
   - Return a list of strings, e.g., ["11.05.1960", "09.03.2025"].
6. BKP Code: Extract Swiss BKP cost codes (look for "BKP" + digits). These are 1-4 digit numeric codes, optionally followed by a dot and one more digit for 4th-level codes (e.g. "2", "25", "256", "256.2"). Return the code exactly as it appears (e.g. "256.2"), without the "BKP" label.
7. Null Handling: If a field is missing, unreadable, or does not exist, use `null` (not the string "null").

Output constraints:
1. Format: Return a raw JSON object ONLY.
2. No Markdown: Do NOT wrap the output in markdown code blocks (```json ... ```).
3. No Explanations: Do not add text, comments, or thoughts before or after the JSON.
4. Valid JSON: Ensure all brackets and quotes are balanced.

JSON schema:
{{
  "plan_no": "string | null",
  "plan_name": "string | null",
  "scale": "string[] | null",
  "date": "string[] | null",
  "bkp_code": "string[] | null"
}}"""

PROMPT_CROP = _PROMPT_SHARED.format(
    scope_rule="1. Scope: The image is the bottom-right corner of a full architectural drawing. Focus ONLY on the Title Block. Ignore dimensions, wall notes, or signatures outside this panel."
)

PROMPT_FULL = _PROMPT_SHARED.format(
    scope_rule="1. Scope: This is a full architectural drawing. The title block or metadata may be anywhere — bottom-right corner, right edge, bottom strip, or scattered in the border area. Search the entire image for the title block and extract metadata wherever it appears."
)


# - - - - - - - - - - DATA MODELS - - - - - - - - - -

class ModelResponse(BaseModel):
    """Validated metadata returned by the vision model before pipeline post-processing."""

    plan_no: str | None = None
    plan_name: str | None = None
    scale: list[str] | None = None
    date: list[str] | None = None
    bkp_code: list[str] | None = None

    @field_validator("scale", "date", "bkp_code", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and len(v) == 0:
            return None
        return v

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        normalized = [cls._normalize_scale(s) for s in v]
        validated = [s for s in normalized if re.match(r"^1:[\d.,]+$", s)]
        return validated or None
    
    @staticmethod
    def _normalize_scale(scale_str: str) -> str:
        s = re.sub(r"\s*:\s*", ":", scale_str)
        return s

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        normalized = [cls._normalize_date(d) for d in v]
        validated = [d for d in normalized if re.match(r"^(\d{2}\.\d{2}\.\d{4}|\d{2}\.\d{4})$", d)]
        return validated or None

    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Fix common model transpositions before regex validation."""
        # expand 2-digit years: dd.mm.yy → dd.mm.yyyy, mm.yy → mm.yyyy
        m = re.match(r"^(\d{2}\.\d{2}\.)(\d{2})$", date_str)
        if m:
            yy = int(m.group(2))
            yyyy = 2000 + yy if yy <= 30 else 1900 + yy
            return f"{m.group(1)}{yyyy}"
        m = re.match(r"^(\d{2}\.)(\d{2})$", date_str)
        if m:
            yy = int(m.group(2))
            yyyy = 2000 + yy if yy <= 30 else 1900 + yy
            return f"{m.group(1)}{yyyy}"
        # fix transposed year-month: yyyy.mm → mm.yyyy
        m = re.match(r"^(\d{4})\.(\d{2})$", date_str)
        if m:
            return f"{m.group(2)}.{m.group(1)}"
        return date_str

    @field_validator("bkp_code")
    @classmethod
    def validate_bkp_code(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        pattern = re.compile(r"^(\d{1,4}|\d{3}\.\d)$")
        validated = [c for c in v if pattern.match(c)]
        return validated or None


class TitleBlockResult(BaseModel):
    """Final extraction payload returned to callers, including pipeline diagnostics."""
    path: str | None = None
    plan_no: str | None = None
    plan_name: str | None = None
    scale: list[str] | None = None
    date: list[str] | None = None
    bkp_code: list[str] | None = None
    notes: list[str] = []


# - - - - - - - - - - EXTRACTOR - - - - - - - - - -

class TitleBlockExtractor:
    def __init__(
        self,
        model: str = MODEL,
        base_url: str = BASE_URL,
        num_ctx: int = NUM_CTX,
        temperature: int = TEMPERATURE,
        timeout: int = TIMEOUT,
        fallback: bool = FALLBACK,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout
        self.fallback = fallback

    def _infer(self, image_b64: str, prompt: str, notes: list[str]) -> ModelResponse | None:
        """Run one Ollama inference call and validate the returned JSON payload.

        Returns a `ModelResponse` when the model response can be parsed and validated.
        On transport, parsing, or schema issues, returns `None` and appends a short
        diagnostic message to `notes`.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
            "think": False,
        }
        try:
            logger.debug("Ollama request sent")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            logger.debug("Ollama response received")
        except requests.ConnectionError:
            msg = f"could not connect to Ollama at {self.base_url}"
            logger.error(msg)
            notes.append(msg)
            return None
        except requests.Timeout:
            msg = "Ollama request timed out"
            logger.error(msg)
            notes.append(msg)
            return None
        except Exception as e:
            msg = f"unexpected error: {e}"
            logger.error(msg)
            notes.append(msg)
            return None

        raw_text = response.json().get("response", "")
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip()

        try:
            raw_dict = json.loads(raw_text)
            logger.debug("JSON parsed successfully")
        except json.JSONDecodeError:
            msg = f"model returned malformed JSON -> {raw_text}"
            logger.warning(msg)
            notes.append(msg)
            return None

        if not isinstance(raw_dict, dict):
            msg = f"model returned non-object JSON: {raw_text}"
            logger.warning(msg)
            notes.append(msg)
            return None

        expected_keys = {"plan_no", "plan_name", "scale", "date", "bkp_code"}
        missing = expected_keys - raw_dict.keys()
        if missing:
            logger.warning("Missing keys filled with null: %s", missing)
            for key in missing:
                raw_dict[key] = None

        validated = ModelResponse(**{k: raw_dict.get(k) for k in expected_keys})

        for field in ("scale", "date", "bkp_code"):
            raw_val = raw_dict.get(field)
            if raw_val and getattr(validated, field) is None:
                msg = f"{field} failed validation: got '{raw_val}'"
                logger.warning(msg)
                notes.append(msg)

        return validated

    def _encode_image(self, image: Image.Image) -> str:
        """Encode a PIL image as base64 PNG for the Ollama image API."""
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def extract(self, crop: Image.Image, thumbnail: Image.Image | None = None) -> dict:
        """Extract title block metadata from a crop and optional full-page thumbnail.

        The crop is expected to contain the bottom-right title block region. If the
        crop result lacks the key descriptive fields, the extractor can retry on the
        thumbnail to search the full drawing border area. The returned dict contains
        validated metadata plus `fallback` and diagnostic `notes`.
        """
        _start = time.perf_counter()
        logger.debug(f"extract() called — crop size: {crop.size}, thumbnail size: {thumbnail.size if thumbnail else None}")

        notes: list[str] = []

        image_b64 = self._encode_image(crop)
        validated = self._infer(image_b64, PROMPT_CROP, notes) or ModelResponse()

        fallback_fired = False
        insufficient = validated.plan_name is None and validated.date is None and validated.scale is None
        if self.fallback and insufficient:
            if thumbnail is None:
                logger.warning("fallback triggered but no thumbnail provided")
            else:
                logger.info("crop result insufficient, firing fallback")
                try:
                    thumb_b64 = self._encode_image(thumbnail)
                    fb_validated = self._infer(thumb_b64, PROMPT_FULL, notes)
                    if fb_validated is not None:
                        logger.info("fallback inference complete")
                        for field in ("plan_no", "plan_name", "scale", "date", "bkp_code"):
                            if getattr(validated, field) is None and getattr(fb_validated, field) is not None:
                                setattr(validated, field, getattr(fb_validated, field))
                        logger.debug("fallback merged into crop result")
                        fallback_fired = True
                    else:
                        msg = f"fallback inference returned: {fb_validated}"
                        logger.debug(msg)
                        notes.append(msg)
                except Exception as e:
                    msg = f"fallback failed: {e}"
                    logger.warning(msg)
                    notes.append(msg)

        result = TitleBlockResult(
            plan_no=validated.plan_no,
            plan_name=validated.plan_name,
            scale=validated.scale,
            date=validated.date,
            bkp_code=validated.bkp_code,
            notes=notes,
        )
        output = result.model_dump()
        output["fallback"] = fallback_fired
        logger.info("Extraction complete in %.3fs", time.perf_counter() - _start)
        return output
