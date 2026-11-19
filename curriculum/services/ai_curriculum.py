import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone as dj_timezone
from django.utils.text import slugify
from decouple import config

from curriculum.models import Subject, Module, Topic, TopicKeyword, GenerationJob

logger = logging.getLogger(__name__)


# =====================================================================
# Custom Exception Hierarchy
# =====================================================================

class CurriculumGenerationError(Exception):
    """Base exception for curriculum generation failures."""
    pass


class GeminiTransientError(CurriculumGenerationError):
    """Raised when transient errors (503, 500, 429) persist after all retries."""
    pass

AITransientError = GeminiTransientError


class AIRateLimitError(AITransientError):
    """Raised when Groq API rate limit (HTTP 429) is exceeded after retries."""
    pass


class AIServerError(AITransientError):
    """Raised when Groq returns 5xx server errors after retries."""
    pass


class AINetworkError(CurriculumGenerationError):
    """Raised when network, connection reset, or timeout errors occur contacting Groq."""
    pass


class GeminiAuthError(CurriculumGenerationError):
    """Raised for authentication or permission errors (HTTP 401, 403)."""
    pass

AIAuthError = GeminiAuthError


class AIPermissionError(AIAuthError):
    """Raised when Groq API returns HTTP 403 (model permission or account tier restriction)."""
    pass


class GeminiInvalidRequestError(CurriculumGenerationError):
    """Raised for bad or malformed requests (HTTP 400)."""
    pass

AIInvalidRequestError = GeminiInvalidRequestError


class GeminiModelNotFoundError(CurriculumGenerationError):
    """Raised when the requested model is not found or decommissioned (HTTP 404 / 400)."""
    pass

AIModelNotFoundError = GeminiModelNotFoundError


class CurriculumParseError(CurriculumGenerationError):
    """Raised when the AI response cannot be parsed into a valid curriculum structure."""
    pass


class CurriculumValidationError(CurriculumGenerationError):
    """Raised when the generated curriculum fails validation against the prescribed source syllabus."""
    pass


class CurriculumFencingError(CurriculumGenerationError):
    """Raised when job claim token or lease validation fails during curriculum persistence."""
    pass


# =====================================================================
# Helper Functions: Sanitization, Timing, and Model Resolution
# =====================================================================

DISALLOWED_MODEL_CAPABILITIES = ("tts", "embedding", "audio", "imagen", "robotics", "whisper")


def is_valid_text_generation_model(model_name: str) -> bool:
    """
    Verifies that a model identifier is intended for text and structured JSON generation,
    strictly excluding TTS, embedding, or audio models.
    """
    if not model_name or not isinstance(model_name, str):
        return False
    lower = model_name.strip().lower()
    return not any(disallowed in lower for disallowed in DISALLOWED_MODEL_CAPABILITIES)


def sanitize_error_message(response_or_text) -> str:
    """
    Sanitizes error text to ensure API keys or credentials are never exposed in logs or user messages.
    """
    if isinstance(response_or_text, requests.Response):
        try:
            data = response_or_text.json()
            err = data.get("error", {})
            msg = err.get("message") or err.get("status") or response_or_text.text
        except Exception:
            msg = response_or_text.text
    else:
        msg = str(response_or_text)

    # Scrub query param keys or raw Google/Groq API keys
    msg = re.sub(r'key=[A-Za-z0-9_\-]+', 'key=[REDACTED]', msg)
    msg = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_API_KEY]', msg)
    msg = re.sub(r'gsk_[0-9A-Za-z_-]{10,}', '[REDACTED_API_KEY]', msg)
    msg = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', msg, flags=re.IGNORECASE)
    return msg.strip()[:300]


def normalize_groq_model_name(model_name: str) -> str:
    """
    Normalizes shorthand Groq model names to official production identifiers.
    Automatically redirects decommissioned models (such as llama-3.3-70b-versatile)
    to the active production model 'openai/gpt-oss-120b'.
    """
    if not model_name:
        return "openai/gpt-oss-120b"
    name = model_name.strip()
    clean = name.lower().replace("_", "-")

    # Flagship OpenAI GPT-OSS on Groq
    if clean in ("openai/gpt-oss-120b", "gpt-oss-120b", "openai-gpt-oss-120b", "gpt-oss", "oss-120b", "120b", "openai/gpt-oss"):
        return "openai/gpt-oss-120b"
    if clean in ("openai/gpt-oss-20b", "gpt-oss-20b", "openai-gpt-oss-20b", "oss-20b", "20b"):
        return "openai/gpt-oss-20b"

    # Automatically migrate decommissioned Llama 3.3 70B model to supported production model
    if clean in ("llama-3.3-70b-versatile", "llama-3.3-70b", "llama3.3-70b", "llama-70b", "llama3-70b", "llama-3.3-70b-instruct"):
        logger.warning(
            "Deprecated Groq model '%s' requested (decommissioned August 2026); migrating to 'openai/gpt-oss-120b'.",
            name
        )
        return "openai/gpt-oss-120b"

    if clean in ("llama-3.1-8b", "llama3.1-8b", "llama-8b", "llama3-8b", "llama-3.1-8b-instant", "llama-3.1-8b-instruct"):
        return "llama-3.1-8b-instant"
    if clean in ("mixtral", "mixtral-8x7b", "mixtral-8x7b-instruct", "mixtral-8x7b-32768"):
        return "mixtral-8x7b-32768"

    return name


def resolve_ai_provider() -> tuple[str, str]:
    """
    Determines active AI provider and API key.
    Prioritizes Groq API via GROQ_API_KEY.
    Returns (provider_name, api_key), where provider_name is 'groq' or 'gemini'.
    Raises AIAuthError if no configured key is found.
    """
    raw_groq = (
        getattr(settings, 'GROQ_API_KEY', None)
        or config('GROQ_API_KEY', default='')
    )
    if isinstance(raw_groq, str):
        raw_groq = raw_groq.strip().strip('\'"')
    else:
        raw_groq = ''

    raw_gemini = (
        getattr(settings, 'GEMINI_API_KEY', None)
        or config('GEMINI_API_KEY', default='')
    )
    if isinstance(raw_gemini, str):
        raw_gemini = raw_gemini.strip().strip('\'"')
    else:
        raw_gemini = ''

    provider_pref = config('AI_PROVIDER', default='auto').strip().lower()

    if provider_pref == 'gemini' and raw_gemini:
        return 'gemini', raw_gemini

    # Groq is our primary AI engine
    if raw_groq:
        return 'groq', raw_groq

    if raw_gemini:
        return 'gemini', raw_gemini

    raise AIAuthError(
        "GROQ_API_KEY is not configured in your environment. "
        "Please add GROQ_API_KEY to your .env file or production environment variables."
    )


def calculate_backoff_delay(
    attempt: int,
    retry_after_header: str = None,
    base_delay: float = 1.0,
    max_delay: float = 8.0
) -> float:
    """
    Computes delay using bounded exponential backoff with jitter, respecting Retry-After when available.
    attempt is 1-indexed (1 for first retry, 2 for second, etc.).
    """
    if retry_after_header:
        try:
            val = float(str(retry_after_header).strip())
            if val > 0:
                return min(max(val, 0.5), max_delay)
        except (ValueError, TypeError):
            try:
                target = parsedate_to_datetime(str(retry_after_header).strip())
                now = datetime.now(timezone.utc)
                diff = (target - now).total_seconds()
                if diff > 0:
                    return min(diff, max_delay)
            except Exception:
                pass

    exp_delay = base_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0.1, 0.4)
    return min(exp_delay + jitter, max_delay)


def resolve_configured_models(api_key: str) -> tuple:
    """
    Determines primary and fallback Gemini models.
    Prioritizes explicit environment configuration (GEMINI_PRIMARY_MODEL / GEMINI_FALLBACK_MODEL),
    strictly filtering out TTS, embedding, or audio models.
    If not specified or invalid, queries Google's ModelService.ListModels to detect active text models
    supporting generateContent for this key, ensuring valid model IDs.
    Returns (primary_model, fallback_model).
    """
    raw_primary = config('GEMINI_PRIMARY_MODEL', default=config('GEMINI_MODEL', default='')).strip()
    raw_fallback = config('GEMINI_FALLBACK_MODEL', default='').strip()

    primary_model = raw_primary.replace('models/', '') if raw_primary else None
    fallback_model = raw_fallback.replace('models/', '') if raw_fallback else None

    # Validate that models are text-capable and not TTS/audio
    if primary_model and not is_valid_text_generation_model(primary_model):
        logger.warning(
            "Configured primary model '%s' is unsuitable for text/JSON curriculum generation (contains TTS/audio capability). Ignoring.",
            primary_model
        )
        primary_model = None

    if fallback_model and not is_valid_text_generation_model(fallback_model):
        logger.warning(
            "Configured fallback model '%s' is unsuitable for text/JSON curriculum generation (contains TTS/audio capability). Ignoring.",
            fallback_model
        )
        fallback_model = None

    if primary_model and fallback_model and primary_model != fallback_model:
        logger.info(
            "Using environment-configured Gemini models - Primary: '%s', Fallback: '%s'",
            primary_model, fallback_model
        )
        return primary_model, fallback_model

    discovered = []
    if api_key:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }
        try:
            res = requests.get(list_url, headers=headers, timeout=12)
            if res.status_code == 200:
                models_data = res.json().get("models", [])
                supported = [
                    m["name"].replace("models/", "")
                    for m in models_data
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                    and is_valid_text_generation_model(m.get("name", ""))
                ]
                discovered = supported
            elif res.status_code in (401, 403):
                err_msg = sanitize_error_message(res)
                logger.error("ListModels authentication failed (Status %s): %s", res.status_code, err_msg)
                raise GeminiAuthError(f"Google API key authentication failed ({res.status_code}): {err_msg}")
            else:
                logger.warning("ListModels returned status %s: %s", res.status_code, sanitize_error_message(res))
        except GeminiAuthError:
            raise
        except requests.RequestException as e:
            logger.warning("ListModels request failed (%s). Using verified model defaults.", type(e).__name__)

    priority_order = ("3.5-flash", "3.5-flash-lite", "2.0-flash", "1.5-flash", "1.5-flash-8b", "flash")

    if not primary_model:
        if discovered:
            for pref in priority_order:
                match = next((m for m in discovered if pref in m.lower()), None)
                if match:
                    primary_model = match
                    break
            if not primary_model:
                primary_model = discovered[0]
        else:
            primary_model = "gemini-1.5-flash"

    if not fallback_model or fallback_model == primary_model:
        if discovered:
            for pref in priority_order:
                match = next((m for m in discovered if pref in m.lower() and m != primary_model), None)
                if match:
                    fallback_model = match
                    break
        if not fallback_model or fallback_model == primary_model:
            if primary_model == "gemini-1.5-flash":
                fallback_model = "gemini-2.0-flash"
            else:
                fallback_model = "gemini-1.5-flash"

    logger.info(
        "Active Gemini models resolved - Primary: '%s', Fallback: '%s'",
        primary_model, fallback_model
    )
    return primary_model, fallback_model


def resolve_active_gemini_endpoint(api_key: str) -> str:
    """Backwards-compatible endpoint resolver for existing callers."""
    primary, _ = resolve_configured_models(api_key)
    return f"https://generativelanguage.googleapis.com/v1beta/models/{primary}:generateContent"


def resolve_groq_models(api_key: str) -> tuple[str, str]:
    """
    Determines primary and fallback Groq models.
    Prioritizes centralized configuration:
      settings.GROQ_MODEL / GROQ_MODEL / GROQ_PRIMARY_MODEL (default: 'openai/gpt-oss-120b')
      settings.GROQ_FALLBACK_MODEL / GROQ_FALLBACK_MODEL (default: 'openai/gpt-oss-20b')
    If not explicitly specified, optionally queries Groq's /models endpoint to discover
    active production models for this account.
    Returns (primary_model, fallback_model).
    """
    configured_primary = config(
        'GROQ_MODEL',
        default=config('GROQ_PRIMARY_MODEL', default=getattr(settings, 'GROQ_MODEL', 'openai/gpt-oss-120b'))
    )
    raw_primary = str(configured_primary).strip() if configured_primary else 'openai/gpt-oss-120b'

    configured_fallback = config(
        'GROQ_FALLBACK_MODEL',
        default=getattr(settings, 'GROQ_FALLBACK_MODEL', 'openai/gpt-oss-20b')
    )
    raw_fallback = str(configured_fallback).strip() if configured_fallback else 'openai/gpt-oss-20b'

    primary_model = normalize_groq_model_name(raw_primary)
    fallback_model = normalize_groq_model_name(raw_fallback)

    if primary_model and fallback_model and primary_model != fallback_model:
        logger.info(
            "Using configured Groq models - Primary: '%s', Fallback: '%s'",
            primary_model, fallback_model
        )
        return primary_model, fallback_model

    discovered = []
    if api_key:
        list_url = "https://api.groq.com/openai/v1/models"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            res = requests.get(list_url, headers=headers, timeout=10)
            if res.status_code == 200:
                models_data = res.json().get("data", [])
                supported = [
                    m["id"]
                    for m in models_data
                    if m.get("active", True)
                    and not any(disallowed in m.get("id", "").lower() for disallowed in ("whisper", "tts", "embedding", "audio"))
                ]
                discovered = supported
                logger.info("Discovered %d active Groq models for this account.", len(discovered))
            elif res.status_code in (401, 403):
                err_msg = sanitize_error_message(res)
                logger.error("Groq ListModels authentication failed (%s): %s", res.status_code, err_msg)
                raise AIAuthError(f"Groq API key authentication failed ({res.status_code}): {err_msg}")
            else:
                logger.warning("Groq ListModels returned status %s: %s", res.status_code, sanitize_error_message(res))
        except AIAuthError:
            raise
        except requests.RequestException as e:
            logger.warning("Groq ListModels request failed (%s). Using verified model defaults.", type(e).__name__)

    groq_priority_order = (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "qwen-2.5-32b",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it",
    )

    if not primary_model:
        if discovered:
            for pref in groq_priority_order:
                match = next((m for m in discovered if pref in m.lower()), None)
                if match:
                    primary_model = match
                    break
            if not primary_model:
                primary_model = discovered[0]
        else:
            primary_model = "openai/gpt-oss-120b"

    if not fallback_model or fallback_model == primary_model:
        if discovered:
            for pref in groq_priority_order:
                match = next((m for m in discovered if pref in m.lower() and m != primary_model), None)
                if match:
                    fallback_model = match
                    break
        if not fallback_model or fallback_model == primary_model:
            if primary_model == "openai/gpt-oss-120b":
                fallback_model = "openai/gpt-oss-20b"
            else:
                fallback_model = "llama-3.1-8b-instant"

    logger.info("Active Groq models resolved - Primary: '%s', Fallback: '%s'", primary_model, fallback_model)
    return primary_model, fallback_model


# =====================================================================
# Source Syllabus Parser & Verified Lookup
# =====================================================================

def _extract_topics_from_text(raw: str) -> list:
    """
    Extracts individual topic phrases from a module's raw syllabus text.
    Handles bullets, chapter headers, commas, and semicolons.
    """
    topics = []
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    for line in lines:
        bullet_match = re.match(r"^[-*•–—\d.]+\s*(.*)$", line)
        cleaned_line = bullet_match.group(1).strip() if bullet_match else line

        # Strip prefixes like "Topics:", "Content:", "Chapter 1:"
        cleaned_line = re.sub(
            r"^(?:topics?|contents?|syllabus|chapters?|\bchp\b[\s\d\-]*)\s*[:\-–—]\s*",
            "",
            cleaned_line,
            flags=re.IGNORECASE
        ).strip()

        if not cleaned_line:
            continue

        if "," in cleaned_line or ";" in cleaned_line:
            parts = re.split(r"[,;]+", cleaned_line)
            for p in parts:
                p_clean = p.strip()
                if len(p_clean) >= 3 and not p_clean.lower().startswith("chp-"):
                    topics.append(p_clean)
        else:
            if len(cleaned_line) >= 3 and not cleaned_line.lower().startswith("chp-"):
                topics.append(cleaned_line)

    return topics[:15]


def _extract_chapters_from_text(raw: str) -> list:
    """
    Parses a module's raw syllabus text into structured chapters and topics.
    Recognizes patterns such as:
      - 'CHP-1' or 'Chapter 1: ...'
      - Multiple paragraphs separated by blank lines or titled headers (e.g. 'Public-Key Management:')
    Returns a list of dicts:
      [
        {
          'chapter_number': 1,
          'chapter_title': 'Key Management Fundamentals',
          'topics': ['Key lengths and lifetimes', 'Key generation', ...]
        },
        ...
      ]
    """
    if not raw or not raw.strip():
        return []

    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    chp_blocks = []
    curr_num = 1
    curr_title = ""
    curr_lines = []

    chp_header_re = re.compile(
        r"^(?:chp|chapter|part|section)\s*[-:]?\s*(\d+|[ivx]+)\s*[:\-–—]?\s*(.*)$",
        re.IGNORECASE
    )

    has_explicit_chp = any(chp_header_re.match(l) for l in lines)

    if has_explicit_chp:
        for line in lines:
            m = chp_header_re.match(line)
            if m:
                if curr_lines or curr_title:
                    chp_blocks.append((curr_num, curr_title, curr_lines))
                num_str = m.group(1).lower()
                roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
                curr_num = roman.get(num_str, int(num_str) if num_str.isdigit() else len(chp_blocks) + 1)
                curr_title = m.group(2).strip()
                curr_lines = []
            else:
                curr_lines.append(line)
        if curr_lines or curr_title:
            chp_blocks.append((curr_num, curr_title, curr_lines))
    else:
        # Check paragraphs separated by double-newlines
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            for idx, p in enumerate(paragraphs, start=1):
                p_lines = [l.strip() for l in p.split("\n") if l.strip()]
                first_line = p_lines[0]
                rest_lines = p_lines[1:]
                c_title = ""
                if ":" in first_line:
                    parts = first_line.split(":", 1)
                    c_title = parts[0].strip()
                    first_line = parts[1].strip()
                chp_blocks.append((idx, c_title, [first_line] + rest_lines))
        else:
            chp_blocks.append((1, "", lines))

    result = []
    for c_num, c_title, c_lines in chp_blocks:
        body = " ".join(c_lines).strip()
        if not c_title and body:
            if ":" in body:
                parts = body.split(":", 1)
                c_title = parts[0].strip()
                body = parts[1].strip()
            elif "," in body:
                parts = body.split(",", 1)
                first_part = parts[0].strip()
                if len(first_part.split()) <= 6:
                    c_title = first_part
                    body = parts[1].strip()

        # Clean title
        c_title = re.sub(r"^(?:topics?|contents?|syllabus)\s*[:\-–—]?\s*", "", c_title, flags=re.IGNORECASE).strip()

        topics = []
        for t in re.split(r"[,;]+", body):
            clean_t = re.sub(r"^[.\s\-–—]+|[.\s\-–—]+$", "", t).strip()
            if len(clean_t) >= 3 and not clean_t.lower().startswith("chp-"):
                topics.append(clean_t)

        result.append({
            "chapter_number": c_num,
            "chapter_title": c_title,
            "topics": topics[:15]
        })

    return result


def parse_prescribed_syllabus(raw_text: str) -> list:
    """
    Parses unstructured or semi-structured syllabus text into prescribed modules and topics.
    Supports formats like:
      - 'Module 1: Classical Encryption & Block Ciphers'
      - 'MODULE I - Classical Encryption & Block Ciphers'
      - 'Unit 1: Classical Encryption & Block Ciphers'
      - '1. Classical Encryption & Block Ciphers'
      - Exactly 5 lines of module titles
    Returns a list of dicts: [{'module_number': 1, 'title': '...', 'raw_text': '...', 'topics': [...]}, ...]
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip().replace("\r\n", "\n")
    lines = [l.rstrip() for l in text.split("\n")]

    roman_map = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}

    header_pattern = re.compile(
        r"^(?:\s*#+\s*)?"
        r"(?:(?:module|unit|chapter)\s*([1-5]|i|ii|iii|iv|v)\b[\s:.\-–—]*|"
        r"([1-5])[\s:.\-–—]+)"
        r"(.*)$",
        re.IGNORECASE
    )

    modules_found = []
    current_mod = None
    current_content = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        match = header_pattern.match(stripped)
        if match:
            num_str = match.group(1) or match.group(2)
            title = match.group(3).strip()
            title = re.sub(r"^[\s:.\-–—]+", "", title).strip()

            num_lower = num_str.lower()
            mod_num = roman_map.get(num_lower, None)
            if mod_num is None:
                try:
                    mod_num = int(num_lower)
                except ValueError:
                    mod_num = len(modules_found) + 1

            if current_mod is not None:
                current_mod["raw_text"] = "\n".join(current_content).strip()
                modules_found.append(current_mod)

            current_mod = {
                "module_number": mod_num,
                "title": title or f"Module {mod_num}",
                "raw_text": "",
                "topics": []
            }
            current_content = []
        else:
            if current_mod is not None:
                current_content.append(stripped)

    if current_mod is not None:
        current_mod["raw_text"] = "\n".join(current_content).strip()
        modules_found.append(current_mod)

    # Fallback: What if no "Module X" headings matched, but there are exactly 5 non-empty lines?
    if len(modules_found) != 5:
        candidate_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        if len(candidate_lines) == 5:
            modules_found = []
            for idx, cline in enumerate(candidate_lines, start=1):
                clean_title = re.sub(r"^(?:module|unit)?\s*[0-9iIvVxX]+[\s:.\-–—]+", "", cline, flags=re.IGNORECASE).strip()
                modules_found.append({
                    "module_number": idx,
                    "title": clean_title or cline,
                    "raw_text": "",
                    "topics": []
                })

    for mod in modules_found:
        raw = mod.get("raw_text", "")
        if raw:
            mod["chapters"] = _extract_chapters_from_text(raw)
            mod["topics"] = _extract_topics_from_text(raw)

    modules_found.sort(key=lambda m: m["module_number"])
    return modules_found


_VERIFIED_SYLLABUS_CACHE = None

def get_verified_syllabus_source(course_code: str, subject_name: str) -> list | None:
    """
    Checks verified curriculum seed records for known official university course codes (e.g. BAD703, BAI701).
    Returns structured prescribed modules if a verified official syllabus is available.
    """
    global _VERIFIED_SYLLABUS_CACHE
    if _VERIFIED_SYLLABUS_CACHE is None:
        _VERIFIED_SYLLABUS_CACHE = {}
        seed_path = Path(__file__).resolve().parent.parent.parent / "curriculum_seed.json"
        if seed_path.exists():
            try:
                with open(seed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                subjects = {item["pk"]: item["fields"] for item in data if item["model"] == "curriculum.subject"}
                modules_by_subj = {}
                for item in data:
                    if item["model"] == "curriculum.module":
                        s_id = item["fields"]["subject"]
                        modules_by_subj.setdefault(s_id, []).append(item["fields"])

                for s_id, s_fields in subjects.items():
                    code = s_fields.get("code", "").strip().upper()
                    s_mods = sorted(modules_by_subj.get(s_id, []), key=lambda m: m["module_number"])
                    if code and s_mods:
                        _VERIFIED_SYLLABUS_CACHE[code] = [
                            {
                                "module_number": m["module_number"],
                                "title": m["title"],
                                "raw_text": m.get("description", ""),
                                "chapters": _extract_chapters_from_text(m.get("description", "")),
                                "topics": _extract_topics_from_text(m.get("description", ""))
                            }
                            for m in s_mods
                        ]
            except Exception as e:
                logger.warning("Could not load verified syllabus cache: %s", e)

    clean_code = (course_code or "").strip().upper()
    if clean_code and clean_code in _VERIFIED_SYLLABUS_CACHE:
        return _VERIFIED_SYLLABUS_CACHE[clean_code]

    return None


# =====================================================================
# Strict Syllabus Validation Against Authoritative Source
# =====================================================================

def validate_curriculum_against_source(data: dict, prescribed_modules: list | None = None) -> dict:
    """
    Validates generated curriculum JSON.
    If prescribed_modules are provided:
      1. Exactly 5 modules are required.
      2. Preserves all prescribed module titles.
      3. Verifies that prescribed topics are not silently removed.
      4. Validates description and keyword completeness.
    Raises CurriculumValidationError if output fails validation.
    """
    modules = data.get("modules")
    if not isinstance(modules, list):
        raise CurriculumValidationError("Invalid curriculum structure: 'modules' key must be a list.")

    if len(modules) != 5:
        raise CurriculumValidationError(
            f"Invalid curriculum structure: expected exactly 5 modules, got {len(modules)}."
        )

    for idx, mod in enumerate(modules, start=1):
        if not isinstance(mod, dict):
            raise CurriculumValidationError(f"Module {idx} must be a dictionary object.")

        title = (mod.get("title") or "").strip()
        if not title:
            raise CurriculumValidationError(f"Module {idx} is missing a valid title.")

        topics = mod.get("topics")
        if not isinstance(topics, list) or len(topics) == 0:
            raise CurriculumValidationError(f"Module {idx} ('{title}') contains no topics.")

        for t_idx, topic in enumerate(topics, start=1):
            if not isinstance(topic, dict):
                raise CurriculumValidationError(f"Module {idx}, Topic {t_idx} must be a dictionary.")
            if not (topic.get("title") or "").strip():
                raise CurriculumValidationError(f"Module {idx}, Topic {t_idx} is missing a title.")
            if not (topic.get("description") or "").strip():
                raise CurriculumValidationError(f"Module {idx}, Topic {t_idx} is missing a description.")
            keywords = topic.get("keywords")
            if not isinstance(keywords, list) or len(keywords) == 0:
                raise CurriculumValidationError(f"Module {idx}, Topic {t_idx} contains no search keywords.")

    # Validate against authoritative prescribed source modules if present
    if prescribed_modules and len(prescribed_modules) == 5:
        for i, pres in enumerate(prescribed_modules):
            gen_mod = modules[i]
            pres_title = pres["title"].strip()
            gen_title = (gen_mod.get("title") or "").strip()

            # Check title preservation
            norm_pres = re.sub(r"[^a-z0-9]", "", pres_title.lower())
            norm_gen = re.sub(r"[^a-z0-9]", "", gen_title.lower())

            # Check overlap between prescribed title and generated title
            pres_words = set(re.findall(r"\w+", pres_title.lower()))
            gen_words = set(re.findall(r"\w+", gen_title.lower()))
            common_words = pres_words.intersection(gen_words)

            is_match = (
                norm_pres in norm_gen
                or norm_gen in norm_pres
                or (len(pres_words) > 0 and len(common_words) / len(pres_words) >= 0.35)
            )

            if not is_match:
                raise CurriculumValidationError(
                    f"Module {i+1} title mismatch with prescribed syllabus: "
                    f"expected '{pres_title}', but received '{gen_title}'."
                )

            # Preserve the authoritative original title verbatim on the record
            gen_mod["title"] = pres_title

            # Check prescribed topics preservation
            pres_topics = pres.get("topics", [])
            if pres_topics:
                gen_topic_text = " ".join(
                    (t.get("title", "") + " " + t.get("description", ""))
                    for t in gen_mod.get("topics", [])
                ).lower()

                # Check if at least 1 core topic identifier is represented
                represented = 0
                for pt in pres_topics:
                    pt_words = [w for w in re.findall(r"\w+", pt.lower()) if len(w) > 3]
                    if any(w in gen_topic_text for w in pt_words):
                        represented += 1

                if len(pres_topics) >= 2 and represented == 0:
                    raise CurriculumValidationError(
                        f"Module {i+1} ('{pres_title}') omitted prescribed syllabus topics: {pres_topics[:3]}."
                    )

    return data


# =====================================================================
# Request Execution with Retry & Fallback
# =====================================================================

def _send_model_request_with_retries(
    model_name: str,
    payload: dict,
    api_key: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    request_timeout: int = 30,
    is_fallback: bool = False
):
    """
    Sends request to a specific Gemini model with bounded exponential backoff for transient errors.
    Returns (response, None) on HTTP 200.
    Returns (None, last_error_message) if max retries exhausted for transient errors.
    Raises non-retryable exceptions immediately (GeminiAuthError, GeminiInvalidRequestError, GeminiModelNotFoundError).
    """
    clean_model = model_name.replace("models/", "")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    role_label = "fallback" if is_fallback else "primary"
    last_transient_error = ""

    for attempt in range(max_retries + 1):
        attempt_num = attempt + 1
        total_attempts = max_retries + 1
        logger.info(
            "Dispatching Gemini generation request to %s model '%s' (attempt %d/%d)",
            role_label, clean_model, attempt_num, total_attempts
        )

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=request_timeout)

            # Success
            if res.status_code == 200:
                logger.info(
                    "Received HTTP 200 OK from %s model '%s' on attempt %d/%d",
                    role_label, clean_model, attempt_num, total_attempts
                )
                return res, None

            err_msg = sanitize_error_message(res)

            # Non-retryable errors: 400, 401, 403, 404
            if res.status_code in (401, 403):
                logger.error(
                    "Non-retryable authentication error HTTP %d from %s model '%s': %s",
                    res.status_code, role_label, clean_model, err_msg
                )
                raise GeminiAuthError(
                    f"Gemini API authentication failed ({res.status_code}): {err_msg}"
                )

            if res.status_code == 400:
                logger.error(
                    "Non-retryable malformed request HTTP %d from %s model '%s': %s",
                    res.status_code, role_label, clean_model, err_msg
                )
                raise GeminiInvalidRequestError(
                    f"Gemini API rejected request as invalid (Status 400): {err_msg}"
                )

            if res.status_code == 404:
                logger.error(
                    "Non-retryable invalid model HTTP %d from %s model '%s': %s",
                    res.status_code, role_label, clean_model, err_msg
                )
                raise GeminiModelNotFoundError(
                    f"Gemini model '{clean_model}' not found (Status 404): {err_msg}"
                )

            # Transient errors: 503 (Unavailable), 500 (Internal), 429 (Rate Limit / Too Many Requests), 502, 504
            if res.status_code in (429, 500, 502, 503, 504):
                retry_after_hdr = res.headers.get("Retry-After")
                last_transient_error = f"HTTP {res.status_code}: {err_msg}"

                if attempt < max_retries:
                    delay = calculate_backoff_delay(
                        attempt=attempt_num,
                        retry_after_header=retry_after_hdr,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    logger.warning(
                        "Transient HTTP %d from %s model '%s' (attempt %d/%d). "
                        "Retrying in %.2fs (Retry-After: %s). Message: %s",
                        res.status_code, role_label, clean_model, attempt_num, total_attempts,
                        delay, retry_after_hdr or "None", err_msg
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        "%s model '%s' exhausted all %d retries with transient error: %s",
                        role_label.capitalize(), clean_model, max_retries, last_transient_error
                    )
                    return None, last_transient_error

            # Unexpected HTTP status
            logger.error(
                "Unexpected HTTP %d from %s model '%s': %s",
                res.status_code, role_label, clean_model, err_msg
            )
            raise CurriculumGenerationError(
                f"Gemini API returned unexpected status {res.status_code}: {err_msg}"
            )

        except (GeminiAuthError, GeminiInvalidRequestError, GeminiModelNotFoundError, CurriculumGenerationError):
            raise
        except requests.exceptions.Timeout as timeout_err:
            last_transient_error = f"Network Timeout: {sanitize_error_message(timeout_err)}"
            logger.warning(
                "Timeout (%ds) calling %s model '%s' (attempt %d/%d): %s",
                request_timeout, role_label, clean_model, attempt_num, total_attempts, last_transient_error
            )
            # For network timeouts (e.g. ReadTimeout), allow at most 1 retry per model
            # so a stalled model does not exhaust worker resources or loop indefinitely.
            if attempt < min(max_retries, 1):
                delay = calculate_backoff_delay(
                    attempt=attempt_num,
                    retry_after_header=None,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                logger.info("Retrying %s model '%s' after timeout in %.2fs...", role_label, clean_model, delay)
                time.sleep(delay)
                continue
            else:
                logger.warning(
                    "%s model '%s' exhausted timeout budget (%d timeout attempts). Switching or failing.",
                    role_label.capitalize(), clean_model, attempt_num
                )
                return None, last_transient_error
        except requests.RequestException as req_err:
            last_transient_error = f"Network {type(req_err).__name__}: {sanitize_error_message(req_err)}"
            if attempt < max_retries:
                delay = calculate_backoff_delay(
                    attempt=attempt_num,
                    retry_after_header=None,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                logger.warning(
                    "Network error calling %s model '%s' (attempt %d/%d): %s. Retrying in %.2fs...",
                    role_label, clean_model, attempt_num, total_attempts, last_transient_error, delay
                )
                time.sleep(delay)
                continue
            else:
                logger.warning(
                    "%s model '%s' exhausted all %d retries with network exceptions. Last: %s",
                    role_label.capitalize(), clean_model, max_retries, last_transient_error
                )
                return None, last_transient_error

    return None, last_transient_error


def execute_gemini_generation(
    payload: dict,
    api_key: str,
    primary_model: str,
    fallback_model: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    request_timeout: int = 30
):
    """
    Executes curriculum generation with primary model retries, switching to fallback model
    if primary exhausts retries or returns model not found (404).
    """
    last_err = ""
    try:
        res, last_err = _send_model_request_with_retries(
            model_name=primary_model,
            payload=payload,
            api_key=api_key,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            request_timeout=request_timeout,
            is_fallback=False
        )
        if res is not None:
            return res, primary_model
    except GeminiModelNotFoundError as mnf_err:
        logger.warning("Primary Gemini model '%s' not found (%s). Attempting fallback model...", primary_model, mnf_err)
        last_err = str(mnf_err)

    if fallback_model and fallback_model != primary_model:
        logger.warning(
            "Primary model '%s' unavailable (%s). Switching to fallback model '%s'...",
            primary_model, last_err, fallback_model
        )
        try:
            res, fallback_err = _send_model_request_with_retries(
                model_name=fallback_model,
                payload=payload,
                api_key=api_key,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                request_timeout=request_timeout,
                is_fallback=True
            )
            if res is not None:
                logger.info("Fallback model '%s' succeeded where primary failed.", fallback_model)
                return res, fallback_model
            last_err = f"Primary ({primary_model}): {last_err}; Fallback ({fallback_model}): {fallback_err}"
        except GeminiModelNotFoundError as f_err:
            last_err = f"Primary ({primary_model}): {last_err}; Fallback ({fallback_model}): {f_err}"

    # Try universal safe Google Gemini models if both primary and fallback failed with 404
    for safe_model in ("gemini-1.5-flash", "gemini-2.0-flash"):
        if safe_model not in (primary_model, fallback_model):
            try:
                logger.info("Attempting universal safe Gemini model '%s'...", safe_model)
                res, _ = _send_model_request_with_retries(
                    model_name=safe_model,
                    payload=payload,
                    api_key=api_key,
                    max_retries=1,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    request_timeout=request_timeout,
                    is_fallback=True
                )
                if res is not None:
                    return res, safe_model
            except Exception:
                pass

    logger.error("Curriculum generation failed: All eligible models exhausted. %s", last_err)
    if "not found" in last_err.lower():
        raise GeminiModelNotFoundError(
            f"The configured AI model is not available: {last_err}. Please verify your Gemini model configuration."
        )
    raise GeminiTransientError(
        f"The AI curriculum service is temporarily unavailable due to high demand. "
        f"Retries across models were exhausted ({last_err}). Please try again in a few moments."
    )


# =====================================================================
# Groq Request Execution with Retry & Fallback
# =====================================================================

def _send_groq_request_with_retries(
    model_name: str,
    system_instructions: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    request_timeout: int = 30,
    is_fallback: bool = False
):
    """
    Sends chat completion request to Groq API with bounded exponential backoff for transient errors.
    Returns (response, None) on HTTP 200.
    Returns (None, last_error_message) if max retries exhausted for transient errors.
    Raises non-retryable exceptions immediately:
      - AIAuthError on 401
      - AIPermissionError on 403
      - AIModelNotFoundError on 404 or decommissioned model 400
      - AIInvalidRequestError on 400
    """
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 8192
    }

    role_label = "fallback" if is_fallback else "primary"
    last_transient_error = ""

    for attempt in range(max_retries + 1):
        attempt_num = attempt + 1
        total_attempts = max_retries + 1
        logger.info(
            "Dispatching Groq generation request to %s model '%s' (attempt %d/%d)",
            role_label, model_name, attempt_num, total_attempts
        )

        try:
            res = requests.post(url, json=payload, headers=headers, timeout=request_timeout)

            # Success
            if res.status_code == 200:
                logger.info(
                    "Received HTTP 200 OK from Groq %s model '%s' on attempt %d/%d",
                    role_label, model_name, attempt_num, total_attempts
                )
                return res, None

            err_msg = sanitize_error_message(res)

            # Authentication error: 401
            if res.status_code == 401:
                logger.error(
                    "Non-retryable authentication error HTTP 401 from Groq %s model '%s': %s",
                    role_label, model_name, err_msg
                )
                raise AIAuthError(
                    f"Groq API authentication failed (Status 401): {err_msg}"
                )

            # Permission error: 403
            if res.status_code == 403:
                logger.error(
                    "Non-retryable permission error HTTP 403 from Groq %s model '%s': %s",
                    role_label, model_name, err_msg
                )
                raise AIPermissionError(
                    f"Groq API access forbidden (Status 403) for model '{model_name}': {err_msg}"
                )

            # Model not found: 404
            if res.status_code == 404:
                logger.error(
                    "Non-retryable invalid model HTTP 404 from Groq %s model '%s': %s",
                    role_label, model_name, err_msg
                )
                raise AIModelNotFoundError(
                    f"Groq model '{model_name}' not found (Status 404): {err_msg}"
                )

            # Bad Request: 400 (Check if decommissioned or invalid model)
            if res.status_code == 400:
                logger.error(
                    "Non-retryable HTTP 400 from Groq %s model '%s': %s",
                    role_label, model_name, err_msg
                )
                if any(w in err_msg.lower() for w in ("model", "decommissioned", "not found", "does not exist", "unsupported")):
                    raise AIModelNotFoundError(
                        f"Groq model '{model_name}' is unavailable or decommissioned (Status 400): {err_msg}"
                    )
                raise AIInvalidRequestError(
                    f"Groq API rejected request as invalid (Status 400): {err_msg}"
                )

            # Rate limit error: 429
            if res.status_code == 429:
                retry_after_hdr = res.headers.get("Retry-After")
                last_transient_error = f"Rate limit reached (HTTP 429): {err_msg}"

                if attempt < max_retries:
                    delay = calculate_backoff_delay(
                        attempt=attempt_num,
                        retry_after_header=retry_after_hdr,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    logger.warning(
                        "Rate limit HTTP 429 from Groq %s model '%s' (attempt %d/%d). "
                        "Retrying in %.2fs (Retry-After: %s). Message: %s",
                        role_label, model_name, attempt_num, total_attempts,
                        delay, retry_after_hdr or "None", err_msg
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        "Groq %s model '%s' exhausted all %d retries on rate limit (HTTP 429): %s",
                        role_label, model_name, max_retries, last_transient_error
                    )
                    return None, last_transient_error

            # Server errors: 500, 502, 503, 504
            if res.status_code in (500, 502, 503, 504):
                retry_after_hdr = res.headers.get("Retry-After")
                last_transient_error = f"Server error (HTTP {res.status_code}): {err_msg}"

                if attempt < max_retries:
                    delay = calculate_backoff_delay(
                        attempt=attempt_num,
                        retry_after_header=retry_after_hdr,
                        base_delay=base_delay,
                        max_delay=max_delay
                    )
                    logger.warning(
                        "Server error HTTP %d from Groq %s model '%s' (attempt %d/%d). "
                        "Retrying in %.2fs. Message: %s",
                        res.status_code, role_label, model_name, attempt_num, total_attempts,
                        delay, err_msg
                    )
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(
                        "Groq %s model '%s' exhausted all %d retries with server error: %s",
                        role_label, model_name, max_retries, last_transient_error
                    )
                    return None, last_transient_error

            logger.error(
                "Unexpected HTTP %d from Groq %s model '%s': %s",
                res.status_code, role_label, model_name, err_msg
            )
            raise CurriculumGenerationError(
                f"Groq API returned unexpected status {res.status_code}: {err_msg}"
            )

        except (AIAuthError, AIPermissionError, AIInvalidRequestError, AIModelNotFoundError, CurriculumGenerationError):
            raise
        except requests.exceptions.Timeout as timeout_err:
            last_transient_error = f"Network Timeout ({request_timeout}s): {sanitize_error_message(timeout_err)}"
            logger.warning(
                "Timeout calling Groq %s model '%s' (attempt %d/%d): %s",
                role_label, model_name, attempt_num, total_attempts, last_transient_error
            )
            if attempt < min(max_retries, 1):
                delay = calculate_backoff_delay(
                    attempt=attempt_num,
                    retry_after_header=None,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                time.sleep(delay)
                continue
            else:
                return None, last_transient_error
        except requests.exceptions.ConnectionError as conn_err:
            last_transient_error = f"Network ConnectionError: {sanitize_error_message(conn_err)}"
            logger.warning(
                "Connection error calling Groq %s model '%s' (attempt %d/%d): %s",
                role_label, model_name, attempt_num, total_attempts, last_transient_error
            )
            if attempt < min(max_retries, 1):
                delay = calculate_backoff_delay(
                    attempt=attempt_num,
                    retry_after_header=None,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                time.sleep(delay)
                continue
            else:
                return None, last_transient_error
        except requests.RequestException as req_err:
            last_transient_error = f"Network {type(req_err).__name__}: {sanitize_error_message(req_err)}"
            if attempt < max_retries:
                delay = calculate_backoff_delay(
                    attempt=attempt_num,
                    retry_after_header=None,
                    base_delay=base_delay,
                    max_delay=max_delay
                )
                time.sleep(delay)
                continue
            else:
                return None, last_transient_error

    return None, last_transient_error


def execute_groq_generation(
    system_instructions: str,
    user_prompt: str,
    api_key: str,
    primary_model: str = "openai/gpt-oss-120b",
    fallback_model: str = "openai/gpt-oss-20b",
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    request_timeout: int = 30
):
    """
    Executes curriculum generation via Groq API with primary model retries,
    switching to fallback model if primary exhausts retries or returns model not found (404/400).
    """
    last_err = ""
    try:
        res, last_err = _send_groq_request_with_retries(
            model_name=primary_model,
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            api_key=api_key,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            request_timeout=request_timeout,
            is_fallback=False
        )
        if res is not None:
            return res, primary_model
    except AIModelNotFoundError as mnf_err:
        logger.warning("Primary Groq model '%s' not found (%s). Attempting fallback model...", primary_model, mnf_err)
        last_err = str(mnf_err)

    if fallback_model and fallback_model != primary_model:
        logger.warning(
            "Groq primary model '%s' unavailable (%s). Switching to fallback model '%s'...",
            primary_model, last_err, fallback_model
        )
        try:
            res, fallback_err = _send_groq_request_with_retries(
                model_name=fallback_model,
                system_instructions=system_instructions,
                user_prompt=user_prompt,
                api_key=api_key,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                request_timeout=request_timeout,
                is_fallback=True
            )
            if res is not None:
                logger.info("Groq fallback model '%s' succeeded where primary failed.", fallback_model)
                return res, fallback_model
            last_err = f"Primary ({primary_model}): {last_err}; Fallback ({fallback_model}): {fallback_err}"
        except AIModelNotFoundError as f_err:
            last_err = f"Primary ({primary_model}): {last_err}; Fallback ({fallback_model}): {f_err}"

    # Try universal safe production Groq models if both primary and fallback failed
    for safe_model in ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"):
        if safe_model not in (primary_model, fallback_model):
            try:
                logger.info("Attempting safe Groq production model '%s'...", safe_model)
                res, _ = _send_groq_request_with_retries(
                    model_name=safe_model,
                    system_instructions=system_instructions,
                    user_prompt=user_prompt,
                    api_key=api_key,
                    max_retries=1,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    request_timeout=request_timeout,
                    is_fallback=True
                )
                if res is not None:
                    return res, safe_model
            except Exception:
                pass

    logger.error("Curriculum generation failed: All eligible retries exhausted on Groq. %s", last_err)
    if any(w in last_err.lower() for w in ("not found", "decommissioned", "does not exist", "unsupported", "404")):
        raise AIModelNotFoundError(
            f"The configured Groq AI model is not available: {last_err}. Please verify your model settings (default: openai/gpt-oss-120b)."
        )
    if any(w in last_err.lower() for w in ("rate limit", "429")):
        raise AIRateLimitError(
            f"Groq API rate limit reached: {last_err}. Please wait a moment before trying again."
        )
    if any(w in last_err.lower() for w in ("timeout", "connectionerror", "network")):
        raise AINetworkError(
            f"Network error communicating with Groq API: {last_err}. Please check network connection and try again."
        )
    if any(c in last_err for c in ("500", "502", "503", "504")):
        raise AIServerError(
            f"Groq API server error ({last_err}). Please try again in a few moments."
        )
    raise AITransientError(
        f"The AI curriculum service is temporarily unavailable due to high demand. "
        f"Retries across models were exhausted ({last_err}). Please try again in a few moments."
    )


# =====================================================================
# JSON Response Parsing & Extraction Logic (Preserved)
# =====================================================================

def _find_modules_dict(obj, depth=0):
    """
    Recursively searches a parsed JSON object for a dict containing a 'modules' list.
    Handles cases where Gemini wraps the curriculum in an outer key like {"curriculum": {...}}.
    """
    if depth > 4:
        return None
    if isinstance(obj, dict):
        if "modules" in obj and isinstance(obj["modules"], list) and len(obj["modules"]) > 0:
            return obj
        for v in obj.values():
            result = _find_modules_dict(v, depth + 1)
            if result:
                return result
    elif isinstance(obj, list):
        for item in obj:
            result = _find_modules_dict(item, depth + 1)
            if result:
                return result
    return None


def extract_curriculum_data_from_json(raw_text):
    """
    Robustly extracts and decodes the curriculum dictionary from Gemini output.
    Handles markdown code fences, thought preambles, and nested wrapper objects.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    cleaned = cleaned.strip()

    # 1. Try direct json.loads
    try:
        data = json.loads(cleaned)
        result = _find_modules_dict(data)
        if result:
            return result
    except Exception:
        pass

    # 2. Iteratively decode objects using raw_decode
    decoder = json.JSONDecoder()
    idx = 0
    candidate_dicts = []
    while idx < len(cleaned):
        brace_pos = cleaned.find("{", idx)
        if brace_pos == -1:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, brace_pos)
            if isinstance(obj, dict):
                candidate_dicts.append(obj)
                result = _find_modules_dict(obj)
                if result:
                    return result
            idx = max(end, brace_pos + 1)
        except json.JSONDecodeError:
            idx = brace_pos + 1

    # 3. Fallback: clean trailing commas and retry
    cleaned_no_commas = re.sub(r",\s*([\]}])", r"\1", cleaned)
    try:
        data = json.loads(cleaned_no_commas)
        result = _find_modules_dict(data)
        if result:
            return result
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    if candidate_dicts:
        return candidate_dicts[-1]

    raise CurriculumParseError("Could not find a valid curriculum JSON object with 'modules' in the AI response.")


# =====================================================================
# Main Service Entrypoint
# =====================================================================

def generate_curriculum_data(subject_name, subject_code="", syllabus_text=""):
    """
    Calls the configured AI API (Groq or Gemini) to generate and validate structured curriculum data
    without touching the database.
    - When official syllabus text is supplied: extracts and preserves all 5 module titles
      and prescribed topics, validating the AI output against the source.
    - When syllabus text is omitted: checks verified official syllabus records for known
      course codes (e.g. BAD703, BAI701), or marks the generated subject as an unverified
      AI reference curriculum.
    Returns curriculum dictionary payload ready for atomic persistence.
    """
    provider, api_key = resolve_ai_provider()

    clean_name = subject_name.strip()
    clean_code = (subject_code.strip() or "COURSE").upper()
    has_custom_syllabus = bool(syllabus_text and syllabus_text.strip())

    prescribed_modules = None
    is_verified_source = False

    if has_custom_syllabus:
        prescribed_modules = parse_prescribed_syllabus(syllabus_text.strip())
        if len(prescribed_modules) == 5:
            is_verified_source = True
            logger.info("Parsed %d authoritative source modules from user syllabus text for '%s'.", len(prescribed_modules), clean_name)
    else:
        # Check verified curriculum repository for known official course codes
        verified_source = get_verified_syllabus_source(clean_code, clean_name)
        if verified_source and len(verified_source) == 5:
            prescribed_modules = verified_source
            is_verified_source = True
            logger.info("Loaded %d verified official syllabus modules from verified source for '%s' (%s).", len(prescribed_modules), clean_name, clean_code)

    # Construct source-grounded prompt
    system_instructions = (
        "You are an expert university curriculum architect specializing in engineering and technical university curricula "
        "(e.g., VTU, AICTE, standard Computer Science and Engineering syllabi).\n"
        "Your task is to produce a complete, highly structured, 5-module curriculum in strict JSON format.\n\n"
        "Requirements:\n"
        "1. Structure must have exactly 5 Modules (Module 1 through Module 5).\n"
        "2. When a module syllabus is divided into chapters (e.g. Chapter 1, Chapter 2 / CHP-1, CHP-2), preserve the chapter structure:\n"
        "   Each topic must include 'chapter_number' (integer, e.g. 1, 2) and 'chapter_title' (string, e.g. 'Key Management Fundamentals').\n"
        "3. Each Topic must include:\n"
        "   - 'chapter_number': 1 or 2 (chapter sequence number within the module)\n"
        "   - 'chapter_title': concise chapter title (e.g. 'Key Management Fundamentals', 'Public-Key Management')\n"
        "   - 'title': concise topic title (under 80 characters)\n"
        "   - 'description': 1 sentence explaining key conceptual objective\n"
        "   - 'is_high_yield': boolean (CRITICAL EXAM VALUE: set to true ONLY for the top 2-3 most important exam concepts per module that carry 8-10 marks in university exams, such as core algorithms, system architectures, mathematical models, and working principles; set to false for introductory, survey, history, or overview topics)\n"
        "   - 'vtu_frequency_tag': concise exam frequency tag for high-yield topics (e.g. 'Core VTU Concept (Q1/Q2 Choice)', 'Frequent 8-10M Question', or empty string if is_high_yield is false)\n"
        "   - 'keywords': exactly 3 specific, context-anchored search queries optimized for finding accurate engineering lectures and tutorials:\n"
        "       1) '<Topic Title> <Subject Name> lecture tutorial'\n"
        "       2) '<Topic Title> <Subject Name> geeksforgeeks documentation'\n"
        "       3) '<Topic Title> in <Subject Name> engineering notes review'\n"
        "4. Output MUST strictly be valid JSON matching this schema:\n"
        "{\n"
        '  "code": "string",\n'
        '  "name": "string",\n'
        '  "description": "string",\n'
        '  "modules": [\n'
        "    {\n"
        '      "module_number": 1,\n'
        '      "title": "string",\n'
        '      "description": "string",\n'
        '      "topics": [\n'
        "        {\n"
        '          "chapter_number": 1,\n'
        '          "chapter_title": "string",\n'
        '          "title": "string",\n'
        '          "description": "string",\n'
        '          "is_high_yield": true,\n'
        '          "vtu_frequency_tag": "string",\n'
        '          "keywords": ["keyword 1", "keyword 2", "keyword 3"]\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    if prescribed_modules and len(prescribed_modules) == 5:
        module_spec_lines = []
        for m in prescribed_modules:
            spec = f"Module {m['module_number']}: {m['title']}"
            if m.get("chapters"):
                for chp in m["chapters"]:
                    c_title = chp.get("chapter_title") or f"Chapter {chp.get('chapter_number', 1)}"
                    chp_topics = ", ".join(chp.get("topics", []))
                    spec += f"\n  Chapter {chp.get('chapter_number', 1)} ({c_title}): {chp_topics}"
            else:
                m_topics = ", ".join(m["topics"]) if m["topics"] else m.get("raw_text", "")
                if m_topics:
                    spec += f"\n  Prescribed Topics/Content: {m_topics}"
            module_spec_lines.append(spec)

        prescribed_spec_str = "\n\n".join(module_spec_lines)

        user_prompt = (
            f"Subject: {clean_name}\n"
            f"Course Code: {clean_code}\n\n"
            f"AUTHORITATIVE OFFICIAL SYLLABUS (You MUST strictly follow this exact structure):\n"
            f"\"\"\"\n{prescribed_spec_str}\n\"\"\"\n\n"
            f"CRITICAL GROUNDING RULES:\n"
            f"1. You MUST generate exactly 5 modules.\n"
            f"2. Each module's 'title' MUST EXACTLY match the prescribed title above.\n"
            f"3. Preserve all prescribed topics and their conceptual meaning without silently omitting any.\n"
            f"4. For each topic, generate a concise title, 1-sentence objective description, and exactly 3 high-intent search keywords.\n"
            f"5. EXAM IMPORTANCE (80/20 HIGH-YIELD RULE):\n"
            f"   - Evaluate each topic from a university exam perspective (e.g. VTU Q1 vs Q2 choice questions).\n"
            f"   - For every module, pick the top 2-3 most important, high-scoring exam concepts (core algorithms, architectures, derivations, working principles, mathematical models) and set 'is_high_yield': true with a descriptive 'vtu_frequency_tag' (e.g. 'Core 10M Question for Q1/Q2', 'Key Architecture for SEE', 'Frequent Derivation/Algorithm').\n"
            f"   - Set 'is_high_yield': false for introductory, survey, motivation, historical, or broad overview topics (e.g. 'Introduction to...', 'Overview of...', 'Need for...'). Do NOT blindly mark topics 1 and 2 as high-yield just because they are listed first!\n"
            f"6. Do NOT invent alternative modules, and do NOT replace prescribed titles."
        )
    elif has_custom_syllabus:
        user_prompt = (
            f"Subject: {clean_name}\n"
            f"Course Code: {clean_code}\n\n"
            f"Extract and organize the following syllabus into exactly 5 modules with topics and search keywords:\n"
            f"\"\"\"\n{syllabus_text.strip()}\n\"\"\"\n\n"
            f"EXAM IMPORTANCE RULES (HIGH-YIELD 80/20 ANALYSIS):\n"
            f"- Evaluate topics from a university exam perspective (e.g. VTU Q1 vs Q2 choices).\n"
            f"- In every module, identify the top 2-3 core exam concepts (algorithms, architectures, derivations, working principles) and set 'is_high_yield': true with a descriptive 'vtu_frequency_tag'.\n"
            f"- Set 'is_high_yield': false for introductory, survey, motivation, or high-level overview topics (e.g. 'Introduction to...', 'Overview of...'). Never mark topics 1 and 2 as high-yield merely because they are first in order."
        )
    else:
        user_prompt = (
            f"Generate a comprehensive reference 5-module university curriculum for the subject '{clean_name}' "
            f"(Course Code: '{clean_code}'). Provide detailed topics and realistic educational search keywords.\n\n"
            f"EXAM IMPORTANCE RULES (HIGH-YIELD 80/20 ANALYSIS):\n"
            f"- In every module, identify the top 2-3 core exam concepts (algorithms, architectures, derivations, working principles) and set 'is_high_yield': true with a descriptive 'vtu_frequency_tag' (e.g. 'Core 10M Question for Q1/Q2', 'Frequently Tested Algorithm').\n"
            f"- Set 'is_high_yield': false for introductory, survey, or high-level overview topics (e.g. 'Introduction to...', 'Overview of...'). Never mark topics 1 and 2 as high-yield merely because they are first in order."
        )

    if provider == 'groq':
        groq_primary, groq_fallback = resolve_groq_models(api_key)
        max_retries = config('GROQ_MAX_RETRIES', default=getattr(settings, 'GROQ_MAX_RETRIES', 3), cast=int)
        base_delay = config('GROQ_BACKOFF_BASE', default=1.0, cast=float)
        max_delay = config('GROQ_BACKOFF_MAX', default=8.0, cast=float)
        request_timeout = config('GROQ_REQUEST_TIMEOUT', default=getattr(settings, 'GROQ_REQUEST_TIMEOUT', 30), cast=int)

        response, active_model = execute_groq_generation(
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            api_key=api_key,
            primary_model=groq_primary,
            fallback_model=groq_fallback,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            request_timeout=request_timeout
        )

        try:
            res_json = response.json()
            choices = res_json.get("choices", [])
            if not choices:
                raise CurriculumParseError("Groq API returned no response choices.")

            message = choices[0].get("message", {})
            full_text = message.get("content", "")
            if not full_text.strip():
                raise CurriculumParseError("Groq returned empty text in response message.")

            data = extract_curriculum_data_from_json(full_text)

        except (KeyError, IndexError, json.JSONDecodeError, CurriculumParseError) as e:
            logger.error("Failed to parse Groq response: %s", e)
            raise CurriculumParseError(f"Failed to parse AI generated curriculum: {e}")

    else:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{system_instructions}\n\nUser Request:\n{user_prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 8192
            }
        }

        max_retries = config('GEMINI_MAX_RETRIES', default=3, cast=int)
        base_delay = config('GEMINI_BACKOFF_BASE', default=1.0, cast=float)
        max_delay = config('GEMINI_BACKOFF_MAX', default=8.0, cast=float)
        request_timeout = config('GEMINI_REQUEST_TIMEOUT', default=30, cast=int)

        primary_model, fallback_model = resolve_configured_models(api_key)

        response, active_model = execute_gemini_generation(
            payload=payload,
            api_key=api_key,
            primary_model=primary_model,
            fallback_model=fallback_model,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            request_timeout=request_timeout
        )

        try:
            res_json = response.json()
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise CurriculumParseError("Gemini API returned no response candidates.")

            candidate = candidates[0]
            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise CurriculumParseError("Gemini candidate contains no content parts.")

            full_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
            if not full_text.strip():
                raise CurriculumParseError("Gemini returned empty text in candidate parts.")

            data = extract_curriculum_data_from_json(full_text)

        except (KeyError, IndexError, json.JSONDecodeError, CurriculumParseError) as e:
            logger.error("Failed to parse Gemini response: %s", e)
            raise CurriculumParseError(f"Failed to parse AI generated curriculum: {e}")

    # Validate against authoritative prescribed source modules before persistence
    data = validate_curriculum_against_source(data, prescribed_modules=prescribed_modules)

    return {
        "data": data,
        "clean_name": clean_name,
        "clean_code": clean_code,
        "is_verified_source": is_verified_source,
        "active_model": active_model,
    }


def persist_curriculum_records(
    curriculum_payload: dict,
    job_id=None,
    claim_token=None,
    lease_duration=300,
    branch=None,
    semester=None
) -> Subject:
    """
    Persists Subject, Module, Topic, and TopicKeyword records atomically in PostgreSQL.
    When job_id is provided, acquires a row lock on GenerationJob using select_for_update(),
    strictly verifying:
    1. The job exists in the database.
    2. Job status is RUNNING.
    3. Claim token matches claim_token exactly (fencing protection).
    4. The worker lease has not expired.
    If any check fails, raises CurriculumFencingError, rolling back all records without modifying database state.
    Only after successful fencing validation are curriculum records persisted,
    and GenerationJob is marked COMPLETED in the same atomic transaction.
    """
    with transaction.atomic():
        job = None
        if job_id is not None:
            job = GenerationJob.objects.select_for_update().filter(id=job_id).first()
            if not job:
                raise CurriculumFencingError(f"GenerationJob {job_id} not found during persistence.")
            if job.status != GenerationJob.Status.RUNNING:
                raise CurriculumFencingError(
                    f"GenerationJob {job_id} status is '{job.status}', expected RUNNING."
                )
            if claim_token is not None and str(job.claim_token) != str(claim_token):
                raise CurriculumFencingError(
                    f"GenerationJob {job_id} claim token '{job.claim_token}' revoked or mismatch (expected '{claim_token}')."
                )
            effective_lease = lease_duration or 300
            if job.is_lease_expired(lease_duration_seconds=effective_lease):
                raise CurriculumFencingError(
                    f"GenerationJob {job_id} lease expired at {job.locked_at} (duration: {effective_lease}s)."
                )

        data = curriculum_payload["data"]
        clean_name = curriculum_payload["clean_name"]
        clean_code = curriculum_payload["clean_code"]
        is_verified_source = curriculum_payload["is_verified_source"]
        active_model = curriculum_payload.get("active_model", "gemini")

        final_name = (data.get("name") or clean_name).strip()
        final_code = (clean_code if clean_code and clean_code != "COURSE" else (data.get("code") or clean_code)).strip().upper()
        raw_desc = data.get("description") or f"Curriculum and learning resources for {final_name}."

        # Annotate whether curriculum is verified or unverified reference
        if is_verified_source:
            final_desc = raw_desc
        else:
            unverified_tag = "[AI-Generated & Unverified Curriculum - Official university syllabus outline was not supplied. Please verify against your university scheme.]"
            if unverified_tag not in raw_desc:
                final_desc = f"{unverified_tag}\n\n{raw_desc}"
            else:
                final_desc = raw_desc

        if job:
            if not branch and job.branch:
                branch = job.branch
            if not semester and job.semester:
                semester = job.semester

        existing_subject = None
        if final_code and final_code != "COURSE":
            existing_subject = Subject.objects.filter(code__iexact=final_code, name__iexact=final_name).first()
        if not existing_subject:
            existing_subject = Subject.objects.filter(name__iexact=final_name).first()

        if existing_subject and existing_subject.modules.exists():
            logger.info(
                "Subject '%s' (%s) already exists with %d modules. Returning existing subject to prevent duplicates.",
                existing_subject.name, existing_subject.code, existing_subject.modules.count()
            )
            subject = existing_subject
            if branch and subject.branch != branch:
                subject.branch = branch
            if semester and subject.semester != semester:
                subject.semester = semester
            subject.save()
        else:
            if existing_subject:
                subject = existing_subject
                subject.name = final_name[:150]
                subject.code = final_code[:20]
                subject.description = final_desc
                if branch:
                    subject.branch = branch
                if semester:
                    subject.semester = semester
                subject.save()
            else:
                base_slug = slugify(final_name) or "new-subject"
                unique_slug = base_slug
                slug_counter = 1
                while Subject.objects.filter(slug=unique_slug).exists():
                    unique_slug = f"{base_slug}-{slug_counter}"
                    slug_counter += 1

                next_order = (Subject.objects.order_by('-order').values_list('order', flat=True).first() or 0) + 1

                subject = Subject(
                    name=final_name[:150],
                    code=final_code[:20],
                    slug=unique_slug[:160],
                    description=final_desc,
                    order=next_order
                )
                if branch:
                    subject.branch = branch
                if semester:
                    subject.semester = semester
                subject.save()

            if not subject.modules.exists():
                for m_data in data["modules"]:
                    mod_num = m_data.get("module_number", 1)
                    mod_title = m_data.get("title") or f"Module {mod_num}"
                    mod_desc = m_data.get("description", "")

                    module = Module.objects.create(
                        subject=subject,
                        module_number=mod_num,
                        title=mod_title[:100],
                        description=mod_desc
                    )

                    raw_chapters = m_data.get("chapters", [])
                    if raw_chapters and not m_data.get("topics"):
                        unwrapped_topics = []
                        for chp_idx, chp in enumerate(raw_chapters, start=1):
                            c_num = chp.get("chapter_number", chp_idx)
                            c_title = chp.get("title") or chp.get("chapter_title") or f"Chapter {c_num}"
                            for t in chp.get("topics", []):
                                if isinstance(t, dict):
                                    t_dict = dict(t)
                                    t_dict.setdefault("chapter_number", c_num)
                                    t_dict.setdefault("chapter_title", c_title)
                                    unwrapped_topics.append(t_dict)
                                elif isinstance(t, str):
                                    unwrapped_topics.append({
                                        "title": t,
                                        "chapter_number": c_num,
                                        "chapter_title": c_title,
                                        "description": f"Core concepts and study resources for {t}.",
                                        "keywords": [t]
                                    })
                        topics_data = unwrapped_topics
                    else:
                        topics_data = m_data.get("topics", [])

                    for t_idx, t_data in enumerate(topics_data, start=1):
                        t_title = t_data.get("title") or f"Topic {t_idx}"
                        t_desc = t_data.get("description", "")
                        t_chp_num = t_data.get("chapter_number") or 1
                        t_chp_title = (t_data.get("chapter_title") or "").strip()

                        base_t_slug = slugify(t_title) or f"topic-{t_idx}"
                        t_slug = base_t_slug
                        t_counter = 1
                        while Topic.objects.filter(module=module, slug=t_slug).exists():
                            t_slug = f"{base_t_slug}-{t_counter}"
                            t_counter += 1

                        is_hy = bool(t_data.get("is_high_yield", False))
                        vtu_freq = str(t_data.get("vtu_frequency_tag") or "").strip()

                        topic = Topic.objects.create(
                            module=module,
                            chapter_number=t_chp_num,
                            chapter_title=t_chp_title[:150],
                            title=t_title[:200],
                            slug=t_slug[:220],
                            order=t_idx,
                            is_high_yield=is_hy,
                            vtu_frequency_tag=vtu_freq[:100],
                            description=t_desc
                        )

                        keywords_data = t_data.get("keywords", [])
                        for kw in keywords_data:
                            kw_clean = str(kw).strip()
                            if kw_clean:
                                TopicKeyword.objects.get_or_create(
                                    topic=topic,
                                    keyword=kw_clean[:120]
                                )

                    # Ensure each module has high-yield exam topics intelligently identified
                    if not module.topics.filter(is_high_yield=True).exists():
                        from curriculum.services.high_yield import tag_module_high_yield_topics
                        tag_module_high_yield_topics(module)

        # Update GenerationJob status to COMPLETED inside the same atomic transaction
        if job is not None:
            job.status = GenerationJob.Status.COMPLETED
            job.progress = 100
            job.step = 4
            job.subject = subject
            job.finished_at = dj_timezone.now()
            job.progress_message = (
                f"Syllabus for '{subject.name}' ({subject.code}) generated successfully with "
                f"{subject.modules.count()} modules!"
            )
            job.error_message = ""
            job.save(update_fields=['status', 'progress', 'step', 'subject', 'finished_at', 'progress_message', 'error_message'])

        logger.info(
            "Successfully persisted subject '%s' (%s) with %d modules using model '%s'. Source-grounded: %s",
            subject.name, subject.code, subject.modules.count(), active_model, is_verified_source
        )
        return subject


def generate_curriculum_from_prompt(
    subject_name,
    subject_code="",
    syllabus_text="",
    job_id=None,
    claim_token=None,
    lease_duration=300,
    branch=None,
    semester=None
) -> Subject:
    """
    Main service entrypoint: generates curriculum via Gemini/Groq and persists it to PostgreSQL.
    Preserves exact backward-compatibility for direct callers and legacy view invocations.
    When job_id is provided, coordinates atomic row-lock fencing validation during persistence.
    """
    curriculum_payload = generate_curriculum_data(
        subject_name=subject_name,
        subject_code=subject_code,
        syllabus_text=syllabus_text
    )
    return persist_curriculum_records(
        curriculum_payload=curriculum_payload,
        job_id=job_id,
        claim_token=claim_token,
        lease_duration=lease_duration,
        branch=branch,
        semester=semester
    )
