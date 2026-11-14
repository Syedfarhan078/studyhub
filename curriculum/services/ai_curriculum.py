import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests
from django.db import transaction
from django.utils.text import slugify
from decouple import config

from curriculum.models import Subject, Module, Topic, TopicKeyword

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


class GeminiAuthError(CurriculumGenerationError):
    """Raised for authentication or permission errors (HTTP 401, 403)."""
    pass


class GeminiInvalidRequestError(CurriculumGenerationError):
    """Raised for bad or malformed requests (HTTP 400)."""
    pass


class GeminiModelNotFoundError(CurriculumGenerationError):
    """Raised when the requested Gemini model is not found (HTTP 404)."""
    pass


class CurriculumParseError(CurriculumGenerationError):
    """Raised when the AI response cannot be parsed into a valid curriculum structure."""
    pass


class CurriculumValidationError(CurriculumGenerationError):
    """Raised when the generated curriculum fails validation against the prescribed source syllabus."""
    pass


# =====================================================================
# Helper Functions: Sanitization, Timing, and Model Resolution
# =====================================================================

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

    # Scrub query param keys or raw Google API keys
    msg = re.sub(r'key=[A-Za-z0-9_\-]+', 'key=[REDACTED]', msg)
    msg = re.sub(r'AIza[0-9A-Za-z-_]{35}', '[REDACTED_API_KEY]', msg)
    return msg.strip()[:300]


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
    Prioritizes explicit environment configuration (GEMINI_PRIMARY_MODEL / GEMINI_FALLBACK_MODEL).
    If not specified, queries Google's ModelService.ListModels to detect active models
    supporting generateContent for this key, ensuring valid model IDs.
    Returns (primary_model, fallback_model).
    """
    raw_primary = config('GEMINI_PRIMARY_MODEL', default=config('GEMINI_MODEL', default='')).strip()
    raw_fallback = config('GEMINI_FALLBACK_MODEL', default='').strip()

    primary_model = raw_primary.replace('models/', '') if raw_primary else None
    fallback_model = raw_fallback.replace('models/', '') if raw_fallback else None

    if primary_model and fallback_model:
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

    priority_order = ("3.8-flash", "3.5-flash", "3.5-flash-lite", "2.0-flash", "2.5-flash", "1.5-flash", "flash")

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
            primary_model = "gemini-3.5-flash"

    if not fallback_model:
        if discovered:
            for pref in priority_order:
                match = next((m for m in discovered if pref in m.lower() and m != primary_model), None)
                if match:
                    fallback_model = match
                    break
        if not fallback_model:
            if primary_model == "gemini-3.5-flash-lite":
                fallback_model = "gemini-3.5-flash"
            else:
                fallback_model = "gemini-3.5-flash-lite"

    logger.info(
        "Active Gemini models resolved - Primary: '%s', Fallback: '%s'",
        primary_model, fallback_model
    )
    return primary_model, fallback_model


def resolve_active_gemini_endpoint(api_key: str) -> str:
    """Backwards-compatible endpoint resolver for existing callers."""
    primary, _ = resolve_configured_models(api_key)
    return f"https://generativelanguage.googleapis.com/v1beta/models/{primary}:generateContent"


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
    only if primary exhausts retries for eligible transient errors.
    """
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

    if fallback_model and fallback_model != primary_model:
        logger.warning(
            "Primary model '%s' exhausted %d retries with transient errors (%s). "
            "Switching to fallback model '%s'...",
            primary_model, max_retries, last_err, fallback_model
        )
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

    logger.error("Curriculum generation failed: All eligible retries exhausted. %s", last_err)
    raise GeminiTransientError(
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

def generate_curriculum_from_prompt(subject_name, subject_code="", syllabus_text=""):
    """
    Calls the Gemini API to generate a source-grounded university curriculum.
    - When official syllabus text is supplied: extracts and preserves all 5 module titles
      and prescribed topics, validating the AI output against the source.
    - When syllabus text is omitted: checks verified official syllabus records for known
      course codes (e.g. BAD703, BAI701), or marks the generated subject as an unverified
      AI reference curriculum.
    Saves everything atomically into PostgreSQL and returns the Subject instance.
    Prevents duplicate curriculum records if a request is retried.
    """
    raw_key = config('GEMINI_API_KEY', default='')
    api_key = raw_key.strip().strip('\'"') if raw_key else None

    if not api_key:
        raise GeminiAuthError(
            "GEMINI_API_KEY is not configured in your environment. "
            "Please add GEMINI_API_KEY to your .env file or Render environment variables."
        )

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
        "2. Each Module must contain exactly 3 clear, focused Topics (15 topics total).\n"
        "3. Each Topic must include:\n"
        "   - 'title': concise topic title (under 80 characters)\n"
        "   - 'description': 1 sentence explaining key conceptual objective\n"
        "   - 'keywords': exactly 3 specific, high-intent search phrases for video lectures and technical articles.\n"
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
        '          "title": "string",\n'
        '          "description": "string",\n'
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
            m_topics = ", ".join(m["topics"]) if m["topics"] else m.get("raw_text", "")
            spec = f"Module {m['module_number']}: {m['title']}"
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
            f"5. Do NOT invent alternative modules, and do NOT replace prescribed titles."
        )
    elif has_custom_syllabus:
        user_prompt = (
            f"Subject: {clean_name}\n"
            f"Course Code: {clean_code}\n\n"
            f"Extract and organize the following syllabus into exactly 5 modules with topics and search keywords:\n"
            f"\"\"\"\n{syllabus_text.strip()}\n\"\"\""
        )
    else:
        user_prompt = (
            f"Generate a comprehensive reference 5-module university curriculum for the subject '{clean_name}' "
            f"(Course Code: '{clean_code}'). Provide detailed topics and realistic educational search keywords."
        )

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

    # Validate against authoritative prescribed source modules before database insertion
    data = validate_curriculum_against_source(data, prescribed_modules=prescribed_modules)

    # Save to database in an atomic transaction (idempotent duplicate prevention)
    with transaction.atomic():
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
            return existing_subject

        if existing_subject:
            subject = existing_subject
            subject.name = final_name[:150]
            subject.code = final_code[:20]
            subject.description = final_desc
            subject.save()
        else:
            base_slug = slugify(final_name) or "new-subject"
            unique_slug = base_slug
            slug_counter = 1
            while Subject.objects.filter(slug=unique_slug).exists():
                unique_slug = f"{base_slug}-{slug_counter}"
                slug_counter += 1

            next_order = (Subject.objects.order_by('-order').values_list('order', flat=True).first() or 0) + 1

            subject = Subject.objects.create(
                name=final_name[:150],
                code=final_code[:20],
                slug=unique_slug[:160],
                description=final_desc,
                order=next_order
            )

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

                topics_data = m_data.get("topics", [])
                for t_idx, t_data in enumerate(topics_data, start=1):
                    t_title = t_data.get("title") or f"Topic {t_idx}"
                    t_desc = t_data.get("description", "")

                    base_t_slug = slugify(t_title) or f"topic-{t_idx}"
                    t_slug = base_t_slug
                    t_counter = 1
                    while Topic.objects.filter(module=module, slug=t_slug).exists():
                        t_slug = f"{base_t_slug}-{t_counter}"
                        t_counter += 1

                    topic = Topic.objects.create(
                        module=module,
                        title=t_title[:200],
                        slug=t_slug[:220],
                        order=t_idx,
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

    logger.info(
        "Successfully created subject '%s' (%s) with %d modules using model '%s'. Source-grounded: %s",
        subject.name, subject.code, subject.modules.count(), active_model, is_verified_source
    )
    return subject
