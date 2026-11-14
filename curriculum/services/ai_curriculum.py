import json
import logging
import re
import requests
from django.db import transaction
from django.utils.text import slugify
from decouple import config
from curriculum.models import Subject, Module, Topic, TopicKeyword

logger = logging.getLogger(__name__)


def resolve_active_gemini_endpoint(api_key):
    """
    Dynamically asks Google's ModelService.ListModels which models are enabled
    and active for this API key with generateContent support.
    Eliminates hardcoded 404 version mismatches and surfaces actionable diagnostics.
    """
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    try:
        res = requests.get(list_url, headers=headers, timeout=15)
        if res.status_code == 200:
            models_data = res.json().get("models", [])
            # Filter models that support generateContent
            supported = [
                m["name"] for m in models_data
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            if supported:
                # Prioritize: 1.5-flash, 2.0-flash, 2.5-flash, flash, pro
                chosen = None
                for pref in ("3.5-flash", "3.5-flash-lite","3.8-flash","1.5-flash", "2.0-flash", "2.5-flash", "flash", "pro"):
                    for m in supported:
                        if pref in m.lower():
                            chosen = m
                            break
                    if chosen:
                        break
                if not chosen:
                    chosen = supported[0]

                clean_model = chosen.replace("models/", "")
                logger.info(f"Discovered active Gemini model for this key: {clean_model}")
                return f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
            else:
                total_models = len(models_data)
                raise RuntimeError(
                    f"Google API key authenticated, but reports 0 models supporting 'generateContent' "
                    f"(Total models found: {total_models}). This typically occurs when the key has API restrictions "
                    f"or was created in Google Cloud Console without Generative Language API enabled. "
                    f"Please generate a fresh key directly at https://aistudio.google.com/apikey"
                )
        elif res.status_code in (400, 401, 403):
            try:
                err_msg = res.json().get("error", {}).get("message", res.text)
            except Exception:
                err_msg = res.text
            raise RuntimeError(
                f"Google API Key verification error ({res.status_code}): {err_msg}. "
                f"Please verify GEMINI_API_KEY at https://aistudio.google.com/apikey"
            )
        else:
            try:
                err_msg = res.json().get("error", {}).get("message", res.text)
            except Exception:
                err_msg = res.text
            raise RuntimeError(f"Google Generative Language API error ({res.status_code}): {err_msg}")
    except requests.RequestException as e:
        logger.warning(f"ListModels check failed ({e}). Proceeding with standard endpoint fallback.")

    return "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def generate_curriculum_from_prompt(subject_name, subject_code="", syllabus_text=""):
    """
    Calls the Gemini API to generate a complete 5-module university curriculum
    with topics, descriptions, and high-intent search keywords.
    Saves everything atomically into PostgreSQL and returns the created Subject instance.
    """
    raw_key = config('GEMINI_API_KEY', default='')
    api_key = raw_key.strip().strip('\'"') if raw_key else None

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not configured in your environment. "
            "Please add GEMINI_API_KEY to your .env file or Render environment variables."
        )

    # Clean input
    clean_name = subject_name.strip()
    clean_code = (subject_code.strip() or "COURSE").upper()
    has_custom_syllabus = bool(syllabus_text and syllabus_text.strip())

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
        "   - 'keywords': exactly 3 specific, high-intent search phrases for video lectures and technical articles (e.g. 'backpropagation algorithm derivation', 'gradient descent vs adam optimizer').\n"
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
        '          "keywords": ["keyword 1", "keyword 2", ...]\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    if has_custom_syllabus:
        user_prompt = (
            f"Subject: {clean_name}\n"
            f"Code: {clean_code}\n\n"
            f"Extract and organize the following syllabus into 5 modules with topics and search keywords:\n"
            f"\"\"\"\n{syllabus_text.strip()}\n\"\"\""
        )
    else:
        user_prompt = (
            f"Generate a comprehensive, standard 5-module university curriculum for the subject '{clean_name}' "
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
            "maxOutputTokens": 2048
        }
    }

    # Discover the active endpoint for this key dynamically, with fallbacks
    active_endpoint = resolve_active_gemini_endpoint(api_key)
    candidate_endpoints = [
        active_endpoint,
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
    ]
    unique_endpoints = list(dict.fromkeys(candidate_endpoints))

    response = None
    last_error_msg = ""
    post_headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    for endpoint in unique_endpoints:
        url = f"{endpoint}?key={api_key}"
        try:
            res = requests.post(url, json=payload, headers=post_headers, timeout=45)
            if res.status_code == 200:
                response = res
                break
            elif res.status_code == 404:
                logger.info(f"Endpoint '{endpoint}' 404. Attempting next candidate...")
                last_error_msg = (
                    f"Gemini API error (Status 404): {res.text}. "
                    "This usually indicates your Google Cloud project does not have Generative Language API enabled. "
                    "Fix: Generate a fresh key at https://aistudio.google.com/apikey (select 'Create key in new project')."
                )
                continue
            else:
                last_error_msg = f"Gemini API error (Status {res.status_code}): {res.text}"
                break
        except requests.RequestException as e:
            last_error_msg = f"Connection to Gemini API failed: {e}"
            continue

    if not response or response.status_code != 200:
        logger.error(f"All Gemini endpoints failed. Last error: {last_error_msg}")
        raise RuntimeError(last_error_msg or "Failed to connect to any active Gemini model endpoint.")

    try:
        res_json = response.json()
        candidates = res_json.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini API returned no response candidates.")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise ValueError("Gemini candidate contains no content parts.")

        # Search all parts for the one containing JSON brackets '{' and '}'
        raw_text = ""
        for part in parts:
            text = part.get("text", "")
            if "{" in text and "}" in text:
                raw_text = text
                break

        if not raw_text:
            # Fallback to the last part
            raw_text = parts[-1].get("text", "")

        if not raw_text:
            raise ValueError("Gemini returned empty text in candidate parts.")

        # Strip markdown fences (e.g. ```json ... ``` or ``` ... ```)
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Extract only the outermost JSON object bounds
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]

        # Parse JSON with fallback for trailing commas
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            cleaned_no_trailing = re.sub(r",\s*([\]}])", r"\1", cleaned)
            data = json.loads(cleaned_no_trailing)

    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse Gemini response: {e}. Raw response: {response.text[:500]}")
        raise ValueError(f"Failed to parse AI generated curriculum: {e}")

    # Validate essential keys
    if "modules" not in data or not isinstance(data["modules"], list):
        raise ValueError("Invalid curriculum structure: 'modules' key missing or invalid.")

    # Save to database in an atomic transaction
    with transaction.atomic():
        # Determine Subject code & name
        final_name = data.get("name") or clean_name
        final_code = data.get("code") or clean_code
        final_desc = data.get("description") or f"Curriculum and learning resources for {final_name}."

        # Unique slug generation
        base_slug = slugify(final_name) or "new-subject"
        unique_slug = base_slug
        slug_counter = 1
        while Subject.objects.filter(slug=unique_slug).exists():
            unique_slug = f"{base_slug}-{slug_counter}"
            slug_counter += 1

        # Next order index
        next_order = (Subject.objects.order_by('-order').values_list('order', flat=True).first() or 0) + 1

        subject = Subject.objects.create(
            name=final_name[:150],
            code=final_code[:20],
            slug=unique_slug[:160],
            description=final_desc,
            order=next_order
        )

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

                # Unique slug for topic within module
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

    logger.info(f"Successfully created subject '{subject.name}' ({subject.code}) with {subject.modules.count()} modules.")
    return subject
