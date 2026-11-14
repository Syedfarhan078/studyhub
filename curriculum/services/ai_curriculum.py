import json
import logging
import requests
from django.db import transaction
from django.utils.text import slugify
from decouple import config
from curriculum.models import Subject, Module, Topic, TopicKeyword

logger = logging.getLogger(__name__)

CANDIDATE_ENDPOINTS = [
    "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
    "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-pro:generateContent",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
]


def generate_curriculum_from_prompt(subject_name, subject_code="", syllabus_text=""):
    """
    Calls the Gemini API to generate a complete 5-module university curriculum
    with topics, descriptions, and high-intent search keywords.
    Saves everything atomically into PostgreSQL and returns the created Subject instance.
    """
    api_key = config('GEMINI_API_KEY', default=None)
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
        "2. Each Module must contain 4 to 8 clear, distinct Topics.\n"
        "3. Each Topic must include:\n"
        "   - 'title': concise topic title (under 100 characters)\n"
        "   - 'description': 1 to 2 sentences explaining key conceptual objectives\n"
        "   - 'keywords': a list of 6 to 10 specific, realistic search phrases optimized for finding video lectures "
        "and technical documentation (e.g., 'backpropagation algorithm derivation neural networks', 'gradient descent vs adam optimizer tutorial').\n"
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
            "temperature": 0.2
        }
    }

    response = None
    last_error_msg = ""

    for endpoint in CANDIDATE_ENDPOINTS:
        url = f"{endpoint}?key={api_key}"
        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
            if res.status_code == 200:
                response = res
                break
            elif res.status_code == 404:
                logger.info(f"Endpoint '{endpoint}' not found (404). Trying next fallback endpoint...")
                last_error_msg = f"Gemini API error (Status 404): {res.text}"
                continue
            else:
                last_error_msg = f"Gemini API error (Status {res.status_code}): {res.text}"
                break
        except requests.RequestException as e:
            last_error_msg = f"Connection to Gemini API failed: {e}"
            continue

    if not response or response.status_code != 200:
        logger.error(f"All Gemini candidate endpoints failed. Last error: {last_error_msg}")
        raise RuntimeError(last_error_msg or "Failed to connect to any supported Gemini model endpoint.")

    try:
        res_json = response.json()
        candidates = res_json.get("candidates", [])
        if not candidates:
            raise ValueError("Gemini API returned no response candidates.")
        
        raw_text = candidates[0]["content"]["parts"][0]["text"]
        data = json.loads(raw_text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Gemini response: {e}")
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
