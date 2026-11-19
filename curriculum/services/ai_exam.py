import json
import logging
import re
from django.utils import timezone
from django.utils.text import slugify
from curriculum.models import Module, Topic, ExamQuestion
from curriculum.services.ai_curriculum import (
    resolve_ai_provider,
    resolve_groq_models,
    _send_groq_request_with_retries,
    resolve_configured_models,
    CurriculumGenerationError,
)
import requests

logger = logging.getLogger(__name__)


def _call_ai_json(system_instructions: str, user_prompt: str, max_tokens: int = 4096) -> dict:
    """
    Dispatches a structured JSON request to the active AI provider (Groq or Gemini)
    with automatic model resolution, retry backoff, and JSON parsing.
    """
    provider, api_key = resolve_ai_provider()

    if provider == 'groq':
        primary_model, fallback_model = resolve_groq_models(api_key)
        
        # Try primary model
        res, err = _send_groq_request_with_retries(
            model_name=primary_model,
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            api_key=api_key,
            max_retries=2,
            request_timeout=35,
            is_fallback=False
        )

        # Fallback to secondary model if primary exhausted
        if res is None:
            logger.warning(
                "Primary Groq model '%s' failed for exam service (%s). Attempting fallback '%s'...",
                primary_model, err, fallback_model
            )
            res, err = _send_groq_request_with_retries(
                model_name=fallback_model,
                system_instructions=system_instructions,
                user_prompt=user_prompt,
                api_key=api_key,
                max_retries=2,
                request_timeout=35,
                is_fallback=True
            )

        if res is None:
            raise CurriculumGenerationError(f"Groq exam service failed across models: {err}")

        try:
            data = res.json()
            raw_content = data['choices'][0]['message']['content']
            return json.loads(raw_content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Failed to parse JSON response from Groq: %s", e)
            raise CurriculumGenerationError(f"Invalid JSON returned by Groq exam service: {e}")

    elif provider == 'gemini':
        primary_model, fallback_model = resolve_configured_models(api_key)
        models_to_try = [primary_model, fallback_model] if fallback_model else [primary_model]
        
        last_err = ""
        for model in models_to_try:
            clean_model = model.replace("models/", "")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }
            payload = {
                "system_instruction": {
                    "parts": [{"text": system_instructions}]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json"
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=35)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    raw_text = resp_json['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(raw_text)
                else:
                    last_err = f"Gemini HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as e:
                last_err = str(e)
                continue

        raise CurriculumGenerationError(f"Gemini exam service failed across models: {last_err}")

    raise CurriculumGenerationError(f"Unsupported AI provider: {provider}")


def generate_vtu_questions_for_module(module: Module, count: int = 5) -> list[ExamQuestion]:
    """
    Generates authentic, high-yield VTU exam questions for a given syllabus module.
    Analyzes the module topics and VTU curriculum patterns (10M, 8M, 6M questions).
    Persists newly generated questions to the database and returns the queryset.
    """
    subject = module.subject
    topics = list(module.topics.all().values_list('title', flat=True))
    topics_summary = "\n".join(f"- {t}" for t in topics) if topics else "(General module topics)"

    system_instructions = (
        "You are an expert VTU (Visvesvaraya Technological University) Examination Paper Setter "
        "and Senior Professor. Your job is to generate authentic, high-probability VTU semester exam questions "
        "for engineering students. VTU semester exams follow strict patterns: questions ask for neat block diagrams, "
        "step-by-step algorithms, derivations, architecture explanations, or structured comparison tables. "
        "Respond ONLY with a valid JSON object matching the requested schema."
    )

    user_prompt = f"""
Subject: {subject.name} ({subject.code})
Scheme: {subject.scheme or '2022 Scheme'}
Module {module.module_number}: {module.title}
Module Description: {module.description or 'Standard VTU syllabus'}

Topics covered in this module:
{topics_summary}

Please generate {count} authentic VTU examination questions for this module.
Requirements:
1. Question formats:
   - 10-Mark comprehensive questions (e.g., "Explain the architecture of X with a neat labeled diagram. [10M]")
   - 6 to 8-Mark specific/analytical questions (e.g., "Differentiate between X and Y with suitable examples. [6M]" or "Explain the working steps of Z algorithm with a diagram. [8M]")
2. Set frequency to 2 or 3 for core repetitive VTU topics, and 1 for newer topics.
3. Assign realistic VTU examination sessions for 'years_asked' (e.g. 'Jan 2023, Jul 2022', 'Dec 2021', 'Feb 2024').
4. Include 'topic_title' matching one of the topics listed above if applicable.

Return ONLY this JSON schema:
{{
  "questions": [
    {{
      "question_text": "Full question statement with marks indication",
      "marks": 10,
      "frequency": 3,
      "years_asked": "Jan 2023, Jul 2022, Dec 2021",
      "scheme": "{subject.scheme or '2022 Scheme'}",
      "topic_title": "Matching Topic Title"
    }}
  ]
}}
"""

    existing_questions = list(module.exam_questions.all())
    
    try:
        data = _call_ai_json(system_instructions, user_prompt, max_tokens=3000)
        raw_questions = data.get("questions", [])
        
        created_questions = []
        topic_map = {t.title.lower().strip(): t for t in module.topics.all()}

        for q_data in raw_questions:
            q_text = (q_data.get("question_text") or "").strip()
            if not q_text or len(q_text) < 10:
                continue

            # Prevent duplicate question statements for the same module
            if any(q_text.lower() in eq.question_text.lower() or eq.question_text.lower() in q_text.lower() for eq in existing_questions):
                continue

            marks = int(q_data.get("marks", 10))
            if marks not in (5, 6, 7, 8, 10, 12):
                marks = 10

            frequency = max(1, min(int(q_data.get("frequency", 1)), 5))
            years_asked = (q_data.get("years_asked") or "").strip()
            scheme = (q_data.get("scheme") or subject.scheme or "2022 Scheme").strip()

            matching_topic = None
            raw_topic = (q_data.get("topic_title") or "").lower().strip()
            if raw_topic:
                for t_name, t_obj in topic_map.items():
                    if raw_topic in t_name or t_name in raw_topic:
                        matching_topic = t_obj
                        break

            eq = ExamQuestion.objects.create(
                module=module,
                topic=matching_topic,
                question_text=q_text,
                marks=marks,
                frequency=frequency,
                years_asked=years_asked,
                scheme=scheme
            )
            created_questions.append(eq)
            existing_questions.append(eq)

        return existing_questions

    except Exception as e:
        logger.warning("AI question generation failed for module %s: %s", module.id, e)
        # If generation failed and there are no existing questions, create default fallback VTU questions
        if not existing_questions:
            fallback_qs = _generate_fallback_questions(module)
            return fallback_qs
        return existing_questions


def _generate_fallback_questions(module: Module) -> list[ExamQuestion]:
    """Generates standard VTU questions when external AI calls are unavailable."""
    created = []
    topics = list(module.topics.all()[:3])
    subject = module.subject

    if topics:
        t1 = topics[0]
        q1 = ExamQuestion.objects.create(
            module=module,
            topic=t1,
            question_text=f"Explain the architecture and working principles of {t1.title} with a neat labeled block diagram. [10M]",
            marks=10,
            frequency=3,
            years_asked="Jan 2023, Jul 2022",
            scheme=subject.scheme or "2022 Scheme"
        )
        created.append(q1)

        if len(topics) > 1:
            t2 = topics[1]
            q2 = ExamQuestion.objects.create(
                module=module,
                topic=t2,
                question_text=f"Differentiate between key approaches in {t2.title}. Discuss advantages, limitations, and practical applications. [8M]",
                marks=8,
                frequency=2,
                years_asked="Dec 2022, Feb 2024",
                scheme=subject.scheme or "2022 Scheme"
            )
            created.append(q2)
    else:
        q = ExamQuestion.objects.create(
            module=module,
            question_text=f"Discuss the fundamental concepts and system architecture of Module {module.module_number}: {module.title}. [10M]",
            marks=10,
            frequency=2,
            years_asked="Jul 2023",
            scheme=subject.scheme or "2022 Scheme"
        )
        created.append(q)

    return created


def generate_vtu_answer_blueprint(
    question_text: str,
    marks: int = 10,
    subject_name: str = "",
    module_title: str = "",
    topic_title: str = "",
    exam_question: ExamQuestion | None = None
) -> dict:
    """
    Generates a structured 'VTU 10-Mark Answer Architect Blueprint'.
    Returns a structured dictionary with:
    - time_budget_mins: Time allocation in exam
    - marks_breakdown: Rubric breakdown
    - definition: Crisp technical definition (1-2 marks)
    - diagram: Labeled ASCII/block diagram + drawing instructions (2-3 marks)
    - core_concepts: Numbered key concepts with formulas/equations (3-4 marks)
    - working_steps: Chronological mechanism steps
    - pros_and_cons: High-yield advantages and limitations
    - examiner_underlines: 5-8 crucial technical buzzwords students must underline in pencil
    - exam_hall_tips: Examiner evaluator advice to ensure full marks
    """
    # Check if blueprint already cached on the ExamQuestion
    if exam_question and exam_question.answer_blueprint:
        return exam_question.answer_blueprint

    time_budget = 15 if marks >= 10 else 9

    system_instructions = (
        "You are the Chief VTU Examination Evaluator and Senior Engineering Professor. "
        "Your task is to provide the ultimate 'VTU Answer Blueprint' that guarantees full marks (10/10 or 8/8) "
        "in semester exams. VTU examiners grade strictly on structured presentation, neat labeled diagrams, "
        "bulleted mechanism steps, and key technical keywords. "
        "Respond ONLY with a valid JSON object matching the requested schema."
    )

    user_prompt = f"""
Subject: {subject_name or 'Engineering Core Subject'}
Module / Context: {module_title} - {topic_title}
Question: {question_text}
Marks Weightage: {marks} Marks

Provide an exhaustive, high-scoring VTU Answer Blueprint for this question.

Include:
1. 'time_budget_mins': Recommended exam hall time (e.g. {time_budget} mins).
2. 'marks_breakdown': Array of components with marks allocated (e.g., Definition: 2M, Labeled Diagram: 3M, Working Steps: 3M, Applications/Pros: 2M).
3. 'definition': 2-3 sentence formal, rigorous definition.
4. 'diagram':
   - 'title': Title of diagram
   - 'ascii_art': A clean, readable ASCII or Unicode block diagram showing components and data/control flow
   - 'drawing_tips': Clear instructions on how students should draw and label it on their physical answer booklet.
5. 'core_concepts': 2 to 4 key theoretical points or subsections, each with 'heading', 'explanation', and optional 'formula'.
6. 'working_steps': 4 to 6 numbered chronological steps of operation.
7. 'pros_and_cons': 'advantages' (2-3 points) and 'limitations' (2-3 points).
8. 'examiner_underlines': 5 to 7 specific technical keywords/phrases students MUST underline in pencil to catch the evaluator's eye.
9. 'exam_hall_tips': 1 or 2 insider tips on what VTU examiners look for or where students commonly lose marks.

Return ONLY this JSON format:
{{
  "question": "{question_text}",
  "marks": {marks},
  "time_budget_mins": {time_budget},
  "marks_breakdown": [
    {{"component": "Definition & Principle", "marks": 2}},
    {{"component": "Neat Labeled Diagram", "marks": 3}},
    {{"component": "Working Mechanism", "marks": 3}},
    {{"component": "Key Advantages & Applications", "marks": 2}}
  ],
  "definition": "Clear conceptual definition...",
  "diagram": {{
    "title": "Block Diagram / Architecture Layout",
    "ascii_art": "+-------------------------+\\n|      Component A        |\\n+-------------------------+",
    "drawing_tips": "Pencil advice for drawing in exam..."
  }},
  "core_concepts": [
    {{
      "heading": "1. Core Concept",
      "explanation": "Detailed explanation...",
      "formula": "Relevant mathematical equation or syntax"
    }}
  ],
  "working_steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "pros_and_cons": {{
    "advantages": ["Advantage 1", "Advantage 2"],
    "limitations": ["Limitation 1", "Limitation 2"]
  }},
  "examiner_underlines": ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4", "Keyword 5"],
  "exam_hall_tips": "Evaluator advice..."
}}
"""

    try:
        blueprint = _call_ai_json(system_instructions, user_prompt, max_tokens=3500)
        
        # Cache on model if exam_question was supplied
        if exam_question:
            exam_question.answer_blueprint = blueprint
            exam_question.blueprint_generated_at = timezone.now()
            exam_question.save(update_fields=['answer_blueprint', 'blueprint_generated_at'])

        return blueprint

    except Exception as e:
        logger.error("Failed to generate answer blueprint: %s", e)
        # Generate clean structured fallback blueprint
        fallback = _generate_fallback_blueprint(question_text, marks, time_budget)
        if exam_question:
            exam_question.answer_blueprint = fallback
            exam_question.blueprint_generated_at = timezone.now()
            exam_question.save(update_fields=['answer_blueprint', 'blueprint_generated_at'])
        return fallback


def _generate_fallback_blueprint(question_text: str, marks: int, time_budget: int) -> dict:
    """Fallback blueprint structure when external AI is temporarily unreachable."""
    return {
        "question": question_text,
        "marks": marks,
        "time_budget_mins": time_budget,
        "marks_breakdown": [
            {"component": "Formal Definition & Principle", "marks": 2},
            {"component": "Neat Labeled Diagram", "marks": 3},
            {"component": "Step-by-Step Working Mechanism", "marks": 3},
            {"component": "Summary & Key Applications", "marks": 2 if marks >= 10 else 1}
        ],
        "definition": f"Core technical definition addressing: {question_text}. State the fundamental engineering principle clearly in 2-3 sentences without introductory fluff.",
        "diagram": {
            "title": "System Architecture / Procedural Flow",
            "ascii_art": (
                "+------------------+      +------------------+      +------------------+\n"
                "|   Input / Source | ---> | Processing Unit  | ---> |   Output Result  |\n"
                "+------------------+      +------------------+      +------------------+\n"
                "                                   |                                    \n"
                "                          +------------------+                          \n"
                "                          | Control / State  |                          \n"
                "                          +------------------+"
            ),
            "drawing_tips": "Draw using a ruler and HB pencil. Clearly label inputs on the left, primary processing in the center, and outputs on the right. Box all component names."
        },
        "core_concepts": [
            {
                "heading": "1. Architectural Components",
                "explanation": "Describe each primary functional block shown in the diagram and its role within the overall system.",
                "formula": "State any governing equations, transfer functions, or time complexity formulas."
            },
            {
                "heading": "2. Operational Workflow",
                "explanation": "Explain how data or control signals traverse the system from initialization to termination.",
                "formula": ""
            }
        ],
        "working_steps": [
            "Step 1: Initialization of system state and parameter calibration.",
            "Step 2: Processing and transformation of primary input vectors.",
            "Step 3: Intermediate state verification and feedback control loop execution.",
            "Step 4: Output generation, error checking, and final state commitment."
        ],
        "pros_and_cons": {
            "advantages": [
                "High operational throughput and computational efficiency.",
                "Modular architecture allowing independent scaling and testing."
            ],
            "limitations": [
                "Increased implementation complexity and resource overhead.",
                "Requires careful parameter tuning to avoid degradation."
            ]
        },
        "examiner_underlines": [
            "Functional Block Architecture",
            "State Transition Mechanism",
            "Throughput Efficiency",
            "Feedback Loop Control",
            "Boundary Conditions"
        ],
        "exam_hall_tips": "VTU evaluators look for the labeled diagram first. Dedicate the first 4 minutes to drawing the diagram neatly before writing detailed paragraphs."
    }


def generate_module_cheat_sheet(module: Module, force_refresh: bool = False) -> dict:
    """
    Generates or retrieves a cached 1-page compact revision cheat sheet for a module.
    Includes:
    - key_definitions: Core terms and high-yield definitions
    - acronyms: Important abbreviations and full forms
    - formulas_and_complexities: Governing equations and asymptotic complexities
    - architecture_summary: Block diagram ASCII art + structural takeaways
    - last_minute_rules: Crucial exam pearls to review right before entering exam hall
    """
    if not force_refresh and module.cheat_sheet:
        return module.cheat_sheet

    subject = module.subject
    topics = list(module.topics.all().values_list('title', flat=True))
    topics_str = ", ".join(topics) if topics else "General module topics"

    system_instructions = (
        "You are an elite Engineering Professor known for creating legendary 1-page exam revision cheat sheets. "
        "Students review these cheat sheets 30 minutes before exams for rapid retention. "
        "Keep descriptions ultra-dense, precise, and high-yield. Avoid introductory fluff. "
        "Respond ONLY with a valid JSON object matching the requested schema."
    )

    user_prompt = f"""
Subject: {subject.name} ({subject.code})
Scheme: {subject.scheme or '2022 Scheme'}
Module {module.module_number}: {module.title}
Module Description: {module.description or ''}
Topics: {topics_str}

Create an ultra-compact 1-Page Last-Minute Revision Cheat Sheet for this module.

Include:
1. 'key_definitions': 4 to 6 crucial definitions (concise 1-2 sentence definition per term).
2. 'acronyms': 3 to 6 essential acronyms with full forms and brief context.
3. 'formulas_and_complexities': 3 to 5 core mathematical formulas, equations, or time/space complexities.
4. 'architecture_summary':
   - 'title': Name of core system/algorithm architecture in this module
   - 'ascii_diagram': Clean, readable ASCII block diagram
   - 'key_takeaway': 1-2 sentence core design principle
5. 'last_minute_rules': 4 to 6 bullet points of high-probability exam rules, theorems, or common exam traps.

Return ONLY this JSON schema:
{{
  "module_number": {module.module_number},
  "module_title": "{module.title}",
  "subject_code": "{subject.code}",
  "subject_name": "{subject.name}",
  "key_definitions": [
    {{"term": "Term Name", "definition": "Direct technical definition."}}
  ],
  "acronyms": [
    {{"acronym": "ABC", "stands_for": "Always Be Coding", "context": "Usage context"}}
  ],
  "formulas_and_complexities": [
    {{"name": "Concept", "formula": "Mathematical form", "complexity": "O(n)"}}
  ],
  "architecture_summary": {{
    "title": "Architecture Name",
    "ascii_diagram": "+---+  -->  +---+\\n| A |       | B |\\n+---+       +---+",
    "key_takeaway": "Takeaway rule"
  }},
  "last_minute_rules": [
    "High-yield exam rule 1",
    "High-yield exam rule 2"
  ]
}}
"""

    try:
        data = _call_ai_json(system_instructions, user_prompt, max_tokens=3500)
        module.cheat_sheet = data
        module.cheat_sheet_generated_at = timezone.now()
        module.save(update_fields=['cheat_sheet', 'cheat_sheet_generated_at'])
        return data
    except Exception as e:
        logger.error("Failed to generate cheat sheet for module %s: %s", module.id, e)
        fallback = _generate_fallback_cheat_sheet(module)
        module.cheat_sheet = fallback
        module.cheat_sheet_generated_at = timezone.now()
        module.save(update_fields=['cheat_sheet', 'cheat_sheet_generated_at'])
        return fallback


def _generate_fallback_cheat_sheet(module: Module) -> dict:
    """Deterministic fallback cheat sheet when external AI calls fail."""
    subject = module.subject
    topics = list(module.topics.all()[:4])
    definitions = []
    for t in topics:
        definitions.append({
            "term": t.title,
            "definition": f"Core technical concept in {module.title}. Focus on its architectural role and functional mechanisms."
        })
    if not definitions:
        definitions.append({
            "term": module.title,
            "definition": "Fundamental module topic encompassing core engineering principles and system methodologies."
        })

    return {
        "module_number": module.module_number,
        "module_title": module.title,
        "subject_code": subject.code,
        "subject_name": subject.name,
        "key_definitions": definitions,
        "acronyms": [
            {"acronym": subject.code, "stands_for": subject.name, "context": "Course Identifier"},
            {"acronym": f"M{module.module_number}", "stands_for": module.title, "context": "Syllabus Module"}
        ],
        "formulas_and_complexities": [
            {"name": "System Complexity", "formula": "T(n) = O(n log n)", "complexity": "Log-linear processing"},
            {"name": "Space Requirement", "formula": "S(n) = O(n)", "complexity": "Linear memory footprint"}
        ],
        "architecture_summary": {
            "title": f"Module {module.module_number} Workflow Pipeline",
            "ascii_diagram": (
                "+------------------+      +-------------------+      +------------------+\n"
                "|   Input Phase    | ---> | Core Processing   | ---> | Output / State   |\n"
                "+------------------+      +-------------------+      +------------------+"
            ),
            "key_takeaway": "Inputs must be validated and normalized before traversal through the transformation stage."
        },
        "last_minute_rules": [
            f"Always start 10-mark questions in Module {module.module_number} with a neat labeled block diagram.",
            "Write the formal mathematical expression before explaining procedural steps.",
            "Highlight boundary conditions and edge cases to score full technical marks."
        ]
    }


def generate_module_quiz(module: Module, force_refresh: bool = False) -> dict:
    """
    Generates or retrieves a cached 5-minute Micro-Quiz and Flashcard deck for self-assessment.
    Contains:
    - questions: 5-6 conceptual MCQs with options, correct answer index, and 1-sentence explanations.
    - flashcards: 5-8 active recall prompt/answer pairs.
    """
    if not force_refresh and module.quiz_data:
        return module.quiz_data

    subject = module.subject
    topics = list(module.topics.all().values_list('title', flat=True))
    topics_str = ", ".join(topics) if topics else "General module topics"

    system_instructions = (
        "You are an expert Engineering Professor designing a rapid 5-minute Micro-Quiz and Active Recall Flashcard set. "
        "Questions must test genuine conceptual understanding (mechanisms, trade-offs, definitions) rather than trivial trivia. "
        "Provide exactly 5 multiple choice questions and 6 flashcards. "
        "Respond ONLY with a valid JSON object matching the requested schema."
    )

    user_prompt = f"""
Subject: {subject.name} ({subject.code})
Module {module.module_number}: {module.title}
Topics: {topics_str}

Generate a self-assessment quiz deck:
1. 'questions': 5 conceptual multiple choice questions.
   - 'question': Clear question statement
   - 'options': Exactly 4 plausible answer options
   - 'correct_answer': 0-based integer index (0, 1, 2, or 3) of the correct option
   - 'explanation': Exactly 1 crisp sentence explaining why that choice is correct.
2. 'flashcards': 5 to 7 high-impact active recall cards:
   - 'front': A question or concept prompt to test recall (e.g. 'What is the primary trade-off in X?')
   - 'back': The direct, punchy technical answer or explanation.

Return ONLY this JSON schema:
{{
  "module_number": {module.module_number},
  "questions": [
    {{
      "id": 1,
      "question": "Question text?",
      "options": ["Option 0", "Option 1", "Option 2", "Option 3"],
      "correct_answer": 1,
      "explanation": "Option 1 is correct because..."
    }}
  ],
  "flashcards": [
    {{
      "id": 1,
      "front": "Prompt / Question?",
      "back": "Answer / Mechanism"
    }}
  ]
}}
"""

    try:
        data = _call_ai_json(system_instructions, user_prompt, max_tokens=3000)
        module.quiz_data = data
        module.quiz_generated_at = timezone.now()
        module.save(update_fields=['quiz_data', 'quiz_generated_at'])
        return data
    except Exception as e:
        logger.error("Failed to generate quiz for module %s: %s", module.id, e)
        fallback = _generate_fallback_quiz(module)
        module.quiz_data = fallback
        module.quiz_generated_at = timezone.now()
        module.save(update_fields=['quiz_data', 'quiz_generated_at'])
        return fallback


def _generate_fallback_quiz(module: Module) -> dict:
    """Deterministic fallback quiz and flashcards when external AI is unavailable."""
    topics = list(module.topics.all()[:3])
    t1_title = topics[0].title if topics else module.title
    t2_title = topics[1].title if len(topics) > 1 else "Engineering Design"

    return {
        "module_number": module.module_number,
        "questions": [
            {
                "id": 1,
                "question": f"What is the primary functional objective of {t1_title} in Module {module.module_number}?",
                "options": [
                    "Direct state transformation and integrity verification",
                    "Random memory allocation without bounds checking",
                    "Redundant loop iteration without state change",
                    "Ignoring input parity and hardware constraints"
                ],
                "correct_answer": 0,
                "explanation": f"{t1_title} establishes structured state transformation and guarantees processing integrity."
            },
            {
                "id": 2,
                "question": f"Which trade-off is most characteristic of systems utilizing {t2_title}?",
                "options": [
                    "Zero latency at the expense of non-determinism",
                    "Throughput and precision versus computational and memory overhead",
                    "Infinite storage capacity with degraded consistency",
                    "Complete loss of fault tolerance"
                ],
                "correct_answer": 1,
                "explanation": "Engineering implementations consistently balance higher accuracy and throughput against memory and CPU consumption."
            }
        ],
        "flashcards": [
            {
                "id": 1,
                "front": f"Define the core principle of {t1_title}.",
                "back": f"{t1_title} governs the primary operational behavior and data representations within Module {module.module_number}."
            },
            {
                "id": 2,
                "front": "Why are diagrams essential in VTU evaluation rubrics?",
                "back": "VTU scheme evaluation rubrics assign 2-3 marks specifically to labeled diagrams showing inputs, outputs, and control flows."
            }
        ]
    }

