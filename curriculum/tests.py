import json
from unittest.mock import MagicMock, call, patch

import requests
from django.contrib.messages import get_messages
from django.test import Client, TestCase, TransactionTestCase, override_settings
from django.urls import reverse
from django.core.management.base import CommandError

import uuid
from datetime import timedelta
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.utils import timezone
from django.core.files.uploadedfile import SimpleUploadedFile
from curriculum.models import Module, Subject, Topic, TopicKeyword, GenerationJob, ModuleNote, ExamQuestion, TopicProgress
from curriculum.services.ai_curriculum import (
    CurriculumParseError,
    CurriculumValidationError,
    CurriculumFencingError,
    generate_curriculum_data,
    persist_curriculum_records,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    GeminiTransientError,
    AIAuthError,
    AIPermissionError,
    AIModelNotFoundError,
    AIRateLimitError,
    AIServerError,
    AINetworkError,
    AITransientError,
    AIInvalidRequestError,
    calculate_backoff_delay,
    generate_curriculum_from_prompt,
    parse_prescribed_syllabus,
    _extract_chapters_from_text,
    resolve_configured_models,
    resolve_ai_provider,
    resolve_groq_models,
    normalize_groq_model_name,
    sanitize_error_message,
)
from curriculum.worker import (
    start_in_process_worker,
    stop_in_process_worker,
    notify_new_job,
    get_worker_thread,
    is_worker_accepting_jobs,
    GenerationWorkerEngine,
)


def make_mock_gemini_json_response(data_dict, status_code=200, headers=None):
    """Helper creating a mock requests.Response with Gemini candidates structure."""
    res = requests.Response()
    res.status_code = status_code
    if headers:
        res.headers.update(headers)
    body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(data_dict)}
                    ]
                }
            }
        ]
    }
    res._content = json.dumps(body).encode("utf-8")
    return res


def make_mock_groq_json_response(data_dict, status_code=200, headers=None):
    """Helper creating a mock requests.Response with Groq chat completion choices structure."""
    res = requests.Response()
    res.status_code = status_code
    if headers:
        res.headers.update(headers)
    body = {
        "id": "chatcmpl-test-groq",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(data_dict)
                },
                "finish_reason": "stop"
            }
        ]
    }
    res._content = json.dumps(body).encode("utf-8")
    return res


def make_mock_error_response(status_code, error_message, headers=None):
    """Helper creating a mock requests.Response representing Gemini errors."""
    res = requests.Response()
    res.status_code = status_code
    if headers:
        res.headers.update(headers)
    res._content = json.dumps({
        "error": {
            "code": status_code,
            "message": error_message,
            "status": "UNAVAILABLE" if status_code == 503 else "ERROR"
        }
    }).encode("utf-8")
    return res


SAMPLE_CURRICULUM_DATA = {
    "code": "BCS501",
    "name": "Software Engineering",
    "description": "Comprehensive course on software engineering methodologies and lifecycle.",
    "modules": [
        {
            "module_number": i,
            "title": f"Module {i} Foundations",
            "description": f"Detailed coverage of module {i} core topics.",
            "topics": [
                {
                    "title": f"Topic {i}.{j} Core Concepts",
                    "description": f"Fundamental explanation for topic {i}.{j}",
                    "keywords": [f"search phrase {i}.{j}.1", f"search phrase {i}.{j}.2", f"search phrase {i}.{j}.3"]
                }
                for j in range(1, 4)
            ]
        }
        for i in range(1, 6)
    ]
}

EXPECTED_BAD703_MODULE_TITLES = [
    "Classical Encryption & Block Ciphers",
    "Public-Key Cryptography & Key Exchange",
    "Key Management & Public-Key Infrastructure",
    "Web & IP Security",
    "Data Security & Data Hiding / Privacy",
]

BAD703_OFFICIAL_SYLLABUS_TEXT = """
Module 1: Classical Encryption & Block Ciphers
Topics: Symmetric cipher model, Substitution ciphers, Transposition ciphers, DES, Block cipher principles.

Module 2: Public-Key Cryptography & Key Exchange
Topics: Principles of public key cryptosystems, RSA algorithm, Diffie-Hellman key exchange, Elliptic curve cryptography.

Module 3: Key Management & Public-Key Infrastructure
Topics: Distribution of symmetric keys, Distribution of public keys, X.509 certificates, Public key infrastructure.

Module 4: Web & IP Security
Topics: IP Security overview and architecture, Encapsulating security payload, SSL/TLS architecture, HTTPS.

Module 5: Data Security & Data Hiding / Privacy
Topics: Data hiding techniques, Steganography, Digital watermarking, Privacy preserving data mining, Privacy regulations.
"""


class SourceGroundedCurriculumTests(TestCase):
    """
    Test suite for source-grounded curriculum generation:
    - BAD703 regression test preserving exact prescribed module titles
    - Parser handling of varied syllabus formatting (headings, roman numerals, bullets)
    - Rejection of hallucinated/incorrect module titles
    - Rejection of missing prescribed topics
    - Unverified AI marking when syllabus is omitted
    - Fallback retrieval from verified seed repository
    - 503 retries, transient backoff, and view-level graceful handling
    """

    def setUp(self):
        self.client = Client()

    def test_calculate_backoff_delay_exponential(self):
        """Tests that exponential backoff increases delay up to max_delay."""
        d1 = calculate_backoff_delay(attempt=1, base_delay=1.0, max_delay=8.0)
        d2 = calculate_backoff_delay(attempt=2, base_delay=1.0, max_delay=8.0)
        d3 = calculate_backoff_delay(attempt=3, base_delay=1.0, max_delay=8.0)
        d4 = calculate_backoff_delay(attempt=4, base_delay=1.0, max_delay=8.0)

        self.assertTrue(1.0 <= d1 <= 1.5)
        self.assertTrue(2.0 <= d2 <= 2.5)
        self.assertTrue(4.0 <= d3 <= 4.5)
        self.assertEqual(d4, 8.0)

    def test_calculate_backoff_delay_respects_retry_after_header(self):
        """Tests that a numeric Retry-After header is respected and capped."""
        delay = calculate_backoff_delay(attempt=1, retry_after_header="3.5", max_delay=8.0)
        self.assertEqual(delay, 3.5)

        capped = calculate_backoff_delay(attempt=1, retry_after_header="60", max_delay=8.0)
        self.assertEqual(capped, 8.0)

    def test_sanitize_error_message_scrubs_api_keys(self):
        """Tests that sensitive keys are completely redacted from error strings."""
        sensitive_text = "Failed with key=AIzaSyDxyz1234567890abcdefghijklmnopqrst and secret"
        sanitized = sanitize_error_message(sensitive_text)
        self.assertNotIn("AIzaSyDxyz1234567890abcdefghijklmnopqrst", sanitized)
        self.assertIn("[REDACTED]", sanitized)

    @patch("curriculum.services.ai_curriculum.config")
    def test_resolve_configured_models_from_env(self, mock_config):
        """Tests that GEMINI_PRIMARY_MODEL and GEMINI_FALLBACK_MODEL from env are used directly."""
        def config_side_effect(key, default="", cast=None):
            if key in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL"):
                return "gemini-3.8-flash"
            if key == "GEMINI_FALLBACK_MODEL":
                return "gemini-3.5-flash-lite"
            return default

        mock_config.side_effect = config_side_effect
        primary, fallback = resolve_configured_models("dummy-key")
        self.assertEqual(primary, "gemini-3.8-flash")
        self.assertEqual(fallback, "gemini-3.5-flash-lite")

    @patch("curriculum.services.ai_curriculum.time.sleep")
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_successful_fallback_after_primary_exhausts_retries(self, mock_config, mock_post, mock_sleep):
        """
        Tests that when primary model exhausts all 3 retries with HTTP 503,
        execution switches to the fallback model which succeeds on its 1st attempt.
        """
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else 3 if k == "GEMINI_MAX_RETRIES"
            else default
        )

        resp_503 = make_mock_error_response(503, "High server demand.")
        resp_200 = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)

        mock_post.side_effect = [resp_503, resp_503, resp_503, resp_503, resp_200]

        subject = generate_curriculum_from_prompt("Software Engineering", "BCS501_FALLBACK")

        self.assertIsNotNone(subject)
        self.assertEqual(subject.modules.count(), 5)
        self.assertEqual(mock_post.call_count, 5)

    # -----------------------------------------------------------------
    # 1. Syllabus Parser Tests
    # -----------------------------------------------------------------

    def test_parse_prescribed_syllabus_standard_headings(self):
        """Tests parsing standard 'Module X: Title' syllabus text."""
        parsed = parse_prescribed_syllabus(BAD703_OFFICIAL_SYLLABUS_TEXT)
        self.assertEqual(len(parsed), 5)
        titles = [m["title"] for m in parsed]
        self.assertEqual(titles, EXPECTED_BAD703_MODULE_TITLES)
        self.assertIn("Symmetric cipher model", parsed[0]["topics"])

    def test_parse_prescribed_syllabus_roman_numerals_and_units(self):
        """Tests parsing with 'UNIT I - Title' format."""
        raw = """
        Unit I - Classical Encryption & Block Ciphers
        Unit II - Public-Key Cryptography & Key Exchange
        Unit III - Key Management & Public-Key Infrastructure
        Unit IV - Web & IP Security
        Unit V - Data Security & Data Hiding / Privacy
        """
        parsed = parse_prescribed_syllabus(raw)
        self.assertEqual(len(parsed), 5)
        titles = [m["title"] for m in parsed]
        self.assertEqual(titles, EXPECTED_BAD703_MODULE_TITLES)

    def test_parse_prescribed_syllabus_numbered_lines(self):
        """Tests parsing when user pastes simple numbered lines."""
        raw = """
        1. Classical Encryption & Block Ciphers
        2. Public-Key Cryptography & Key Exchange
        3. Key Management & Public-Key Infrastructure
        4. Web & IP Security
        5. Data Security & Data Hiding / Privacy
        """
        parsed = parse_prescribed_syllabus(raw)
        self.assertEqual(len(parsed), 5)
        titles = [m["title"] for m in parsed]
        self.assertEqual(titles, EXPECTED_BAD703_MODULE_TITLES)

    # -----------------------------------------------------------------
    # 2. BAD703 Regression Test (Requirement 6)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_bad703_regression_preserves_expected_module_titles(self, mock_config, mock_post):
        """
        Regression Test: Confirms that generating BAD703 with official syllabus text
        preserves all 5 expected module titles in the database without inventing replacements.
        """
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else default
        )

        gemini_response_data = {
            "code": "BAD703",
            "name": "Data Security and Privacy",
            "description": "VTU scheme course on cryptography, network security, and data privacy.",
            "modules": [
                {
                    "module_number": idx,
                    "title": title,
                    "description": f"Curriculum for {title}",
                    "topics": [
                        {
                            "title": f"{title} Core Concept {t}",
                            "description": f"Comprehensive objective for topic {t}",
                            "keywords": [f"search keyword {idx} {t} a", f"search keyword {idx} {t} b", f"search keyword {idx} {t} c"]
                        }
                        for t in range(1, 4)
                    ]
                }
                for idx, title in enumerate(EXPECTED_BAD703_MODULE_TITLES, start=1)
            ]
        }
        mock_post.return_value = make_mock_gemini_json_response(gemini_response_data)

        subject = generate_curriculum_from_prompt(
            subject_name="Data Security and Privacy",
            subject_code="BAD703",
            syllabus_text=BAD703_OFFICIAL_SYLLABUS_TEXT
        )

        self.assertIsNotNone(subject)
        saved_titles = list(subject.modules.order_by("module_number").values_list("title", flat=True))
        self.assertEqual(saved_titles, EXPECTED_BAD703_MODULE_TITLES)
        self.assertEqual(subject.modules.count(), 5)
        self.assertEqual(Topic.objects.filter(module__subject=subject).count(), 15)

    # -----------------------------------------------------------------
    # 3. Missing Syllabus Input Marked as Unverified (Requirement 3 & 7)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_missing_syllabus_input_marks_curriculum_unverified(self, mock_config, mock_post):
        """
        Tests that when syllabus text is omitted for a new/unrecognized subject,
        the generated subject is annotated as unverified reference content.
        """
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )

        mock_post.return_value = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)

        subject = generate_curriculum_from_prompt(
            subject_name="Advanced Quantum Cryptography",
            subject_code="AQC901",
            syllabus_text=""
        )

        self.assertIn("Unverified", subject.description)
        self.assertIn("AI-Generated", subject.description)

    # -----------------------------------------------------------------
    # 4. Incorrect Module Titles Rejected (Requirement 5 & 7)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_incorrect_module_titles_rejected_by_validation(self, mock_config, mock_post):
        """
        Tests that if Gemini invents completely unrelated module titles
        instead of preserving the prescribed source titles, validation rejects it.
        """
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )

        # Gemini returns hallucinated module titles
        hallucinated_data = {
            "code": "BAD703",
            "name": "Data Security and Privacy",
            "description": "Hallucinated response",
            "modules": [
                {
                    "module_number": 1,
                    "title": "Introductory Web Development & CSS Styling",  # Incorrect!
                    "description": "HTML and CSS basics.",
                    "topics": [
                        {
                            "title": "CSS Grid",
                            "description": "Layouts",
                            "keywords": ["css", "grid", "web"]
                        }
                    ]
                },
                {
                    "module_number": 2,
                    "title": "Public-Key Cryptography & Key Exchange",
                    "description": "...",
                    "topics": [{"title": "RSA", "description": "algo", "keywords": ["rsa", "key", "math"]}]
                },
                {
                    "module_number": 3,
                    "title": "Key Management & Public-Key Infrastructure",
                    "description": "...",
                    "topics": [{"title": "PKI", "description": "certs", "keywords": ["pki", "x509", "cert"]}]
                },
                {
                    "module_number": 4,
                    "title": "Web & IP Security",
                    "description": "...",
                    "topics": [{"title": "IPsec", "description": "vpn", "keywords": ["ipsec", "vpn", "tunnel"]}]
                },
                {
                    "module_number": 5,
                    "title": "Data Security & Data Hiding / Privacy",
                    "description": "...",
                    "topics": [{"title": "Hiding", "description": "stego", "keywords": ["stego", "hide", "data"]}]
                },
            ]
        }
        mock_post.return_value = make_mock_gemini_json_response(hallucinated_data)

        with self.assertRaises(CurriculumValidationError):
            generate_curriculum_from_prompt(
                subject_name="Data Security and Privacy",
                subject_code="BAD703_TEST_FAIL",
                syllabus_text=BAD703_OFFICIAL_SYLLABUS_TEXT
            )

        self.assertEqual(Subject.objects.filter(code="BAD703_TEST_FAIL").count(), 0)

    # -----------------------------------------------------------------
    # 5. Missing Prescribed Topics Rejected (Requirement 5 & 7)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_missing_prescribed_topics_rejected_by_validation(self, mock_config, mock_post):
        """
        Tests that if prescribed source topics are silently omitted from the generated module,
        validation raises CurriculumValidationError and prevents database insertion.
        """
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )

        # Prescribed topics for Module 1 in BAD703_OFFICIAL_SYLLABUS_TEXT are:
        # "Symmetric cipher model, Substitution ciphers, Transposition ciphers, DES, Block cipher principles."
        # Here we mock Gemini returning topics that completely ignore those prescribed topics:
        omitted_topics_data = {
            "code": "BAD703",
            "name": "Data Security and Privacy",
            "description": "Response omitting topics",
            "modules": [
                {
                    "module_number": 1,
                    "title": "Classical Encryption & Block Ciphers",
                    "description": "Module 1",
                    "topics": [
                        {
                            "title": "Culinary Baking Techniques",
                            "description": "Baking sourdough bread in a high temperature stone oven",
                            "keywords": ["baking", "sourdough", "bread"]
                        }
                    ]
                },
                {
                    "module_number": 2,
                    "title": "Public-Key Cryptography & Key Exchange",
                    "description": "...",
                    "topics": [{"title": "RSA", "description": "Public key cryptosystem", "keywords": ["rsa", "prime", "math"]}]
                },
                {
                    "module_number": 3,
                    "title": "Key Management & Public-Key Infrastructure",
                    "description": "...",
                    "topics": [{"title": "X.509", "description": "Certificates", "keywords": ["x509", "pki", "cert"]}]
                },
                {
                    "module_number": 4,
                    "title": "Web & IP Security",
                    "description": "...",
                    "topics": [{"title": "IPsec", "description": "IP Security architecture", "keywords": ["ipsec", "ssl", "tls"]}]
                },
                {
                    "module_number": 5,
                    "title": "Data Security & Data Hiding / Privacy",
                    "description": "...",
                    "topics": [{"title": "Steganography", "description": "Data hiding techniques", "keywords": ["stego", "watermarking", "privacy"]}]
                },
            ]
        }
        mock_post.return_value = make_mock_gemini_json_response(omitted_topics_data)

        with self.assertRaises(CurriculumValidationError):
            generate_curriculum_from_prompt(
                subject_name="Data Security and Privacy",
                subject_code="BAD703_TEST_TOPICS_FAIL",
                syllabus_text=BAD703_OFFICIAL_SYLLABUS_TEXT
            )

        self.assertEqual(Subject.objects.filter(code="BAD703_TEST_TOPICS_FAIL").count(), 0)

    # -----------------------------------------------------------------
    # 6. Malformed Gemini JSON (Requirement 7)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_malformed_gemini_json_rejected(self, mock_config, mock_post):
        """Tests that malformed Gemini JSON raises CurriculumParseError."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )

        res = requests.Response()
        res.status_code = 200
        res._content = b'{"candidates": [{"content": {"parts": [{"text": "{ broken json: ["}]}}]}'
        mock_post.return_value = res

        with self.assertRaises(CurriculumParseError):
            generate_curriculum_from_prompt("Operating Systems", "BCS502", syllabus_text="")

    # -----------------------------------------------------------------
    # 7. Gemini 503 Transient Retries and Fallback (Requirement 7)
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.time.sleep")
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_gemini_503_retries_and_succeeds_on_primary(self, mock_config, mock_post, mock_sleep):
        """Tests that HTTP 503 triggers bounded exponential backoff and succeeds on retry."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else 3 if k == "GEMINI_MAX_RETRIES"
            else default
        )

        resp_503 = make_mock_error_response(503, "High server demand.")
        resp_200 = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)

        mock_post.side_effect = [resp_503, resp_503, resp_200]

        subject = generate_curriculum_from_prompt("Software Engineering", "BCS501_RETRY")
        self.assertIsNotNone(subject)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    # -----------------------------------------------------------------
    # 8. Idempotent Duplicate Prevention
    # -----------------------------------------------------------------

    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_prevent_duplicate_curriculum_when_request_retried(self, mock_config, mock_post):
        """Tests that retrying a generation request returns the existing subject without creating duplicates."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )
        mock_post.return_value = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)

        s1 = generate_curriculum_from_prompt("Software Engineering", "BCS501_DUP")
        s2 = generate_curriculum_from_prompt("Software Engineering", "BCS501_DUP")

        self.assertEqual(s1.id, s2.id)
        self.assertEqual(Subject.objects.filter(code="BCS501_DUP").count(), 1)
        self.assertEqual(Module.objects.filter(subject=s1).count(), 5)

    # -----------------------------------------------------------------
    # 9. View Layer Graceful Handling (Redirect + Message, No 500)
    # -----------------------------------------------------------------

    @patch("curriculum.views.generate_curriculum_from_prompt")
    def test_view_handles_validation_error_gracefully(self, mock_generate):
        """Tests that CurriculumValidationError produces a user-friendly error and clean redirect."""
        mock_generate.side_effect = CurriculumValidationError("Module 1 title mismatch")

        response = self.client.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "Test Subject", "subject_code": "TEST101", "syllabus_text": "text"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, "/")

        messages = list(get_messages(response.wsgi_request))
        self.assertEqual(len(messages), 1)
        self.assertIn("Syllabus validation failed", messages[0].message)

    @patch("curriculum.views.generate_curriculum_from_prompt")
    def test_view_handles_503_gracefully(self, mock_generate):
        """Tests that GeminiTransientError produces a user-friendly error and clean redirect."""
        mock_generate.side_effect = GeminiTransientError("503 Server Unavailable")

        response = self.client.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "Test Subject", "subject_code": "TEST102"}
        )

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.url, "/")

        messages = list(get_messages(response.wsgi_request))
        self.assertEqual(len(messages), 1)
        self.assertIn("temporarily experiencing high server demand", messages[0].message)


# =====================================================================
# PostgreSQL-Backed Background Task & Standalone Worker Tests
# =====================================================================

class PostgresBackgroundTaskWorkerTests(TestCase):
    """
    Test suite for PostgreSQL-backed background task queue and standalone worker:
    1. GenerationJob model lifecycle, UUID primary key, and lease expiration logic.
    2. AJAX job submission (<1s), PENDING job creation, and valid UUID task_id.
    3. AJAX submission validation (missing subject_name returns 400).
    4. Duplicate submission check (immediate COMPLETED return without queueing).
    5. Status polling endpoint (PENDING, RUNNING, COMPLETED, FAILED).
    6. Status polling validation (malformed UUID returns 400, unknown UUID returns 404).
    7. Ownership security (403 for unauthorized cross-user polling).
    8. Standalone worker successful job claiming, execution, and Subject persistence.
    9. Standalone worker Gemini HTTP 503 retry and recovery.
    10. Standalone worker network ReadTimeout triggering fallback model.
    11. Standalone worker all retries exhausted marking job FAILED with sanitized error.
    12. Standalone worker malformed JSON handling and clean failure.
    13. Standalone worker source-grounded validation failure and atomic rollback.
    14. Standalone worker authoritative BAD703 VTU syllabus preservation.
    15. Lease reaper: requeue abandoned RUNNING jobs when retries remain.
    16. Lease reaper: mark abandoned RUNNING jobs FAILED when max attempts exhausted.
    17. Generation modal template elements, progress container, and badges.
    """

    def setUp(self):
        self.client = Client()
        self.user_model = get_user_model()
        self.user_a = self.user_model.objects.create_user(username="student_a", password="password123")
        self.user_b = self.user_model.objects.create_user(username="student_b", password="password123")

    # 1. GenerationJob model lifecycle
    def test_generation_job_model_lifecycle(self):
        """Tests GenerationJob default values, UUID primary key, and lease expiration checks."""
        job = GenerationJob.objects.create(
            subject_name="Cloud Computing",
            subject_code="BCS702"
        )
        self.assertIsInstance(job.id, uuid.UUID)
        self.assertEqual(job.status, GenerationJob.Status.PENDING)
        self.assertEqual(job.progress, 0)
        self.assertEqual(job.step, 1)
        self.assertEqual(job.total_steps, 4)
        self.assertEqual(job.attempts, 0)
        self.assertEqual(job.max_attempts, 3)
        self.assertTrue(job.is_lease_expired(300))

        # Active lock within duration
        job.locked_at = timezone.now()
        job.save()
        self.assertFalse(job.is_lease_expired(300))

        # Expired lock
        job.locked_at = timezone.now() - timedelta(seconds=350)
        job.save()
        self.assertTrue(job.is_lease_expired(300))

    # 2. AJAX task submission creates pending job
    def test_ajax_task_submission_creates_pending_job(self):
        """Tests AJAX task submission returns 200 with UUID task_id and persists PENDING job."""
        resp = self.client.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "Distributed Systems", "subject_code": "BCS701"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data.get("status"), "PENDING")
        self.assertIn("task_id", data)

        task_uuid = uuid.UUID(data["task_id"])
        job = GenerationJob.objects.get(id=task_uuid)
        self.assertEqual(job.subject_name, "Distributed Systems")
        self.assertEqual(job.subject_code, "BCS701")
        self.assertEqual(job.status, GenerationJob.Status.PENDING)

    # 3. Missing subject_name returns 400
    def test_ajax_submission_without_subject_name_returns_400(self):
        """Tests that submitting an empty subject name via AJAX returns HTTP 400."""
        resp = self.client.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "", "subject_code": "EMPTY"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertFalse(data.get("success"))
        self.assertIn("error", data)

    # 4. Duplicate submission immediate return
    def test_duplicate_task_submission_immediate_return(self):
        """Tests that submitting a subject that already exists returns immediate COMPLETED without creating a job."""
        subject = Subject.objects.create(name="Computer Networks", code="BCS708", slug="computer-networks", order=10)
        for i in range(1, 6):
            Module.objects.create(subject=subject, module_number=i, title=f"Module {i}")

        initial_count = GenerationJob.objects.count()
        response = self.client.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "Computer Networks", "subject_code": "BCS708"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertEqual(data["status"], "COMPLETED")
        self.assertIsNone(data["task_id"])
        self.assertEqual(data["redirect_url"], f"/subjects/{subject.slug}/")
        self.assertEqual(GenerationJob.objects.count(), initial_count)

    # 5. Status endpoint for pending job
    def test_status_endpoint_pending_job(self):
        """Tests status endpoint reports PENDING for an unprocessed job."""
        job = GenerationJob.objects.create(
            user=self.user_a,
            subject_name="Compiler Design",
            subject_code="BCS707"
        )
        self.client.force_login(self.user_a)
        status_resp = self.client.get(
            reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}),
            HTTP_ACCEPT="application/json"
        )
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["status"], "PENDING")
        self.assertEqual(data["progress"], 0)
        self.assertEqual(data["step"], 1)
        self.assertIsNone(data["redirect_url"])
        self.assertIsNone(data["error"])

    # 6. Status endpoint validation: malformed and nonexistent UUIDs
    def test_status_endpoint_uuid_validation(self):
        """Tests that malformed UUIDs return 400 and nonexistent UUIDs return 404."""
        resp_400 = self.client.get("/subjects/generation-status/not-a-valid-uuid/")
        self.assertEqual(resp_400.status_code, 400)
        self.assertFalse(resp_400.json()["success"])

        random_uuid = str(uuid.uuid4())
        resp_404 = self.client.get(f"/subjects/generation-status/{random_uuid}/")
        self.assertEqual(resp_404.status_code, 404)
        self.assertFalse(resp_404.json()["success"])

    # 7. Status endpoint ownership isolation (SEC-01)
    def test_status_endpoint_ownership_isolation(self):
        """Tests strict ownership isolation: authenticated users, anonymous creators, and unauthorized visitors."""
        job = GenerationJob.objects.create(
            user=self.user_a,
            subject_name="Confidential Computing",
            subject_code="CC901"
        )

        # 1. User B cannot access User A's job -> 403
        self.client.force_login(self.user_b)
        resp_user_b = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(resp_user_b.status_code, 403)
        self.assertFalse(resp_user_b.json()["success"])

        # 2. Anonymous visitor cannot access User A's job -> 403
        self.client.logout()
        resp_anon = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(resp_anon.status_code, 403)
        self.assertFalse(resp_anon.json()["success"])

        # 3. User A can access their own job -> 200
        self.client.force_login(self.user_a)
        resp_user_a = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(resp_user_a.status_code, 200)
        self.assertEqual(resp_user_a.json()["task_id"], str(job.id))

        # 4. Anonymous creator generates a job in their session -> creator can access it (200)
        client_guest = Client()
        submit_resp = client_guest.post(
            reverse("curriculum:generate_subject_ai"),
            {"subject_name": "Guest Subject", "subject_code": "GST101"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest"
        )
        self.assertEqual(submit_resp.status_code, 200)
        guest_task_id = submit_resp.json()["task_id"]

        # Same guest session can query status -> 200
        poll_resp_same = client_guest.get(reverse("curriculum:generation_status", kwargs={"task_id": guest_task_id}))
        self.assertEqual(poll_resp_same.status_code, 200)

        # Different guest session cannot access that job -> 403
        client_other_guest = Client()
        poll_resp_diff = client_other_guest.get(reverse("curriculum:generation_status", kwargs={"task_id": guest_task_id}))
        self.assertEqual(poll_resp_diff.status_code, 403)

        # Authenticated user cannot access another session's anonymous job -> 403
        poll_resp_auth = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": guest_task_id}))
        self.assertEqual(poll_resp_auth.status_code, 403)

    # 8. Standalone worker successful job execution
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_successful_job_execution(self, mock_config, mock_post):
        """Tests that the standalone worker claims a PENDING job and generates the curriculum."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else default
        )
        mock_post.return_value = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)

        job = GenerationJob.objects.create(
            user=self.user_a,
            subject_name="Software Engineering",
            subject_code="BCS701",
            syllabus_text=""
        )

        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.step, 4)
        self.assertIsNotNone(job.subject)
        self.assertEqual(job.subject.name, "Software Engineering")
        self.assertEqual(job.subject.modules.count(), 5)
        self.assertEqual(job.attempts, 1)

        # Polling status endpoint returns COMPLETED with redirect_url
        self.client.force_login(self.user_a)
        status_resp = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.json()
        self.assertEqual(data["status"], "COMPLETED")
        self.assertEqual(data["redirect_url"], f"/subjects/{job.subject.slug}/")

    # 9. Standalone worker handles HTTP 503 retry and recovery
    @patch("curriculum.services.ai_curriculum.time.sleep")
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_gemini_503_retry_and_recovery(self, mock_config, mock_post, mock_sleep):
        """Tests that HTTP 503 triggers exponential retry in worker and succeeds."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else 3 if k == "GEMINI_MAX_RETRIES"
            else default
        )
        resp_503 = make_mock_error_response(503, "High server demand.")
        resp_200 = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)
        mock_post.side_effect = [resp_503, resp_200]

        job = GenerationJob.objects.create(subject_name="Web Security", subject_code="BCS703")
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(Subject.objects.filter(code="BCS703").count(), 1)

    # 10. Standalone worker network ReadTimeout triggers fallback model
    @patch("curriculum.services.ai_curriculum.time.sleep")
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_read_timeout_switches_to_fallback(self, mock_config, mock_post, mock_sleep):
        """Tests that a network ReadTimeout switches to fallback model without blocking worker indefinitely."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else default
        )
        timeout_err = requests.exceptions.ReadTimeout("HTTPSConnectionPool: Read timed out.")
        resp_200 = make_mock_gemini_json_response(SAMPLE_CURRICULUM_DATA)
        mock_post.side_effect = [timeout_err, timeout_err, resp_200]

        job = GenerationJob.objects.create(subject_name="Internet of Things", subject_code="BCS704")
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        self.assertIsNotNone(job.subject)

    # 11. Standalone worker all retries exhausted marks job FAILED with sanitized error
    @patch("curriculum.services.ai_curriculum.time.sleep")
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_all_retries_exhausted_marks_failed(self, mock_config, mock_post, mock_sleep):
        """Tests that exhausted retries across models marks job FAILED with sanitized user-facing message."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else "gemini-3.5-flash-lite" if k == "GEMINI_FALLBACK_MODEL"
            else 1 if k == "GEMINI_MAX_RETRIES"
            else default
        )
        resp_503 = make_mock_error_response(503, "High server demand.")
        mock_post.return_value = resp_503

        job = GenerationJob.objects.create(user=self.user_a, subject_name="Deep Learning", subject_code="BCS706")
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("temporarily experiencing high server demand", job.error_message)
        self.assertIsNone(job.subject)

        self.client.force_login(self.user_a)
        status_resp = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.json()
        self.assertFalse(data["success"])
        self.assertEqual(data["status"], "FAILED")
        self.assertIn("temporarily experiencing high server demand", data["error"])

    # 12. Standalone worker malformed JSON marks job FAILED cleanly
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_invalid_json_marks_failed_cleanly(self, mock_config, mock_post):
        """Tests that malformed JSON from model marks job FAILED without creating orphan database records."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )
        res = requests.Response()
        res.status_code = 200
        res._content = b'{"candidates": [{"content": {"parts": [{"text": "{ incomplete"}]}}]}'
        mock_post.return_value = res

        job = GenerationJob.objects.create(subject_name="Compiler Design", subject_code="BCS707")
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertEqual(Subject.objects.filter(code="BCS707").count(), 0)

    # 13. Standalone worker source-grounded validation failure and atomic rollback
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_validation_failure_rollbacks_atomically(self, mock_config, mock_post):
        """Tests that syllabus validation failure triggers a clean atomic rollback without leftover subject records."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )
        bad_data = json.loads(json.dumps(SAMPLE_CURRICULUM_DATA))
        bad_data["modules"][0]["title"] = "Completely Hallucinated Module"
        mock_post.return_value = make_mock_gemini_json_response(bad_data)

        job = GenerationJob.objects.create(
            subject_name="Data Security and Privacy",
            subject_code="BAD703_ROLLBACK_TEST",
            syllabus_text=BAD703_OFFICIAL_SYLLABUS_TEXT
        )
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("Syllabus validation failed", job.error_message)
        self.assertEqual(Subject.objects.filter(code="BAD703_ROLLBACK_TEST").count(), 0)
        self.assertEqual(Module.objects.filter(subject__code="BAD703_ROLLBACK_TEST").count(), 0)

    # 14. Standalone worker authoritative BAD703 VTU syllabus preservation
    @patch("curriculum.services.ai_curriculum.requests.post")
    @patch("curriculum.services.ai_curriculum.config")
    def test_worker_successful_bad703_source_grounded_execution(self, mock_config, mock_post):
        """Tests complete end-to-end background generation for BAD703 preserving all 5 module titles verbatim."""
        mock_config.side_effect = lambda k, default="", cast=None: (
            "test-key" if k == "GEMINI_API_KEY"
            else "gemini-3.5-flash" if k in ("GEMINI_PRIMARY_MODEL", "GEMINI_MODEL")
            else default
        )
        grounded_data = json.loads(json.dumps(SAMPLE_CURRICULUM_DATA))
        grounded_data["code"] = "BAD703"
        grounded_data["name"] = "Data Security and Privacy"
        for i, title in enumerate(EXPECTED_BAD703_MODULE_TITLES):
            grounded_data["modules"][i]["title"] = title
            grounded_data["modules"][i]["topics"] = [
                {"title": f"{title} Core Concept {j}", "description": "Covers syllabus", "keywords": ["k1", "k2", "k3"]}
                for j in range(1, 4)
            ]
        mock_post.return_value = make_mock_gemini_json_response(grounded_data)

        job = GenerationJob.objects.create(
            subject_name="Data Security and Privacy",
            subject_code="BAD703",
            syllabus_text=BAD703_OFFICIAL_SYLLABUS_TEXT
        )
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        saved_subject = Subject.objects.get(code="BAD703")
        saved_titles = list(saved_subject.modules.order_by("module_number").values_list("title", flat=True))
        self.assertEqual(saved_titles, EXPECTED_BAD703_MODULE_TITLES)

    # 15. Lease reaper requeues abandoned RUNNING jobs when retries remain
    def test_lease_reaper_requeues_abandoned_job(self):
        """Tests that stale RUNNING jobs whose lease expired are requeued to PENDING when retries remain."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        stale_time = timezone.now() - timedelta(seconds=400)
        job = GenerationJob.objects.create(
            subject_name="Orphaned Job 1",
            subject_code="ORPH01",
            status=GenerationJob.Status.RUNNING,
            attempts=1,
            max_attempts=3,
            locked_at=stale_time,
            locked_by="dead-worker-123"
        )

        worker = WorkerCommand()
        worker.reap_abandoned_jobs(lease_duration_seconds=300)

        job.refresh_from_db()
        # Should be requeued to PENDING with cleared lock
        self.assertEqual(job.status, GenerationJob.Status.PENDING)
        self.assertIsNone(job.locked_at)
        self.assertEqual(job.locked_by, "")
        self.assertIn("requeued", job.progress_message)

    # 16. Lease reaper marks abandoned RUNNING jobs FAILED when max attempts exhausted
    def test_lease_reaper_marks_failed_when_max_attempts_exhausted(self):
        """Tests that stale RUNNING jobs that exhausted max attempts are marked FAILED."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        stale_time = timezone.now() - timedelta(seconds=400)
        job = GenerationJob.objects.create(
            subject_name="Exhausted Job 1",
            subject_code="EXH01",
            status=GenerationJob.Status.RUNNING,
            attempts=3,
            max_attempts=3,
            locked_at=stale_time,
            locked_by="dead-worker-456"
        )

        worker = WorkerCommand()
        worker.reap_abandoned_jobs(lease_duration_seconds=300)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("lease expired", job.error_message.lower())

    # 17. Mobile generation modal behavior and template rendering
    def test_mobile_generation_modal_template_rendering(self):
        """Tests that the modal contains the background progress bar, step badges, and error alert elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.content.decode("utf-8")

        self.assertIn('id="generateSubjectModal"', content)
        self.assertIn('id="aiGenerateForm"', content)
        self.assertIn('id="aiProgressContainer"', content)
        self.assertIn('id="aiProgressBar"', content)
        self.assertIn('id="aiModalErrorAlert"', content)
        self.assertIn('id="aiStepBadge"', content)

    # 18. Worker claiming assigns unique claim_token (CONC-01)
    def test_worker_claim_generates_unique_claim_token(self):
        """Tests that claiming a pending job generates a fresh UUID claim token."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        job = GenerationJob.objects.create(
            subject_name="Operating Systems",
            subject_code="BCS401",
            status=GenerationJob.Status.PENDING
        )
        self.assertIsNone(job.claim_token)

        worker = WorkerCommand()
        claimed = worker.claim_next_job()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.id, job.id)
        self.assertIsInstance(claimed.claim_token, uuid.UUID)
        self.assertEqual(claimed.status, GenerationJob.Status.RUNNING)

    # 19. Optimistic fencing prevents stale worker from finalizing (CONC-01)
    @patch("curriculum.management.commands.run_generation_worker.generate_curriculum_from_prompt")
    def test_fencing_stale_worker_cannot_finalize_completed_job(self, mock_generate):
        """Tests that a worker whose claim token was revoked cannot overwrite job state or finalize."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        token_a = uuid.uuid4()
        token_b = uuid.uuid4()

        job = GenerationJob.objects.create(
            subject_name="Stale Worker Test",
            subject_code="STALE01",
            status=GenerationJob.Status.RUNNING,
            claim_token=token_a,
            locked_at=timezone.now(),
            locked_by="worker-a"
        )

        dummy_subject = Subject.objects.create(name="Stale Subject", code="STALE01", slug="stale-subject")
        mock_generate.return_value = dummy_subject

        worker_a = WorkerCommand()
        worker_a.worker_id = "worker-a"

        # Simulate Worker B reclaiming the job while Worker A is in-flight
        def fake_generate(*args, **kwargs):
            # Token changed by worker B in database
            GenerationJob.objects.filter(id=job.id).update(
                claim_token=token_b,
                locked_by="worker-b"
            )
            return dummy_subject

        mock_generate.side_effect = fake_generate

        # Worker A attempts to execute and finalize with token_a
        worker_a.execute_job(job, heartbeat_interval=0)

        job.refresh_from_db()
        # Job must NOT be marked COMPLETED by Worker A; token_b must remain untouched
        self.assertEqual(job.claim_token, token_b)
        self.assertEqual(job.locked_by, "worker-b")
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)

    # 20. Optimistic fencing prevents stale worker from marking job FAILED (CONC-01)
    def test_fencing_stale_worker_cannot_mark_job_failed(self):
        """Tests that a stale worker cannot record FAILED status after its claim token has been revoked."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        token_a = uuid.uuid4()
        token_b = uuid.uuid4()

        job = GenerationJob.objects.create(
            subject_name="Stale Fail Test",
            subject_code="FAIL01",
            status=GenerationJob.Status.RUNNING,
            claim_token=token_b,  # Database already has token_b
            locked_at=timezone.now(),
            locked_by="worker-b"
        )

        worker_a = WorkerCommand()
        worker_a.worker_id = "worker-a"

        # Worker A attempts to record failure with old token_a
        worker_a._mark_job_failed(job.id, token_a, "Old failure message")

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)
        self.assertEqual(job.claim_token, token_b)
        self.assertNotEqual(job.error_message, "Old failure message")

    # 21. Lease reaper clears claim_token on requeue and fail (SHUT-01)
    def test_lease_reaper_clears_claim_token_on_requeue_and_fail(self):
        """Tests that reap_abandoned_jobs clears claim_token both when requeueing and when failing a job."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand

        stale_time = timezone.now() - timedelta(seconds=400)

        # Job 1: Requeued
        job_requeue = GenerationJob.objects.create(
            subject_name="Requeue Job",
            status=GenerationJob.Status.RUNNING,
            attempts=1,
            max_attempts=3,
            claim_token=uuid.uuid4(),
            locked_at=stale_time,
            locked_by="worker-stale"
        )

        # Job 2: Failed
        job_fail = GenerationJob.objects.create(
            subject_name="Fail Job",
            status=GenerationJob.Status.RUNNING,
            attempts=3,
            max_attempts=3,
            claim_token=uuid.uuid4(),
            locked_at=stale_time,
            locked_by="worker-stale"
        )

        worker = WorkerCommand()
        worker.reap_abandoned_jobs(lease_duration_seconds=300)

        job_requeue.refresh_from_db()
        self.assertEqual(job_requeue.status, GenerationJob.Status.PENDING)
        self.assertIsNone(job_requeue.claim_token)
        self.assertIsNone(job_requeue.locked_at)

        job_fail.refresh_from_db()
        self.assertEqual(job_fail.status, GenerationJob.Status.FAILED)
        self.assertIsNone(job_fail.claim_token)
        self.assertIsNone(job_fail.locked_at)

    # 22. Worker invalid configuration rejection (PHASE 6)
    def test_worker_invalid_configuration_rejected(self):
        """Tests that invalid parameter combinations (e.g. lease_duration < heartbeat_interval * 2) raise CommandError."""
        with self.assertRaises(CommandError):
            call_command("run_generation_worker", lease_duration=20, heartbeat_interval=30, once=True)

        with self.assertRaises(CommandError):
            call_command("run_generation_worker", poll_interval=0, once=True)

    # 23. Cleanup old generation jobs management command (PHASE 7)
    def test_cleanup_old_generation_jobs_command(self):
        """Tests that cleanup_old_generation_jobs safely purges only eligible finished records."""
        old_time = timezone.now() - timedelta(days=40)
        recent_time = timezone.now() - timedelta(days=5)

        # Old COMPLETED and FAILED (should be deleted)
        old_comp = GenerationJob.objects.create(
            subject_name="Old Completed",
            status=GenerationJob.Status.COMPLETED,
            finished_at=old_time
        )
        old_fail = GenerationJob.objects.create(
            subject_name="Old Failed",
            status=GenerationJob.Status.FAILED,
            finished_at=old_time
        )

        # Recent COMPLETED (should be kept)
        recent_comp = GenerationJob.objects.create(
            subject_name="Recent Completed",
            status=GenerationJob.Status.COMPLETED,
            finished_at=recent_time
        )

        # PENDING and RUNNING (must NEVER be deleted)
        pending_job = GenerationJob.objects.create(
            subject_name="Pending Job",
            status=GenerationJob.Status.PENDING,
            created_at=old_time
        )
        running_job = GenerationJob.objects.create(
            subject_name="Running Job",
            status=GenerationJob.Status.RUNNING,
            locked_at=old_time
        )

        # 1. Test Dry-Run: reports 2 found, deletes 0
        call_command("cleanup_old_generation_jobs", days=30, dry_run=True)
        self.assertTrue(GenerationJob.objects.filter(id=old_comp.id).exists())
        self.assertTrue(GenerationJob.objects.filter(id=old_fail.id).exists())

        # 2. Test Execution: deletes old_comp and old_fail only
        call_command("cleanup_old_generation_jobs", days=30)
        self.assertFalse(GenerationJob.objects.filter(id=old_comp.id).exists())
        self.assertFalse(GenerationJob.objects.filter(id=old_fail.id).exists())
        self.assertTrue(GenerationJob.objects.filter(id=recent_comp.id).exists())
        self.assertTrue(GenerationJob.objects.filter(id=pending_job.id).exists())
        self.assertTrue(GenerationJob.objects.filter(id=running_job.id).exists())


# =====================================================================
# Concurrent PostgreSQL Worker Testing (SKIP LOCKED)
# =====================================================================

class PostgresWorkerConcurrencyTests(TransactionTestCase):
    """
    Concurrency test suite using TransactionTestCase to test multi-connection
    PostgreSQL SELECT ... FOR UPDATE SKIP LOCKED claiming and heartbeat thread semantics.
    """
    def test_concurrent_worker_claiming_skip_locked(self):
        """Tests that two concurrent workers competing for one job resolve atomically without race conditions."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand
        import threading

        job = GenerationJob.objects.create(
            subject_name="Concurrency Subject",
            subject_code="CONC101",
            status=GenerationJob.Status.PENDING
        )

        claimed_results = []
        barrier = threading.Barrier(2)

        def worker_claim_target(worker_id):
            try:
                worker = WorkerCommand()
                worker.worker_id = worker_id
                barrier.wait()  # Synchronize threads to claim at the exact same instant
                res = worker.claim_next_job()
                if res:
                    claimed_results.append((worker_id, res.id, res.claim_token))
            finally:
                from django.db import connection
                connection.close()

        t1 = threading.Thread(target=worker_claim_target, args=("worker-thread-1",))
        t2 = threading.Thread(target=worker_claim_target, args=("worker-thread-2",))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly ONE worker must have successfully claimed the job
        self.assertEqual(len(claimed_results), 1)
        winner_worker, claimed_id, claimed_token = claimed_results[0]
        self.assertEqual(claimed_id, job.id)
        self.assertIsNotNone(claimed_token)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)
        self.assertEqual(job.locked_by, winner_worker)
        self.assertEqual(job.claim_token, claimed_token)

    def test_heartbeat_refreshes_active_job_lease(self):
        """Tests that JobHeartbeat actively advances locked_at during execution."""
        from curriculum.management.commands.run_generation_worker import JobHeartbeat
        import time

        claim_token = uuid.uuid4()
        past_time = timezone.now() - timedelta(seconds=20)
        job = GenerationJob.objects.create(
            subject_name="Heartbeat Test",
            subject_code="HBT01",
            status=GenerationJob.Status.RUNNING,
            claim_token=claim_token,
            locked_at=past_time
        )

        # Run heartbeat with short 0.05s interval
        with JobHeartbeat(job.id, claim_token, interval_seconds=0.05, worker_id="test-worker") as hbt:
            time.sleep(0.15)
            self.assertFalse(hbt.lost_event.is_set())

        job.refresh_from_db()
        self.assertGreater(job.locked_at, past_time)

    def test_heartbeat_flags_lost_event_when_token_revoked(self):
        """Tests that JobHeartbeat detects when a claim token is revoked in database and sets lost_event."""
        from curriculum.management.commands.run_generation_worker import JobHeartbeat
        import time

        token_a = uuid.uuid4()
        token_b = uuid.uuid4()

        job = GenerationJob.objects.create(
            subject_name="Heartbeat Revocation Test",
            subject_code="HBT02",
            status=GenerationJob.Status.RUNNING,
            claim_token=token_a,
            locked_at=timezone.now()
        )

        with JobHeartbeat(job.id, token_a, interval_seconds=0.05, worker_id="test-worker") as hbt:
            # Change token in database to simulate lease revocation
            GenerationJob.objects.filter(id=job.id).update(claim_token=token_b)
            is_lost = hbt.lost_event.wait(timeout=2.0)
            self.assertTrue(is_lost)

    @patch("curriculum.management.commands.run_generation_worker.generate_curriculum_from_prompt")
    def test_worker_heartbeat_survives_long_running_gemini_call(self, mock_generate):
        """Tests that an active heartbeat keeps the job lease alive during a long simulated Gemini call."""
        from curriculum.management.commands.run_generation_worker import Command as WorkerCommand
        import time

        dummy_subject = Subject.objects.create(name="Long Subject", code="LNG01", slug="long-subject")
        for i in range(1, 6):
            Module.objects.create(subject=dummy_subject, module_number=i, title=f"Module {i}")

        def slow_generate(*args, **kwargs):
            time.sleep(0.15)
            return dummy_subject

        mock_generate.side_effect = slow_generate

        job = GenerationJob.objects.create(
            subject_name="Long Running AI Call",
            subject_code="LNG01",
            status=GenerationJob.Status.RUNNING,
            claim_token=uuid.uuid4(),
            locked_at=timezone.now()
        )

        worker = WorkerCommand()
        worker.execute_job(job, heartbeat_interval=1)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        self.assertEqual(job.progress, 100)


# =====================================================================
# In-Process Background Worker Thread Tests
# =====================================================================

class InProcessWorkerThreadTests(TransactionTestCase):
    """
    Tests the managed in-process background worker thread lifecycle,
    idempotent start/stop, instant notification wake-up, fallback polling,
    and shutdown safety.
    """

    def setUp(self):
        super().setUp()
        stop_in_process_worker(timeout=2.0)

    def tearDown(self):
        stop_in_process_worker(timeout=3.0)
        from django.db import connection
        try:
            connection.close()
        except Exception:
            pass
        super().tearDown()

    def test_in_process_worker_start_and_stop_idempotent(self):
        """Tests that starting the worker spawns a daemon thread, multiple starts are idempotent, and stop is clean."""
        thread1 = start_in_process_worker()
        self.assertIsNotNone(thread1)
        self.assertTrue(thread1.is_alive())
        self.assertTrue(thread1.daemon)

        # Second call returns the exact same running thread without duplicating
        thread2 = start_in_process_worker()
        self.assertIs(thread1, thread2)

        # Stop worker thread
        stopped = stop_in_process_worker(timeout=5.0)
        self.assertTrue(stopped)
        self.assertFalse(thread1.is_alive())

        # Stopping again when already stopped returns True
        self.assertTrue(stop_in_process_worker(timeout=1.0))

    @patch("curriculum.worker.generate_curriculum_from_prompt")
    def test_in_process_worker_instant_notification_wakes_worker(self, mock_generate):
        """Tests that notify_new_job wakes the worker immediately to process an enqueued job."""
        import time
        dummy_subject = Subject.objects.create(name="Instant Woken Subject", code="INST01", slug="instant-woken-subject")
        for i in range(1, 6):
            Module.objects.create(subject=dummy_subject, module_number=i, title=f"Module {i}")
        mock_generate.return_value = dummy_subject

        job = GenerationJob.objects.create(
            subject_name="Instant Woken Subject",
            subject_code="INST01",
            status=GenerationJob.Status.PENDING
        )

        # Use an engine with a long poll_interval (10s) to prove the wake-up is instant from the event
        engine = GenerationWorkerEngine(poll_interval=10.0, lease_duration=60, heartbeat_interval=30)
        start_in_process_worker(engine=engine)

        # Send instant notification
        notify_new_job()

        # Wait up to 3s for thread to complete the job
        completed = False
        for _ in range(30):
            job.refresh_from_db()
            if job.status == GenerationJob.Status.COMPLETED:
                completed = True
                break
            time.sleep(0.1)

        self.assertTrue(completed, "Worker did not process job promptly via notify_new_job event.")
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.step, 4)

    @patch("curriculum.worker.generate_curriculum_from_prompt")
    def test_in_process_worker_fallback_polling_without_notification(self, mock_generate):
        """Tests that if no notification event is received, periodic polling still discovers and processes pending jobs."""
        import time
        dummy_subject = Subject.objects.create(name="Polling Subject", code="POLL01", slug="polling-subject")
        for i in range(1, 6):
            Module.objects.create(subject=dummy_subject, module_number=i, title=f"Module {i}")
        mock_generate.return_value = dummy_subject

        job = GenerationJob.objects.create(
            subject_name="Polling Subject",
            subject_code="POLL01",
            status=GenerationJob.Status.PENDING
        )

        # Short 0.2s poll interval, DO NOT call notify_new_job()
        engine = GenerationWorkerEngine(poll_interval=0.2, lease_duration=60, heartbeat_interval=30)
        start_in_process_worker(engine=engine)

        completed = False
        for _ in range(30):
            job.refresh_from_db()
            if job.status == GenerationJob.Status.COMPLETED:
                completed = True
                break
            time.sleep(0.1)

        self.assertTrue(completed, "Worker fallback polling failed to pick up pending job.")

    def test_in_process_worker_shutdown_stops_claiming_new_jobs(self):
        """Tests that stopping the worker prevents it from claiming subsequent jobs."""
        import time
        engine = GenerationWorkerEngine(poll_interval=0.1)
        start_in_process_worker(engine=engine)

        stopped = stop_in_process_worker(timeout=3.0)
        self.assertTrue(stopped)

        # Create a pending job after shutdown
        job = GenerationJob.objects.create(
            subject_name="After Shutdown",
            subject_code="AFT01",
            status=GenerationJob.Status.PENDING
        )
        notify_new_job()
        time.sleep(0.3)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.PENDING)

    def test_interrupted_running_job_recovered_by_lease_expiration(self):
        """Tests that a job left in RUNNING due to sudden process termination is recovered by lease expiration."""
        past_time = timezone.now() - timedelta(seconds=400)
        stale_job = GenerationJob.objects.create(
            subject_name="Crash Recovered Job",
            subject_code="CRASH01",
            status=GenerationJob.Status.RUNNING,
            attempts=1,
            max_attempts=3,
            claim_token=uuid.uuid4(),
            locked_at=past_time,
            locked_by="dead-web-dyno"
        )

        engine = GenerationWorkerEngine(lease_duration=300, max_attempts=3)
        reaped = engine.reap_abandoned_jobs()
        self.assertEqual(reaped, 1)

        stale_job.refresh_from_db()
        self.assertEqual(stale_job.status, GenerationJob.Status.PENDING)
        self.assertIsNone(stale_job.claim_token)
        self.assertIsNone(stale_job.locked_at)
        self.assertEqual(stale_job.locked_by, "")
        self.assertIn("requeued", stale_job.progress_message)


class ProductionHardeningFencingAndHealthTests(TestCase):
    """
    Verification tests for Critical Production Fixes:
    - Atomicity & Fencing: decoupling generation from DB persistence.
    - Stale-worker curriculum rejection: revoked claim tokens and expired leases fail atomically.
    - Transaction rollback: zero records created on fencing rejection.
    - Worker startup failure propagation in gunicorn.conf.py.
    - Worker health check & supervisor revival in worker.py.
    - Rejection with HTTP 503 when worker is offline (no orphaned jobs).
    """

    def setUp(self):
        self.client = Client()

    def tearDown(self):
        stop_in_process_worker(timeout=2.0)

    def test_persist_curriculum_records_with_valid_claim(self):
        """Verifies curriculum persistence succeeds and updates job to COMPLETED under valid claim."""
        claim_token = uuid.uuid4()
        job = GenerationJob.objects.create(
            subject_name="Software Engineering",
            subject_code="BCS501_VALID",
            status=GenerationJob.Status.RUNNING,
            claim_token=claim_token,
            locked_at=timezone.now(),
            locked_by="worker-valid"
        )

        payload = {
            "data": SAMPLE_CURRICULUM_DATA,
            "clean_name": "Software Engineering",
            "clean_code": "BCS501_VALID",
            "is_verified_source": True,
            "active_model": "gemini-3.5-flash",
        }

        subject = persist_curriculum_records(
            curriculum_payload=payload,
            job_id=job.id,
            claim_token=claim_token,
            lease_duration=300
        )

        self.assertIsNotNone(subject)
        self.assertEqual(subject.name, "Software Engineering")
        self.assertEqual(subject.modules.count(), 5)
        self.assertEqual(Topic.objects.filter(module__subject=subject).count(), 15)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.COMPLETED)
        self.assertEqual(job.step, 4)
        self.assertEqual(job.progress, 100)
        self.assertEqual(job.subject, subject)

    def test_persist_curriculum_records_rejects_revoked_claim_token(self):
        """Verifies CurriculumFencingError is raised and 0 records created when claim token is mismatched/revoked."""
        initial_token = uuid.uuid4()
        revoked_token = uuid.uuid4()

        job = GenerationJob.objects.create(
            subject_name="Cyber Threat Intel",
            subject_code="CTI901",
            status=GenerationJob.Status.RUNNING,
            claim_token=revoked_token,  # Another worker re-claimed or updated the token
            locked_at=timezone.now(),
            locked_by="worker-new"
        )

        payload = {
            "data": SAMPLE_CURRICULUM_DATA,
            "clean_name": "Cyber Threat Intel",
            "clean_code": "CTI901",
            "is_verified_source": False,
            "active_model": "gemini-3.5-flash",
        }

        # Attempt to persist using the old initial_token
        with self.assertRaises(CurriculumFencingError):
            persist_curriculum_records(
                curriculum_payload=payload,
                job_id=job.id,
                claim_token=initial_token,
                lease_duration=300
            )

        # Database atomicity check: NO Subject, Module, or Topic should have been committed
        self.assertFalse(Subject.objects.filter(code="CTI901").exists())
        self.assertFalse(Module.objects.filter(title__contains="CTI901").exists())

        # Job must NOT be marked COMPLETED or FAILED by stale worker
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)
        self.assertEqual(job.claim_token, revoked_token)

    def test_persist_curriculum_records_rejects_expired_lease(self):
        """Verifies CurriculumFencingError is raised and rolled back when worker lease has expired."""
        claim_token = uuid.uuid4()
        past_time = timezone.now() - timedelta(seconds=500)

        job = GenerationJob.objects.create(
            subject_name="Quantum Computing",
            subject_code="QC999",
            status=GenerationJob.Status.RUNNING,
            claim_token=claim_token,
            locked_at=past_time,
            locked_by="worker-expired"
        )

        payload = {
            "data": SAMPLE_CURRICULUM_DATA,
            "clean_name": "Quantum Computing",
            "clean_code": "QC999",
            "is_verified_source": False,
            "active_model": "gemini-3.5-flash",
        }

        with self.assertRaises(CurriculumFencingError):
            persist_curriculum_records(
                curriculum_payload=payload,
                job_id=job.id,
                claim_token=claim_token,
                lease_duration=300
            )

        self.assertFalse(Subject.objects.filter(code="QC999").exists())
        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)

    def test_execute_job_catches_curriculum_fencing_error_gracefully(self):
        """Verifies execute_job discards output on CurriculumFencingError without overwriting job state."""
        job = GenerationJob.objects.create(
            subject_name="Fencing Test",
            subject_code="FNC101",
            status=GenerationJob.Status.RUNNING,
            claim_token=uuid.uuid4(),
            locked_at=timezone.now(),
            locked_by="worker-fencing"
        )

        def mock_failing_generator(**kwargs):
            raise CurriculumFencingError("Simulated fencing failure: lease expired.")

        engine = GenerationWorkerEngine()
        # Should catch CurriculumFencingError and return cleanly without crashing
        engine.execute_job(job, heartbeat_interval=0, generator_func=mock_failing_generator)

        job.refresh_from_db()
        # The job was not marked FAILED because worker recognized it no longer owned the lease
        self.assertEqual(job.status, GenerationJob.Status.RUNNING)

    def test_worker_offline_returns_503_without_creating_job(self):
        """Verifies generate_subject_ai returns HTTP 503 and creates no GenerationJob when worker is offline."""
        with override_settings(SIMULATE_WORKER_OFFLINE=True):
            resp = self.client.post(
                reverse("curriculum:generate_subject_ai"),
                {"subject_name": "Compiler Design", "subject_code": "CD601"},
                HTTP_X_REQUESTED_WITH="XMLHttpRequest"
            )
            self.assertEqual(resp.status_code, 503)
            data = resp.json()
            self.assertFalse(data.get("success"))
            self.assertEqual(data.get("status"), "FAILED")
            self.assertEqual(
                data.get("error"),
                "Curriculum generation is temporarily unavailable. Please try again later."
            )
            # Ensure zero jobs were created in PostgreSQL
            self.assertFalse(GenerationJob.objects.filter(subject_code="CD601").exists())

    def test_worker_supervisor_revives_dead_thread(self):
        """Verifies that is_worker_accepting_jobs automatically revives an unexpectedly killed worker thread."""
        stop_in_process_worker(timeout=2.0)
        thread = start_in_process_worker()
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive())

        # Mock thread being dead
        with patch.object(thread, "is_alive", return_value=False):
            # Supervisor should detect dead thread and trigger restart
            is_healthy = is_worker_accepting_jobs()
            self.assertTrue(is_healthy)
            revived = get_worker_thread()
            self.assertIsNotNone(revived)
            self.assertTrue(revived.is_alive())

        stop_in_process_worker(timeout=2.0)

    def test_gunicorn_post_worker_init_propagates_startup_failure(self):
        """Verifies that gunicorn.conf.py post_worker_init propagates exceptions to fail worker boot."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("gunicorn_conf_test", "gunicorn.conf.py")
        gconf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gconf)

        mock_worker = MagicMock()
        mock_worker.pid = 99999

        with patch("curriculum.worker.start_in_process_worker", side_effect=Exception("Database connection timeout")):
            with self.assertRaises(RuntimeError) as ctx:
                gconf.post_worker_init(mock_worker)
            self.assertIn("Gunicorn worker (pid=99999) boot failed", str(ctx.exception))


class ChapterHierarchyTests(TestCase):
    """
    Test suite for Approach A (Chapter Hierarchy inside Modules):
    - Topic model chapter fields (chapter_number, chapter_title) and defaults.
    - Module.get_chapters() grouping and progress calculation.
    - Extraction of chapters and topics from VTU syllabus text (matching user screenshot).
    - Curriculum persistence with chapter fields.
    - View rendering with chapter banners and badges.
    """

    def setUp(self):
        self.client = Client()

    def test_topic_chapter_fields_defaults_and_assignment(self):
        """Tests that Topic chapter fields default gracefully and store custom chapter metadata."""
        subject = Subject.objects.create(name="Computer Networks", code="CS501", slug="cs501-cn")
        module = Module.objects.create(subject=subject, module_number=1, title="Introduction")

        # Default topic has chapter_number=1 and empty chapter_title
        t1 = Topic.objects.create(module=module, title="OSI Model", slug="osi-model", order=1)
        self.assertEqual(t1.chapter_number, 1)
        self.assertEqual(t1.chapter_title, "")

        # Custom topic with specific chapter
        t2 = Topic.objects.create(
            module=module,
            title="TCP/IP Protocol Suite",
            slug="tcp-ip-suite",
            order=2,
            chapter_number=2,
            chapter_title="Protocol Stacks"
        )
        self.assertEqual(t2.chapter_number, 2)
        self.assertEqual(t2.chapter_title, "Protocol Stacks")

    def test_module_get_chapters_groups_topics_correctly(self):
        """Tests that module.get_chapters() groups topics into distinct chapter dictionaries with progress."""
        subject = Subject.objects.create(name="Cyber Security", code="CS701", slug="cs701-sec")
        module = Module.objects.create(subject=subject, module_number=3, title="Key Management")

        # Chapter 1: 3 topics
        for i in range(1, 4):
            Topic.objects.create(
                module=module,
                title=f"Key Mgmt Topic {i}",
                slug=f"km-topic-{i}",
                order=i,
                chapter_number=1,
                chapter_title="Key Management Fundamentals"
            )

        # Chapter 2: 2 topics
        for j in range(1, 3):
            Topic.objects.create(
                module=module,
                title=f"Public Key Topic {j}",
                slug=f"pk-topic-{j}",
                order=j,
                chapter_number=2,
                chapter_title="Public-Key Management"
            )

        self.assertTrue(module.has_multiple_chapters)
        chapters = module.get_chapters()
        self.assertEqual(len(chapters), 2)

        # Validate Chapter 1
        chp1 = chapters[0]
        self.assertEqual(chp1["chapter_number"], 1)
        self.assertEqual(chp1["chapter_title"], "Key Management Fundamentals")
        self.assertEqual(len(chp1["topics"]), 3)
        self.assertEqual(chp1["total_topics"], 3)

        # Validate Chapter 2
        chp2 = chapters[1]
        self.assertEqual(chp2["chapter_number"], 2)
        self.assertEqual(chp2["chapter_title"], "Public-Key Management")
        self.assertEqual(len(chp2["topics"]), 2)
        self.assertEqual(chp2["total_topics"], 2)

    def test_extract_chapters_from_vtu_syllabus_text(self):
        """Tests extraction of chapters from the exact syllabus format shown in the user's screenshot."""
        raw_syllabus_text = """CHP-1
Key management fundamentals, Key lengths and lifetimes, Key generation, Key establishment, Key storage, Key usage, Governing key management.

CHP-2
Public-Key Management: Certification of public keys, The certificate lifecycle, Public-key management models, Alternative approaches."""

        chapters = _extract_chapters_from_text(raw_syllabus_text)
        self.assertEqual(len(chapters), 2)

        chp1 = chapters[0]
        self.assertEqual(chp1["chapter_number"], 1)
        self.assertEqual(chp1["chapter_title"], "Key management fundamentals")
        self.assertEqual(len(chp1["topics"]), 6)
        self.assertIn("Key lengths and lifetimes", chp1["topics"])
        self.assertIn("Governing key management", chp1["topics"])

        chp2 = chapters[1]
        self.assertEqual(chp2["chapter_number"], 2)
        self.assertEqual(chp2["chapter_title"], "Public-Key Management")
        self.assertEqual(len(chp2["topics"]), 4)
        self.assertIn("Certification of public keys", chp2["topics"])
        self.assertIn("Alternative approaches", chp2["topics"])

    def test_persist_curriculum_records_with_chapters(self):
        """Tests that persist_curriculum_records correctly saves chapter metadata onto Topic records."""
        payload = {
            "data": {
                "name": "Data Security Test",
                "code": "DST801",
                "description": "Course with chapters",
                "modules": [
                    {
                        "module_number": 1,
                        "title": "Crypto Foundations",
                        "description": "Module 1",
                        "topics": [
                            {
                                "title": "Classical Ciphers",
                                "description": "Symmetric ciphers",
                                "chapter_number": 1,
                                "chapter_title": "Classical Cryptography",
                                "keywords": ["classical ciphers"]
                            },
                            {
                                "title": "Block Ciphers",
                                "description": "DES & AES",
                                "chapter_number": 2,
                                "chapter_title": "Modern Block Ciphers",
                                "keywords": ["block ciphers"]
                            }
                        ]
                    },
                    {
                        "module_number": 2, "title": "M2", "description": "",
                        "topics": [{"title": "T2.1", "description": "D", "keywords": ["k"]}]
                    },
                    {
                        "module_number": 3, "title": "M3", "description": "",
                        "topics": [{"title": "T3.1", "description": "D", "keywords": ["k"]}]
                    },
                    {
                        "module_number": 4, "title": "M4", "description": "",
                        "topics": [{"title": "T4.1", "description": "D", "keywords": ["k"]}]
                    },
                    {
                        "module_number": 5, "title": "M5", "description": "",
                        "topics": [{"title": "T5.1", "description": "D", "keywords": ["k"]}]
                    }
                ]
            },
            "clean_name": "Data Security Test",
            "clean_code": "DST801",
            "is_verified_source": True,
            "active_model": "gemini-3.5-flash",
        }

        subject = persist_curriculum_records(curriculum_payload=payload)
        m1 = subject.modules.get(module_number=1)
        t_classical = m1.topics.get(title="Classical Ciphers")
        self.assertEqual(t_classical.chapter_number, 1)
        self.assertEqual(t_classical.chapter_title, "Classical Cryptography")

        t_block = m1.topics.get(title="Block Ciphers")
        self.assertEqual(t_block.chapter_number, 2)
        self.assertEqual(t_block.chapter_title, "Modern Block Ciphers")

    def test_subject_detail_view_renders_chapters(self):
        """Tests that subject_detail view renders chapter banners and chapter map when chapters exist."""
        subject = Subject.objects.create(name="Network Security View Test", code="NSV901", slug="nsv901-test")
        module = Module.objects.create(subject=subject, module_number=1, title="Key Management")
        t1 = Topic.objects.create(
            module=module,
            title="Key Lengths",
            slug="key-lengths-test",
            order=1,
            chapter_number=1,
            chapter_title="Key Management Fundamentals"
        )
        t2 = Topic.objects.create(
            module=module,
            title="Public Key Certificates",
            slug="pk-certs-test",
            order=2,
            chapter_number=2,
            chapter_title="Public-Key Management"
        )

        resp = self.client.get(reverse("curriculum:subject_detail", kwargs={"slug": subject.slug}))
        self.assertEqual(resp.status_code, 200)
        content = resp.content.decode("utf-8")
        self.assertIn("CHP-1", content)
        self.assertIn("Key Management Fundamentals", content)
        self.assertIn("CHP-2", content)
        self.assertIn("Public-Key Management", content)
        self.assertIn("Chapter Architecture", content)


class GroqIntegrationTests(TestCase):
    """
    Tests for Groq API curriculum generation, provider resolution,
    retry/fallback mechanics, and key redaction.
    """

    def test_sanitizer_redacts_groq_api_keys(self):
        """Verifies that sanitize_error_message scrubs Groq api keys (gsk_...) and Bearer tokens."""
        raw_msg = "Failed with gsk_abcdef1234567890abcdef1234567890 and Bearer gsk_9876543210token"
        sanitized = sanitize_error_message(raw_msg)
        self.assertNotIn("gsk_abcdef", sanitized)
        self.assertNotIn("gsk_987654", sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)
        self.assertIn("Bearer [REDACTED]", sanitized)

    def test_provider_resolution_prefers_groq(self):
        """Tests that auto mode selects Groq when GROQ_API_KEY is present."""
        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_test_groq_key'
            if key == 'GEMINI_API_KEY':
                return 'AIzaSy_test_gemini_key'
            if key == 'AI_PROVIDER':
                return 'auto'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            provider, api_key = resolve_ai_provider()
            self.assertEqual(provider, 'groq')
            self.assertEqual(api_key, 'gsk_test_groq_key')

    def test_provider_resolution_falls_back_to_gemini(self):
        """Tests that auto mode selects Gemini when only GEMINI_API_KEY is present."""
        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return ''
            if key == 'GEMINI_API_KEY':
                return 'AIzaSy_test_gemini_key'
            if key == 'AI_PROVIDER':
                return 'auto'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            provider, api_key = resolve_ai_provider()
            self.assertEqual(provider, 'gemini')
            self.assertEqual(api_key, 'AIzaSy_test_gemini_key')

    def test_provider_resolution_raises_when_no_keys_configured(self):
        """Tests that AIAuthError is raised when neither key is configured."""
        def mock_config(key, default='', **kwargs):
            return ''

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            with self.assertRaises(AIAuthError) as ctx:
                resolve_ai_provider()
            self.assertIn("GROQ_API_KEY is not configured", str(ctx.exception))

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_successful_generation(self, mock_post):
        """Tests end-to-end curriculum generation using Groq API and openai/gpt-oss-120b."""
        mock_post.return_value = make_mock_groq_json_response(SAMPLE_CURRICULUM_DATA, status_code=200)

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_mock_valid_key_12345'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'openai/gpt-oss-120b'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'openai/gpt-oss-20b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            result = generate_curriculum_data("Machine Learning", "CS702")

            self.assertEqual(result["active_model"], "openai/gpt-oss-120b")
            self.assertEqual(len(result["data"]["modules"]), 5)
            self.assertEqual(result["clean_name"], "Machine Learning")

            # Check that request went to Groq OpenAI-compatible endpoint
            self.assertTrue(mock_post.called)
            call_url = mock_post.call_args[0][0]
            self.assertEqual(call_url, "https://api.groq.com/openai/v1/chat/completions")
            call_headers = mock_post.call_args[1]["headers"]
            self.assertEqual(call_headers["Authorization"], "Bearer gsk_mock_valid_key_12345")
            call_payload = mock_post.call_args[1]["json"]
            self.assertEqual(call_payload["model"], "openai/gpt-oss-120b")

    def test_groq_decommissioned_model_auto_redirect(self):
        """Tests that decommissioned llama-3.3-70b-versatile is automatically upgraded to openai/gpt-oss-120b."""
        normalized = normalize_groq_model_name("llama-3.3-70b-versatile")
        self.assertEqual(normalized, "openai/gpt-oss-120b")
        normalized_shorthand = normalize_groq_model_name("llama-3.3-70b")
        self.assertEqual(normalized_shorthand, "openai/gpt-oss-120b")
        default_model = normalize_groq_model_name("")
        self.assertEqual(default_model, "openai/gpt-oss-120b")

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_authentication_failure(self, mock_post):
        """Tests that HTTP 401 from Groq immediately raises AIAuthError without retries."""
        err_res = requests.Response()
        err_res.status_code = 401
        err_res._content = json.dumps({"error": {"message": "Invalid API Key"}}).encode("utf-8")
        mock_post.return_value = err_res

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_bad_key'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'openai/gpt-oss-120b'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'openai/gpt-oss-20b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            with self.assertRaises(AIAuthError):
                generate_curriculum_data("Machine Learning", "CS702")
            # Should fail fast without retry loops
            self.assertEqual(mock_post.call_count, 1)

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_permission_failure_403(self, mock_post):
        """Tests that HTTP 403 from Groq raises AIPermissionError."""
        err_res = requests.Response()
        err_res.status_code = 403
        err_res._content = json.dumps({"error": {"message": "Model access restricted for this tier"}}).encode("utf-8")
        mock_post.return_value = err_res

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_restricted_key'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'openai/gpt-oss-120b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            with self.assertRaises(AIPermissionError):
                generate_curriculum_data("Machine Learning", "CS702")
            self.assertEqual(mock_post.call_count, 1)

    @patch('curriculum.services.ai_curriculum.requests.get')
    def test_groq_list_models_auth_failure(self, mock_get):
        """Tests that HTTP 401 from Groq ListModels immediately raises AIAuthError."""
        err_res = requests.Response()
        err_res.status_code = 401
        err_res._content = json.dumps({"error": {"message": "Invalid API Key"}}).encode("utf-8")
        mock_get.return_value = err_res

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_bad_key'
            if key == 'AI_PROVIDER':
                return 'groq'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            with self.assertRaises(AIAuthError):
                generate_curriculum_data("Machine Learning", "CS702")

    @patch('curriculum.services.ai_curriculum.time.sleep')
    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_transient_retry_and_fallback(self, mock_post, mock_sleep):
        """Tests that transient 429 from Groq triggers retry and falls back to secondary model."""
        rate_limit_res = requests.Response()
        rate_limit_res.status_code = 429
        rate_limit_res._content = json.dumps({"error": {"message": "Rate limit reached"}}).encode("utf-8")

        success_res = make_mock_groq_json_response(SAMPLE_CURRICULUM_DATA, status_code=200)

        # Primary model exhausts 1 retry, fallback model succeeds on 1st attempt
        mock_post.side_effect = [rate_limit_res, rate_limit_res, success_res]

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_mock_valid_key'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key in ('GROQ_MAX_RETRIES', 'GEMINI_MAX_RETRIES'):
                return 1
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'openai/gpt-oss-120b'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'openai/gpt-oss-20b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            result = generate_curriculum_data("Machine Learning", "CS702")
            self.assertEqual(result["active_model"], "openai/gpt-oss-20b")
            self.assertEqual(len(result["data"]["modules"]), 5)

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_model_404_falls_back_to_secondary_model(self, mock_post):
        """Tests that if primary model returns 404, Groq automatically falls back to secondary model."""
        res_404 = requests.Response()
        res_404.status_code = 404
        res_404._content = json.dumps({"error": {"message": "Model not found"}}).encode("utf-8")

        res_200 = make_mock_groq_json_response(SAMPLE_CURRICULUM_DATA, status_code=200)

        # Primary model returns 404, fallback model succeeds
        mock_post.side_effect = [res_404, res_200]

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_mock_valid_key'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'non-existent-primary'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'openai/gpt-oss-20b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            result = generate_curriculum_data("Machine Learning", "CS702")
            self.assertEqual(result["active_model"], "openai/gpt-oss-20b")
            self.assertEqual(len(result["data"]["modules"]), 5)

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_minimal_openai_gpt_oss_120b_chat_completion(self, mock_post):
        """Minimal test verifying openai/gpt-oss-120b request payload structure and response processing."""
        mock_post.return_value = make_mock_groq_json_response(
            {"response": "Python is a high-level programming language known for readability. It supports multiple paradigms."},
            status_code=200
        )
        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_test_key_123'
            if key in ('GROQ_MODEL', 'GROQ_PRIMARY_MODEL'):
                return 'openai/gpt-oss-120b'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            from curriculum.services.ai_curriculum import execute_groq_generation
            res, model_used = execute_groq_generation(
                system_instructions="You are a helpful assistant.",
                user_prompt="Explain what Python is in two sentences.",
                api_key="gsk_test_key_123",
                primary_model="openai/gpt-oss-120b"
            )
            self.assertEqual(model_used, "openai/gpt-oss-120b")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            content = json.loads(data["choices"][0]["message"]["content"])
            self.assertIn("Python is a high-level programming language", content["response"])


class ModuleNoteStreamingTests(TestCase):
    """
    Tests for internal streaming and downloading of module notes.
    Verifies that Cloudinary/remote URLs are proxied and masked under clean website routes
    with branded academic filenames and appropriate MIME/cache headers.
    """

    def setUp(self):
        self.client = Client()
        self.subject = Subject.objects.create(
            name="Advanced Algorithms",
            code="CS701",
            slug="cs701"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Dynamic Programming"
        )

    @patch('curriculum.views.requests.get')
    def test_view_module_note_remote_stream_inline(self, mock_get):
        """Verifies remote Cloudinary note is streamed inline with masked branding and clean filename."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '24'}
        mock_response.iter_content = MagicMock(return_value=[b"%PDF-1.4 ", b"Sample Note Data"])
        mock_get.return_value = mock_response

        note = ModuleNote.objects.create(
            module=self.module,
            title="Module 1 Notes",
            file="https://res.cloudinary.com/testcloud/raw/upload/v1234/test_notes.pdf"
        )

        url = reverse('curriculum:view_module_note', args=[note.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertEqual(response['Content-Disposition'], 'inline; filename="CS701_module-1-notes.pdf"')
        self.assertEqual(response['Cache-Control'], 'public, max-age=86400')
        mock_get.assert_called_once_with(
            "https://res.cloudinary.com/testcloud/raw/upload/v1234/test_notes.pdf",
            stream=True,
            timeout=20
        )
        content = b"".join(response.streaming_content)
        self.assertEqual(content, b"%PDF-1.4 Sample Note Data")

    @patch('curriculum.views.requests.get')
    def test_download_module_note_remote_stream_attachment(self, mock_get):
        """Verifies download note route forces attachment disposition with branded filename."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Length': '24'}
        mock_response.iter_content = MagicMock(return_value=[b"%PDF-1.4 ", b"Download Data"])
        mock_get.return_value = mock_response

        note = ModuleNote.objects.create(
            module=self.module,
            title="Handwritten Formula Sheet",
            file="https://res.cloudinary.com/testcloud/raw/upload/v1234/sheet.pdf"
        )

        url = reverse('curriculum:download_module_note', args=[note.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Disposition'], 'attachment; filename="CS701_handwritten-formula-sheet.pdf"')
        content = b"".join(response.streaming_content)
        self.assertEqual(content, b"%PDF-1.4 Download Data")

    def test_view_module_note_local_file(self):
        """Verifies local file storage fallback works seamlessly with FileResponse."""
        dummy_file = SimpleUploadedFile("dp_cheatsheet.pdf", b"%PDF-1.4 Local Content", content_type="application/pdf")
        note = ModuleNote.objects.create(
            module=self.module,
            title="DP Cheatsheet",
            file=dummy_file
        )

        url = reverse('curriculum:view_module_note', args=[note.id])
        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('inline; filename="CS701_dp-cheatsheet.pdf"', response['Content-Disposition'])

        download_url = reverse('curriculum:download_module_note', args=[note.id])
        dl_response = self.client.get(download_url)
        self.assertEqual(dl_response.status_code, 200)
        self.assertIn('attachment; filename="CS701_dp-cheatsheet.pdf"', dl_response['Content-Disposition'])

    def test_view_module_note_nonexistent_returns_404(self):
        """Requesting a nonexistent note returns HTTP 404."""
        url = reverse('curriculum:view_module_note', args=[999999])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)

    @patch('curriculum.views.requests.get')
    def test_view_module_note_remote_error_returns_404(self, mock_get):
        """If remote Cloudinary fetch fails (e.g. 404 or connection error), view returns 404."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        note = ModuleNote.objects.create(
            module=self.module,
            title="Missing Remote Note",
            file="https://res.cloudinary.com/testcloud/raw/upload/v1234/missing.pdf"
        )

        url = reverse('curriculum:view_module_note', args=[note.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 404)


class HybridDeskAndCatalogTests(TestCase):
    """
    Tests for the Hybrid Academic model:
    - Auto-inferencing branch and semester from VTU and university course codes.
    - Personal study desk pinning and unpinning without accounts or login.
    - University catalog filtering context and cookie synchronization.
    """

    def setUp(self):
        self.client = Client()

    def test_infer_branch_and_semester_vtu_codes(self):
        """Verifies branch and semester are accurately deduced from common VTU codes."""
        b, s = Subject.infer_branch_and_semester("BAI701", "Deep Learning")
        self.assertEqual(b, Subject.Branch.AIML)
        self.assertEqual(s, 7)

        b, s = Subject.infer_branch_and_semester("BCS502", "Database Management Systems")
        self.assertEqual(b, Subject.Branch.CSE)
        self.assertEqual(s, 5)

        b, s = Subject.infer_branch_and_semester("BME755A", "Non-Traditional Machining")
        self.assertEqual(b, Subject.Branch.ME)
        self.assertEqual(s, 7)

        b, s = Subject.infer_branch_and_semester("18CS71", "Web Technology and its applications")
        self.assertEqual(b, Subject.Branch.CSE)
        self.assertEqual(s, 7)

        b, s = Subject.infer_branch_and_semester("BMAT101", "Calculus and Linear Algebra")
        self.assertEqual(b, Subject.Branch.ALL)
        self.assertEqual(s, 1)

    def test_subject_save_auto_infers_branch_and_semester(self):
        """When not provided, saving a Subject auto-infers branch and semester."""
        s = Subject.objects.create(
            name="Deep Learning",
            code="BAI701",
            slug="dl-auto"
        )
        self.assertEqual(s.branch, Subject.Branch.AIML)
        self.assertEqual(s.semester, 7)
        self.assertEqual(s.academic_tag, "AIML · Sem 7")

    def test_subject_save_honors_explicit_branch_and_semester(self):
        """Explicitly specified branch and semester are not overridden."""
        s = Subject.objects.create(
            name="Custom Course",
            code="GEN999",
            slug="custom-course",
            branch=Subject.Branch.ECE,
            semester=3
        )
        self.assertEqual(s.branch, Subject.Branch.ECE)
        self.assertEqual(s.semester, 3)
        self.assertEqual(s.academic_tag, "ECE · Sem 3")

    def test_toggle_subject_pin_flow(self):
        """Verifies pinning and unpinning a subject via AJAX without an account."""
        sub = Subject.objects.create(
            name="Operating Systems",
            code="BCS403",
            slug="operating-systems",
            branch=Subject.Branch.CSE,
            semester=4
        )
        url = reverse('curriculum:toggle_subject_pin')

        # 1. Pin
        res1 = self.client.post(url, data={'slug': sub.slug})
        self.assertEqual(res1.status_code, 200)
        data1 = res1.json()
        self.assertTrue(data1['success'])
        self.assertTrue(data1['is_pinned'])
        self.assertEqual(data1['total_pinned'], 1)
        self.assertIn(sub.slug, res1.cookies['studyhub_pinned'].value)

        # 2. Unpin
        res2 = self.client.post(url, data={'slug': sub.slug})
        self.assertEqual(res2.status_code, 200)
        data2 = res2.json()
        self.assertTrue(data2['success'])
        self.assertFalse(data2['is_pinned'])
        self.assertEqual(data2['total_pinned'], 0)

    def test_toggle_subject_pin_validation(self):
        """Verifies validation and HTTP error codes for toggle pin."""
        url = reverse('curriculum:toggle_subject_pin')

        # Method not allowed (GET)
        get_res = self.client.get(url)
        self.assertEqual(get_res.status_code, 405)

        # Missing slug
        missing_res = self.client.post(url, data={})
        self.assertEqual(missing_res.status_code, 400)

        # Nonexistent subject
        not_found_res = self.client.post(url, data={'slug': 'non-existent-subject'})
        self.assertEqual(not_found_res.status_code, 404)

    def test_home_view_hybrid_desk_context(self):
        """Verifies home view renders pinned subjects from cookie and passes branch/sem choices."""
        s1 = Subject.objects.create(name="Networks", code="BCS501", slug="networks-test", branch=Subject.Branch.CSE, semester=5)
        s2 = Subject.objects.create(name="CAD", code="BME501", slug="cad-test", branch=Subject.Branch.ME, semester=5)

        # Set cookie with s1 pinned
        self.client.cookies['studyhub_pinned'] = s1.slug
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)

        # Check context
        self.assertEqual(res.context['pinned_count'], 1)
        self.assertEqual(len(res.context['pinned_subjects']), 1)
        self.assertEqual(res.context['pinned_subjects'][0].slug, s1.slug)
        self.assertTrue(any(code == 'CSE' for code, label in res.context['branches']))
        self.assertIn(5, res.context['semesters'])

        # Check HTML contains tabs and cards
        html = res.content.decode('utf-8')
        self.assertIn('My Desk', html)
        self.assertIn('University Catalog', html)
        self.assertIn('networks-test', html)
        self.assertIn('cad-test', html)


class VTUExamBankAndBlueprintTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Cryptography and Network Security",
            code="BCS701",
            slug="crypto-security",
            branch=Subject.Branch.CSE,
            semester=7,
            scheme="2022 Scheme"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Symmetric Ciphers and Classical Encryption",
            description="Classical encryption techniques and DES"
        )
        self.topic = Topic.objects.create(
            module=self.module,
            title="Substitution and Transposition Techniques",
            slug="substitution-transposition",
            order=1
        )

    def test_exam_question_model_properties(self):
        """Tests ExamQuestion model fields and is_repeated property."""
        q1 = ExamQuestion.objects.create(
            module=self.module,
            topic=self.topic,
            question_text="Explain Feistel cipher structure with neat diagram. [10M]",
            marks=10,
            frequency=3,
            years_asked="Jan 2023, Jul 2022, Dec 2020",
            scheme="2022 Scheme"
        )
        self.assertTrue(q1.is_repeated)
        self.assertIn("BCS701", str(q1))
        self.assertIn("[10M]", str(q1))

        q2 = ExamQuestion.objects.create(
            module=self.module,
            question_text="Define cryptanalysis and passive attacks. [5M]",
            marks=5,
            frequency=1,
            years_asked="Jan 2024",
            scheme="2022 Scheme"
        )
        self.assertFalse(q2.is_repeated)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_vtu_questions_ai_service(self, mock_ai_call):
        """Tests question generation via AI service and persistence."""
        from curriculum.services.ai_exam import generate_vtu_questions_for_module

        mock_ai_call.return_value = {
            "questions": [
                {
                    "question_text": "Explain DES encryption algorithm with neat block diagram. [10M]",
                    "marks": 10,
                    "frequency": 3,
                    "years_asked": "Jan 2023, Jul 2022",
                    "scheme": "2022 Scheme",
                    "topic_title": "Substitution and Transposition Techniques"
                },
                {
                    "question_text": "Differentiate between block ciphers and stream ciphers. [6M]",
                    "marks": 6,
                    "frequency": 2,
                    "years_asked": "Feb 2024",
                    "scheme": "2022 Scheme"
                }
            ]
        }

        questions = generate_vtu_questions_for_module(self.module)
        self.assertEqual(len(questions), 2)
        self.assertEqual(ExamQuestion.objects.filter(module=self.module).count(), 2)
        q = ExamQuestion.objects.get(marks=6)
        self.assertEqual(q.frequency, 2)
        self.assertTrue(q.is_repeated)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_vtu_answer_blueprint_caching(self, mock_ai_call):
        """Tests blueprint generation and automatic caching on ExamQuestion."""
        from curriculum.services.ai_exam import generate_vtu_answer_blueprint

        mock_ai_call.return_value = {
            "question": "Explain AES with diagram. [10M]",
            "marks": 10,
            "time_budget_mins": 15,
            "definition": "AES is an iterative symmetric block cipher.",
            "diagram": {
                "title": "AES Architecture",
                "ascii_art": "+-----+\n| AES |\n+-----+",
                "drawing_tips": "Draw 4 parallel stages."
            },
            "core_concepts": [
                {"heading": "1. SubBytes", "explanation": "Non-linear byte substitution using S-box."}
            ],
            "working_steps": ["Step 1: AddRoundKey", "Step 2: SubBytes"],
            "examiner_underlines": ["S-box Substitution", "ShiftRows", "MixColumns"],
            "exam_hall_tips": "Always mention the fixed block size is 128 bits."
        }

        eq = ExamQuestion.objects.create(
            module=self.module,
            question_text="Explain AES with diagram. [10M]",
            marks=10
        )
        self.assertIsNone(eq.answer_blueprint)

        bp = generate_vtu_answer_blueprint(
            question_text=eq.question_text,
            marks=eq.marks,
            subject_name=self.subject.name,
            module_title=self.module.title,
            exam_question=eq
        )
        eq.refresh_from_db()
        self.assertIsNotNone(eq.answer_blueprint)
        self.assertEqual(eq.answer_blueprint["time_budget_mins"], 15)
        self.assertEqual(len(eq.answer_blueprint["examiner_underlines"]), 3)

        # Call again without mock returning anything new — should return cached directly
        mock_ai_call.side_effect = Exception("Should not call API when cached!")
        bp2 = generate_vtu_answer_blueprint(
            question_text=eq.question_text,
            marks=eq.marks,
            exam_question=eq
        )
        self.assertEqual(bp2["definition"], "AES is an iterative symmetric block cipher.")

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_module_questions_ajax_view(self, mock_ai_call):
        """Tests AJAX endpoint for module question generation."""
        mock_ai_call.return_value = {
            "questions": [
                {
                    "question_text": "Discuss Modular Arithmetic in Cryptography. [8M]",
                    "marks": 8,
                    "frequency": 2,
                    "years_asked": "Jul 2023",
                    "scheme": "2022 Scheme"
                }
            ]
        }
        url = reverse('curriculum:generate_module_questions', kwargs={'module_id': self.module.id})
        res = self.client.post(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['questions']), 1)
        self.assertEqual(data['questions'][0]['marks'], 8)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_get_question_blueprint_view(self, mock_ai_call):
        """Tests AJAX endpoint for retrieving a question's answer blueprint."""
        mock_ai_call.return_value = {
            "question": "Sample Q",
            "marks": 10,
            "definition": "Sample definition",
            "diagram": {"ascii_art": "+---+"},
            "examiner_underlines": ["Key1", "Key2"]
        }
        eq = ExamQuestion.objects.create(
            module=self.module,
            question_text="Sample Q",
            marks=10
        )
        url = reverse('curriculum:get_question_blueprint', kwargs={'question_id': eq.id})
        res = self.client.get(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertIn('blueprint', data)
        self.assertEqual(data['blueprint']['definition'], "Sample definition")

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_topic_blueprint_view(self, mock_ai_call):
        """Tests AJAX endpoint for generating answer blueprint for any topic."""
        mock_ai_call.return_value = {
            "question": "Explain Substitution Techniques",
            "marks": 10,
            "definition": "Substitution replaces letters with other letters.",
            "diagram": {"ascii_art": "A -> D"},
            "examiner_underlines": ["Caesar Cipher", "Monoalphabetic"]
        }
        url = reverse('curriculum:generate_topic_blueprint', kwargs={'topic_id': self.topic.id})
        res = self.client.get(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['topic_title'], self.topic.title)
        self.assertEqual(data['blueprint']['marks'], 10)

    def test_subject_detail_view_exam_questions_context(self):
        """Tests subject detail view includes active_module_questions and repeated count."""
        ExamQuestion.objects.create(
            module=self.module,
            question_text="Repeated question in detail view. [10M]",
            marks=10,
            frequency=3
        )
        ExamQuestion.objects.create(
            module=self.module,
            question_text="Single question in detail view. [6M]",
            marks=6,
            frequency=1
        )
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.context['active_module_questions']), 2)
        self.assertEqual(res.context['active_module_repeated_count'], 1)
        self.assertContains(res, "PYQ Bank")
        self.assertContains(res, "10-Mark Answer Architect")


class RapidRevisionAndQuizTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Deep Learning and Neural Networks",
            code="BAI702",
            slug="deep-learning",
            branch=Subject.Branch.AIML,
            semester=7,
            scheme="2022 Scheme"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=2,
            title="Convolutional Neural Networks and Computer Vision",
            description="CNN architectures, convolutions, pooling, ResNet"
        )
        self.topic = Topic.objects.create(
            module=self.module,
            title="Convolution and Pooling Operations",
            slug="convolution-pooling",
            order=1
        )

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_module_cheat_sheet_caching(self, mock_ai):
        """Tests that cheat sheet generates via AI and caches on Module record."""
        from curriculum.services.ai_exam import generate_module_cheat_sheet

        mock_ai.return_value = {
            "module_number": 2,
            "module_title": "CNNs",
            "subject_code": "BAI702",
            "key_definitions": [{"term": "Stride", "definition": "Step size of filter sliding across input image."}],
            "acronyms": [{"acronym": "CNN", "stands_for": "Convolutional Neural Network"}],
            "formulas_and_complexities": [{"name": "Output Size", "formula": "O = ((W - K + 2P)/S) + 1"}],
            "architecture_summary": {"title": "CNN Pipeline", "ascii_diagram": "+---+"},
            "last_minute_rules": ["Pooling reduces spatial dimensions without altering channel depth."]
        }

        self.assertIsNone(self.module.cheat_sheet)
        cs = generate_module_cheat_sheet(self.module)
        self.module.refresh_from_db()

        self.assertIsNotNone(self.module.cheat_sheet)
        self.assertEqual(self.module.cheat_sheet["module_number"], 2)
        self.assertEqual(cs["acronyms"][0]["acronym"], "CNN")

        # Second call should use cached data without calling AI
        mock_ai.side_effect = Exception("Should not hit AI API on second call!")
        cs2 = generate_module_cheat_sheet(self.module)
        self.assertEqual(cs2["key_definitions"][0]["term"], "Stride")

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_module_cheat_sheet_fallback(self, mock_ai):
        """Tests deterministic fallback cheat sheet generation when AI fails."""
        from curriculum.services.ai_exam import generate_module_cheat_sheet

        mock_ai.side_effect = Exception("AI Service Unavailable")
        cs = generate_module_cheat_sheet(self.module)

        self.assertEqual(cs["module_number"], 2)
        self.assertTrue(len(cs["key_definitions"]) > 0)
        self.assertIn("formulas_and_complexities", cs)
        self.assertIn("architecture_summary", cs)
        self.assertTrue(len(cs["last_minute_rules"]) > 0)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_generate_module_quiz_caching(self, mock_ai):
        """Tests that micro-quiz generates and caches on Module record."""
        from curriculum.services.ai_exam import generate_module_quiz

        mock_ai.return_value = {
            "module_number": 2,
            "questions": [
                {
                    "id": 1,
                    "question": "What is the primary effect of max pooling?",
                    "options": ["Reduces spatial resolution", "Increases parameter count", "Slows training", "Adds channels"],
                    "correct_answer": 0,
                    "explanation": "Max pooling downsamples spatial height and width."
                }
            ],
            "flashcards": [
                {
                    "id": 1,
                    "front": "What does 1x1 convolution do?",
                    "back": "Performs cross-channel dimension reduction while preserving spatial dimensions."
                }
            ]
        }

        self.assertIsNone(self.module.quiz_data)
        qd = generate_module_quiz(self.module)
        self.module.refresh_from_db()

        self.assertIsNotNone(self.module.quiz_data)
        self.assertEqual(len(qd["questions"]), 1)
        self.assertEqual(len(qd["flashcards"]), 1)

        # Second call uses cache
        mock_ai.side_effect = Exception("Should not hit AI API on second call!")
        qd2 = generate_module_quiz(self.module)
        self.assertEqual(qd2["questions"][0]["correct_answer"], 0)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_get_module_cheat_sheet_view(self, mock_ai):
        """Tests AJAX endpoint for retrieving module cheat sheet."""
        mock_ai.return_value = {
            "module_number": 2,
            "key_definitions": [{"term": "Kernel", "definition": "Convolution filter matrix."}]
        }
        url = reverse('curriculum:module_cheat_sheet', kwargs={'module_id': self.module.id})
        res = self.client.get(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['module_id'], self.module.id)
        self.assertIn('cheat_sheet', data)

    @patch("curriculum.services.ai_exam._call_ai_json")
    def test_get_module_quiz_view(self, mock_ai):
        """Tests AJAX endpoint for retrieving module quiz and flashcards."""
        mock_ai.return_value = {
            "module_number": 2,
            "questions": [{"id": 1, "question": "Test Q", "options": ["A", "B"], "correct_answer": 0, "explanation": "Exp"}],
            "flashcards": [{"id": 1, "front": "F", "back": "B"}]
        }
        url = reverse('curriculum:module_quiz', kwargs={'module_id': self.module.id})
        res = self.client.get(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['module_id'], self.module.id)
        self.assertIn('quiz_data', data)

    def test_subject_detail_view_revision_buttons(self):
        """Tests that subject detail view renders the quick revision bar and buttons."""
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "1-Page Cheat Sheet")
        self.assertContains(res, "Quiz & Flashcards")
        self.assertContains(res, "cheatSheetModal")
        self.assertContains(res, "quizModal")











class NightBeforeHighYieldTriageTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Distributed Systems",
            code="BCS702",
            slug="distributed-systems",
            branch=Subject.Branch.CSE,
            semester=7,
            order=1
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Introduction to Distributed Systems",
            description="Basics"
        )
        # Topic 1: marked is_high_yield=True
        self.t1 = Topic.objects.create(
            module=self.module,
            title="System Models & Architecture",
            slug="system-models",
            order=1,
            is_high_yield=True
        )
        # Topic 2: marked is_high_yield=True
        self.t2 = Topic.objects.create(
            module=self.module,
            title="Networking Protocols & Sockets",
            slug="networking-internetworking",
            order=2,
            is_high_yield=True
        )
        # Topic 3: order 3 -> not high yield
        self.t3 = Topic.objects.create(
            module=self.module,
            title="Interprocess Communication",
            slug="interprocess-communication",
            order=3
        )
        # Topic 4: order 4 but explicitly marked is_high_yield=True
        self.t4 = Topic.objects.create(
            module=self.module,
            title="Remote Method Invocation",
            slug="remote-method-invocation",
            order=4,
            is_high_yield=True,
            vtu_frequency_tag="Asked in Q1 (10M)"
        )

    def test_effective_high_yield_properties(self):
        self.assertTrue(self.t1.is_high_yield_effective)
        self.assertTrue(self.t2.is_high_yield_effective)
        self.assertFalse(self.t3.is_high_yield_effective)
        self.assertTrue(self.t4.is_high_yield_effective)

        hy_topics = self.module.high_yield_topics
        self.assertEqual(len(hy_topics), 3)
        self.assertIn(self.t1, hy_topics)
        self.assertIn(self.t2, hy_topics)
        self.assertIn(self.t4, hy_topics)
        self.assertNotIn(self.t3, hy_topics)

    def test_subject_detail_view_high_yield_elements(self):
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)

        # Context assertions
        self.assertEqual(res.context['high_yield_total'], 3)
        self.assertIn('target_marks_potential', res.context)
        self.assertIn('marks_secured', res.context)

        # HTML assertions
        self.assertContains(res, 'btnToggleHighYield')
        self.assertContains(res, 'highYieldTriageCard')
        self.assertContains(res, 'Target Marks Gauge')
        self.assertContains(res, 'data-high-yield="true"')
        self.assertContains(res, 'data-high-yield="false"')
        self.assertContains(res, 'Night-Before 80/20')


class VisualSheetGalleryTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Computer Graphics",
            code="BCS601",
            slug="computer-graphics",
            branch=Subject.Branch.CSE,
            semester=6,
            order=1
        )
        self.module1 = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Introduction to Graphics",
            description="Graphics pipeline"
        )
        self.module2 = Module.objects.create(
            subject=self.subject,
            module_number=2,
            title="2D Transformations",
            description="Matrix math"
        )
        # Add topics
        Topic.objects.create(module=self.module1, title="Pipeline Stages", slug="pipeline-stages", order=1)
        Topic.objects.create(module=self.module2, title="Rotation Matrices", slug="rotation-matrices", order=1)

    def test_get_subject_visual_gallery_all_and_filtered(self):
        from curriculum.services.ai_exam import get_subject_visual_gallery
        # All modules
        data = get_subject_visual_gallery(self.subject)
        self.assertIn("diagrams", data)
        self.assertIn("formulas", data)
        self.assertTrue(len(data["diagrams"]) > 0)
        self.assertTrue(len(data["formulas"]) > 0)

        diag = data["diagrams"][0]
        self.assertIn("ascii_diagram", diag)
        self.assertIn("drawing_tip", diag)
        self.assertIn("title", diag)

        form = data["formulas"][0]
        self.assertIn("formula", form)
        self.assertIn("name", form)

        # Filtered to Module 1
        data_m1 = get_subject_visual_gallery(self.subject, module_number=1)
        for d in data_m1["diagrams"]:
            self.assertEqual(d["module_number"], 1)

    def test_subject_visual_sheet_view_endpoint(self):
        url = reverse('curriculum:subject_visual_sheet', kwargs={'slug': self.subject.slug})
        res = self.client.get(url, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['subject_slug'], 'computer-graphics')
        self.assertTrue(len(data['diagrams']) > 0)
        self.assertTrue(len(data['formulas']) > 0)

        # Filter query parameter
        res_m2 = self.client.get(f"{url}?module=2", HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res_m2.status_code, 200)
        data_m2 = res_m2.json()
        self.assertTrue(data_m2['success'])
        self.assertEqual(data_m2['active_module_filter'], 2)

    def test_subject_detail_view_visual_gallery_ui(self):
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, 'btn-open-visual-gallery')
        self.assertContains(res, 'visualGalleryModal')
        self.assertContains(res, 'Diagrams Story-Gallery')
        self.assertContains(res, 'Formula Bank')
        self.assertContains(res, '60-Min Visual Sheet')


class PassingPredictorTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Microcontroller and Embedded Systems",
            code="BCS401",
            slug="embedded-systems",
            branch=Subject.Branch.CSE,
            semester=4,
            scheme="2022 Scheme"
        )
        self.mod1 = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="ARM Cortex-M0 Architecture",
            description="Core registers and pipeline"
        )
        self.mod2 = Module.objects.create(
            subject=self.subject,
            module_number=2,
            title="Embedded C and Assembly",
            description="Instructions and interfacing"
        )
        self.t1 = Topic.objects.create(
            module=self.mod1,
            title="ARM Core Registers & Pipeline",
            slug="arm-registers",
            order=1,
            is_high_yield=True,
            vtu_frequency_tag="Repeated 4+ yrs"
        )
        self.t2 = Topic.objects.create(
            module=self.mod1,
            title="Memory Map and Bus Architecture",
            slug="memory-map",
            order=2,
            is_high_yield=False
        )
        self.t3 = Topic.objects.create(
            module=self.mod2,
            title="GPIO Interfacing and Registers",
            slug="gpio-interfacing",
            order=1,
            is_high_yield=True
        )

    def test_simulator_data_in_subject_detail_context(self):
        """Tests that subject_detail view provides serialized simulator_data_json."""
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)

        # Context has simulator_data_json
        self.assertIn('simulator_data_json', res.context)
        import json
        data = json.loads(res.context['simulator_data_json'])
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['module_number'], 1)
        self.assertEqual(data[0]['q_label'], 'Q1 or Q2')
        self.assertEqual(len(data[0]['topics']), 2)

        topic_1_data = next(t for t in data[0]['topics'] if t['id'] == self.t1.id)
        self.assertTrue(topic_1_data['is_high_yield'])
        self.assertEqual(topic_1_data['vtu_frequency_tag'], 'Repeated 4+ yrs')
        self.assertFalse(topic_1_data['is_completed'])

    def test_subject_detail_renders_passing_predictor_modal_and_controls(self):
        """Tests that HTML contains the predictor modal, trigger buttons, and scoreboard elements."""
        url = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        res = self.client.get(url)
        self.assertEqual(res.status_code, 200)

        # Buttons and modal
        self.assertContains(res, 'btnOpenPassingPredictor')
        self.assertContains(res, 'passingPredictorModal')
        self.assertContains(res, 'predMarksDisplay')
        self.assertContains(res, 'predScoreProgressBar')
        self.assertContains(res, 'predTacticalAdviceBox')
        self.assertContains(res, 'predictor-preset-btn')
        self.assertContains(res, 'Target Pass (35M)')
        self.assertContains(res, 'First Class (60M)')
        self.assertContains(res, 'btnApplyPredictorToDesk')
        self.assertContains(res, 'btnSyncPredictorWithDesk')

    def test_batch_update_topic_progress_endpoint(self):
        """Tests /subjects/<slug>/batch-progress/ updates multiple topic progress records."""
        url = reverse('curriculum:batch_update_topic_progress', kwargs={'slug': self.subject.slug})

        # Method not allowed for GET
        res_get = self.client.get(url)
        self.assertEqual(res_get.status_code, 405)

        # POST batch update
        payload = {
            'topic_ids': [self.t1.id, self.t3.id],
            'status': TopicProgress.Status.COMPLETED
        }
        res_post = self.client.post(
            url,
            data=json.dumps(payload),
            content_type='application/json',
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        self.assertEqual(res_post.status_code, 200)
        data = res_post.json()
        self.assertTrue(data['success'])
        self.assertEqual(data['updated_count'], 2)

        # Verify DB records
        p1 = TopicProgress.objects.get(topic=self.t1)
        self.assertEqual(p1.status, TopicProgress.Status.COMPLETED)

        p3 = TopicProgress.objects.get(topic=self.t3)
        self.assertEqual(p3.status, TopicProgress.Status.COMPLETED)


class GuestSessionProgressIsolationTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Cloud Computing & Virtualization",
            code="BCS701",
            slug="cloud-computing",
            branch=Subject.Branch.CSE,
            semester=7,
            scheme="2022 Scheme"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Introduction to Cloud Computing",
            description="Cloud models"
        )
        self.t1 = Topic.objects.create(
            module=self.module,
            title="Introduction to Cloud Architecture",
            slug="cloud-architecture",
            order=1
        )
        self.t2 = Topic.objects.create(
            module=self.module,
            title="Virtualization & Hypervisor Mechanics",
            slug="virtualization-hypervisors",
            order=2
        )

    def test_anonymous_sessions_are_completely_isolated(self):
        """
        Tests that user progress updated on Client A (e.g. laptop) does not leak
        into Client B (e.g. phone), and each client's progress updates independently.
        """
        from django.test import Client
        client_a = Client()
        client_b = Client()

        # Client A visits and completes topic 1
        url_t1 = reverse('curriculum:update_topic_progress', kwargs={'topic_id': self.t1.id})
        res_a = client_a.post(url_t1, {'status': 'COMPLETED'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res_a.status_code, 200)

        # Client A views subject detail: t1 is COMPLETED, 1 of 2 completed
        url_subject = reverse('curriculum:subject_detail', kwargs={'slug': self.subject.slug})
        detail_a = client_a.get(url_subject)
        self.assertEqual(detail_a.context['completed_topics'], 1)
        self.assertEqual(detail_a.context['active_topic'].progress.status, 'COMPLETED')

        # Client B (e.g. phone) visits the subject page: progress MUST BE 0!
        detail_b = client_b.get(url_subject)
        self.assertEqual(detail_b.context['completed_topics'], 0)
        # In client B, t1 has no completed progress
        t1_in_b = next(t for t in detail_b.context['subject'].modules.first().topics.all() if t.id == self.t1.id)
        self.assertIsNone(t1_in_b.progress)

        # Client B completes topic 2
        url_t2 = reverse('curriculum:update_topic_progress', kwargs={'topic_id': self.t2.id})
        res_b = client_b.post(url_t2, {'status': 'COMPLETED'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(res_b.status_code, 200)

        # Verify Client B has 1 completed topic (topic 2), not topic 1
        detail_b_after = client_b.get(url_subject)
        self.assertEqual(detail_b_after.context['completed_topics'], 1)

        # Re-verify Client A: still has ONLY topic 1 completed, completely untouched by Client B!
        detail_a_after = client_a.get(url_subject)
        self.assertEqual(detail_a_after.context['completed_topics'], 1)
        t1_in_a = next(t for t in detail_a_after.context['subject'].modules.first().topics.all() if t.id == self.t1.id)
        t2_in_a = next(t for t in detail_a_after.context['subject'].modules.first().topics.all() if t.id == self.t2.id)
        self.assertEqual(t1_in_a.progress.status, 'COMPLETED')
        self.assertIsNone(t2_in_a.progress)


class ExamHighYieldTaggerTests(TestCase):
    def setUp(self):
        self.subject = Subject.objects.create(
            name="Deep Learning Test",
            code="BAI701T",
            slug="dl-test",
            branch=Subject.Branch.AIML,
            semester=7,
            scheme="2022 Scheme"
        )
        self.module = Module.objects.create(
            subject=self.subject,
            module_number=1,
            title="Foundations of Deep Networks",
            description="Core mechanics"
        )
        self.t_intro = Topic.objects.create(
            module=self.module,
            title="Introduction to Machine Learning Approaches and Overview",
            slug="intro-ml",
            order=1
        )
        self.t_arch = Topic.objects.create(
            module=self.module,
            title="Deep Neural Network Architectures and Model Representation",
            slug="dnn-architectures",
            order=2
        )
        self.t_why = Topic.objects.create(
            module=self.module,
            title="Why Use Deep Learning and Industry Applications",
            slug="why-use-dl",
            order=3
        )
        self.t_backprop = Topic.objects.create(
            module=self.module,
            title="How Deep Learning Works: Weights, Loss Functions, and Backpropagation",
            slug="weights-backprop",
            order=4
        )

    def test_score_topic_exam_importance_penalizes_intro_and_rewards_algorithms(self):
        from curriculum.services.high_yield import score_topic_exam_importance
        # Intro and "Why Use" get negative scores
        self.assertLess(score_topic_exam_importance("Introduction to Machine Learning Approaches"), 0)
        self.assertLess(score_topic_exam_importance("Why Use Deep Learning and Applications"), 0)
        self.assertLess(score_topic_exam_importance("Overview of Neural Networks"), 0)

        # Core algorithms, architectures, and derivations get strong positive scores
        self.assertGreater(score_topic_exam_importance("Weights, Loss Functions, and Backpropagation"), 15)
        self.assertGreater(score_topic_exam_importance("Deep Neural Network Architectures"), 8)
        self.assertGreater(score_topic_exam_importance("The RSA Algorithm and Computational Aspects"), 15)
        self.assertGreater(score_topic_exam_importance("Markov Decision Processes and Bellman Equations"), 15)

    def test_tag_module_high_yield_topics_selects_real_exam_concepts(self):
        from curriculum.services.high_yield import tag_module_high_yield_topics
        picks = tag_module_high_yield_topics(self.module, max_picks=2)

        # Should pick Backpropagation (order 4) and Architectures (order 2)
        pick_ids = [p.id for p in picks]
        self.assertIn(self.t_backprop.id, pick_ids)
        self.assertIn(self.t_arch.id, pick_ids)

        # Must NOT pick introductory topics (order 1 or 3)
        self.assertNotIn(self.t_intro.id, pick_ids)
        self.assertNotIn(self.t_why.id, pick_ids)

        # Verify DB updates
        self.t_backprop.refresh_from_db()
        self.t_intro.refresh_from_db()
        self.assertTrue(self.t_backprop.is_high_yield)
        self.assertFalse(self.t_intro.is_high_yield)

    def test_ai_generated_curriculum_persists_high_yield_tags(self):
        """Tests that when AI generation returns is_high_yield in curriculum payload, it is stored in DB."""
        from curriculum.services.ai_curriculum import persist_curriculum_records

        payload = {
            "code": "BCS801",
            "name": "Distributed Systems AI",
            "description": "Distributed systems syllabus",
            "modules": [
                {
                    "module_number": 1,
                    "title": "Module 1 Models",
                    "description": "Models description",
                    "topics": [
                        {
                            "chapter_number": 1,
                            "chapter_title": "Basics",
                            "title": "Introduction to Distributed Systems",
                            "description": "Intro overview",
                            "is_high_yield": False,
                            "vtu_frequency_tag": "",
                            "keywords": ["intro k1", "intro k2", "intro k3"]
                        },
                        {
                            "chapter_number": 1,
                            "chapter_title": "Consensus",
                            "title": "Raft Consensus Algorithm & State Machine Replication",
                            "description": "Core consensus protocol",
                            "is_high_yield": True,
                            "vtu_frequency_tag": "Frequent 10M VTU Question",
                            "keywords": ["raft k1", "raft k2", "raft k3"]
                        }
                    ]
                }
            ]
        }

        subj = persist_curriculum_records({
            "data": payload,
            "clean_name": "Distributed Systems AI",
            "clean_code": "BCS801",
            "is_verified_source": False,
            "active_model": "groq-test"
        })
        m1 = subj.modules.get(module_number=1)
        t_intro = m1.topics.get(title="Introduction to Distributed Systems")
        t_raft = m1.topics.get(title__icontains="Raft Consensus")

        self.assertFalse(t_intro.is_high_yield)
        self.assertTrue(t_raft.is_high_yield)
        self.assertEqual(t_raft.vtu_frequency_tag, "Frequent 10M VTU Question")
