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
from curriculum.models import Module, Subject, Topic, TopicKeyword, GenerationJob
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
    AITransientError,
    calculate_backoff_delay,
    generate_curriculum_from_prompt,
    parse_prescribed_syllabus,
    _extract_chapters_from_text,
    resolve_configured_models,
    resolve_ai_provider,
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
        "model": "llama-3.3-70b-versatile",
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
            self.assertIn("Neither GROQ_API_KEY nor GEMINI_API_KEY", str(ctx.exception))

    @patch('curriculum.services.ai_curriculum.requests.post')
    def test_groq_successful_generation(self, mock_post):
        """Tests end-to-end curriculum generation using Groq API and Llama 3.3 70B."""
        mock_post.return_value = make_mock_groq_json_response(SAMPLE_CURRICULUM_DATA, status_code=200)

        def mock_config(key, default='', **kwargs):
            if key == 'GROQ_API_KEY':
                return 'gsk_mock_valid_key_12345'
            if key == 'AI_PROVIDER':
                return 'groq'
            if key == 'GROQ_PRIMARY_MODEL':
                return 'llama-3.3-70b-versatile'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'llama-3.1-8b-instant'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            result = generate_curriculum_data("Machine Learning", "CS702")

            self.assertEqual(result["active_model"], "llama-3.3-70b-versatile")
            self.assertEqual(len(result["data"]["modules"]), 5)
            self.assertEqual(result["clean_name"], "Machine Learning")

            # Check that request went to Groq OpenAI-compatible endpoint
            self.assertTrue(mock_post.called)
            call_url = mock_post.call_args[0][0]
            self.assertEqual(call_url, "https://api.groq.com/openai/v1/chat/completions")
            call_headers = mock_post.call_args[1]["headers"]
            self.assertEqual(call_headers["Authorization"], "Bearer gsk_mock_valid_key_12345")

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
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            with self.assertRaises(AIAuthError):
                generate_curriculum_data("Machine Learning", "CS702")
            # Should fail fast without retry loops
            self.assertEqual(mock_post.call_count, 1)

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
            if key == 'GROQ_PRIMARY_MODEL':
                return 'llama-3.3-70b-versatile'
            if key == 'GROQ_FALLBACK_MODEL':
                return 'llama-3.1-8b-instant'
            return default

        with patch('curriculum.services.ai_curriculum.config', side_effect=mock_config):
            result = generate_curriculum_data("Machine Learning", "CS702")
            self.assertEqual(result["active_model"], "llama-3.1-8b-instant")
            self.assertEqual(len(result["data"]["modules"]), 5)





