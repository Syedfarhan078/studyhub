import json
from unittest.mock import MagicMock, call, patch

import requests
from django.contrib.messages import get_messages
from django.test import Client, TestCase, override_settings
from django.urls import reverse

import uuid
from datetime import timedelta
from django.core.management import call_command
from django.contrib.auth import get_user_model
from django.utils import timezone
from curriculum.models import Module, Subject, Topic, TopicKeyword, GenerationJob
from curriculum.services.ai_curriculum import (
    CurriculumParseError,
    CurriculumValidationError,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    GeminiTransientError,
    calculate_backoff_delay,
    generate_curriculum_from_prompt,
    parse_prescribed_syllabus,
    resolve_configured_models,
    sanitize_error_message,
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
            subject_name="Compiler Design",
            subject_code="BCS707"
        )
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

    # 7. Status endpoint ownership isolation
    def test_status_endpoint_ownership_isolation(self):
        """Tests that authenticated users cannot inspect another user's generation job."""
        job = GenerationJob.objects.create(
            user=self.user_a,
            subject_name="Confidential Computing",
            subject_code="CC901"
        )

        # User B cannot access User A's job
        self.client.force_login(self.user_b)
        resp_forbidden = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(resp_forbidden.status_code, 403)

        # User A can access their own job
        self.client.force_login(self.user_a)
        resp_allowed = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(job.id)}))
        self.assertEqual(resp_allowed.status_code, 200)
        self.assertEqual(resp_allowed.json()["task_id"], str(job.id))

        # Anonymous job can be polled by anyone
        anon_job = GenerationJob.objects.create(
            user=None,
            subject_name="Open Computing",
            subject_code="OC901"
        )
        self.client.logout()
        resp_anon = self.client.get(reverse("curriculum:generation_status", kwargs={"task_id": str(anon_job.id)}))
        self.assertEqual(resp_anon.status_code, 200)

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

        job = GenerationJob.objects.create(subject_name="Deep Learning", subject_code="BCS706")
        call_command("run_generation_worker", once=True)

        job.refresh_from_db()
        self.assertEqual(job.status, GenerationJob.Status.FAILED)
        self.assertIn("temporarily experiencing high server demand", job.error_message)
        self.assertIsNone(job.subject)

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


