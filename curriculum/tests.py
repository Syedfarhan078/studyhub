import json
from unittest.mock import MagicMock, call, patch

import requests
from django.contrib.messages import get_messages
from django.test import Client, TestCase
from django.urls import reverse

from curriculum.models import Module, Subject, Topic, TopicKeyword
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
