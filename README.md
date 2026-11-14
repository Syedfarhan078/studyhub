# StudyHub 🎓
### A Syllabus-Aligned Learning Portal Built for Engineering Students

---

## 💡 Why I Built This (The Problem)

Preparing for semester exams and internal tests as an engineering student has always felt needlessly chaotic. Every semester, the same frustrations repeat:

* **Scattered Syllabi**: University curriculum outlines are buried inside clunky 200-page university PDF circulars. Knowing what is actually in Module 3 or Module 5 of a subject often requires searching through downloaded files or asking classmates.
* **The YouTube Rabbit Hole**: When studying a specific syllabus topic (like *Backpropagation Derivation* or *Non-Traditional Machining*), searching on YouTube often returns unrelated hour-long playlists, unhelpful clickbait, or content taught in different notations that doesn't match university exam questions.
* **Lost Notes and Question Banks**: Crucial study resources—professor slides, seniors' handwritten scans, and VTU question banks—are constantly scattered across WhatsApp groups, Telegram channels, and temporary Google Drive links that break or get buried right before exams.
* **Zero Visibility on Preparation**: It is hard to keep track of what you have actually understood versus what you still need to cover, leading to last-minute exam panic.

I wanted a single, dedicated, distraction-free web workspace where all my subjects are organized strictly by syllabus hierarchy, where quality learning videos and articles are mapped directly to every topic, and where module handouts are safely archived in one place.

---

## 🚀 What StudyHub Does (The Solution)

**StudyHub** is a syllabus-first educational platform designed to organize engineering coursework from Day 1 of classes all the way through final exams.

Instead of browsing the open internet aimlessly, StudyHub structures everything into a clean **Subject $\rightarrow$ Module $\rightarrow$ Topic $\rightarrow$ Targeted Resources** hierarchy. Every single topic in the syllabus becomes an interactive workspace equipped with curated video lectures, technical documentation, personal formula notes, and downloadable handouts.

---

## ✨ Core Features

### ⚡ 1. Autonomous AI Curriculum Ingestion
Adding new university courses manually is tedious. StudyHub features an automated curriculum generator powered by Google Gemini:
* Simply enter a subject name (e.g. *Natural Language Processing - BCS701*) or paste raw syllabus text from a college circular.
* In seconds, it generates the full 5-module university curriculum, fills in conceptual learning descriptions for every topic, and auto-generates 100+ targeted search keywords optimized for resource discovery.

### 🎥 2. Topic-Mapped Video Learning
* Each syllabus topic is directly connected to curated YouTube educational video lectures.
* Eliminates search clutter by targeting precise academic queries instead of vague keywords.
* Videos can be watched directly inside a distraction-free, 16:9 embedded theater modal without recommended video sidebars or distractions.

### 📰 3. Prioritized Article Discovery
* Automatically pulls documentation, technical tutorials, and articles from verified developer publications.
* Curated ranking prioritizing student-friendly resources like **GeeksforGeeks** and **W3Schools** at the top, followed by research papers and technical guides.

### 📁 4. Centralized Module Notes Repository
* An integrated, cloud-backed repository for uploading and downloading module handouts, handwritten scans, and exam question banks.
* Backed by Cloudinary permanent storage so files are never lost between deployments or container restarts.
* Displays human-readable file formats (`PDF`, `DOCX`) and file sizes for quick scanning on mobile or laptop.

### 📊 5. Visual Progress Tracking & Exam Notes
* **Topic Mastery Toggles**: Mark topics as `Not Started`, `In Progress`, or `Completed` with a single click.
* **Syllabus Progress Bar**: Real-time visual progress percentage across each subject's entire syllabus.
* **Personal Topic Notes**: An expandable drawer on every topic to jot down quick exam pointers, formulas, and reminders.

### 🎨 6. Engineering-Grade Dark Aesthetic
* Clean, low-fatigue slate/zinc design system (`#0d1117`, `#161b22`) inspired by tools like GitHub and Linear.
* Built mobile-first with touch-friendly navigation, horizontal scrollable breadcrumbs, and zero distracting animations.



## 🛠️ Built With

* **Backend**: Django 5 (Python), PostgreSQL (Relational Database & Task Queue), Gunicorn
* **Background Worker**: Standalone PostgreSQL-backed Django Management Command (`run_generation_worker`)
* **Frontend**: Bootstrap 5.3 (Native Dark Theme), Bootstrap Icons
* **Asset & Media Delivery**: WhiteNoise, Cloudinary
* **APIs & AI**: Google Gemini API, YouTube Data API v3, Feedparser

---

## ⚙️ Architecture: PostgreSQL-Backed Background Task Queue

StudyHub uses a **zero-cost, Fire-and-Poll background processing architecture** backed entirely by PostgreSQL. It eliminates the need for external, paid message brokers like Redis:

1. **Fire (`POST /subjects/generate-ai/`)**:
   - The web client submits a syllabus generation request via AJAX.
   - Django validates the input, creates a `GenerationJob` record in PostgreSQL with status `PENDING`, and immediately responds with a UUID `task_id` in $<1\text{s}$.
   - No slow AI API calls are ever executed inside synchronous HTTP request threads.

2. **Process (`python manage.py run_generation_worker`)**:
   - A standalone Django background worker continuously polls PostgreSQL for pending jobs.
   - It claims jobs atomically using `SELECT ... FOR UPDATE SKIP LOCKED` and immediately releases the lock before initiating the external API call.
   - The worker executes source-grounded syllabus parsing, invokes the configured Google Gemini model with exponential backoff and fallback, validates output against the official syllabus, and saves the curriculum inside an atomic transaction.
   - Includes automatic lease tracking and reaper functionality to recover or fail abandoned jobs.

3. **Poll (`GET /subjects/generation-status/<task_id>/`)**:
   - The frontend periodically polls the status endpoint every 2 seconds.
   - Displays real-time progress percentages and step messages (`Step 1 of 4` through `Step 4 of 4`).
   - Automatically redirects to the generated syllabus workspace upon completion or surfaces user-friendly error diagnostics upon failure.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
