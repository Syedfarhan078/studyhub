import logging
import mimetypes
import uuid
import requests
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, Http404, StreamingHttpResponse, FileResponse
from django.urls import reverse
from django.contrib import messages
from django.db import transaction
from django.utils.text import slugify
from .models import Subject, Module, Topic, TopicProgress, GenerationJob, ModuleNote, ExamQuestion
from .forms import ModuleNoteForm
from .services.ai_curriculum import (
    generate_curriculum_from_prompt,
    CurriculumGenerationError,
    AIAuthError,
    AIPermissionError,
    AIModelNotFoundError,
    AIRateLimitError,
    AIServerError,
    AINetworkError,
    AITransientError,
    AIInvalidRequestError,
    CurriculumParseError,
    CurriculumValidationError,
    GeminiTransientError,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
)
from .services.ai_exam import (
    generate_vtu_questions_for_module,
    generate_vtu_answer_blueprint,
    generate_module_cheat_sheet,
    generate_module_quiz,
)
from curriculum.worker import notify_new_job, is_worker_accepting_jobs

logger = logging.getLogger(__name__)

# Create your views here.

def generate_subject_ai(request):
    """
    Handles AI curriculum generation.
    For AJAX/interactive requests: creates a PostgreSQL-backed GenerationJob and immediately
    returns a JSON response with the task identifier for progress polling (<1s).
    For traditional form submissions: processes generation synchronously and redirects gracefully.
    Protects against duplicate submissions and ensures idempotent creation.
    """
    if request.method == 'POST':
        subject_name = request.POST.get('subject_name', '').strip()
        subject_code = request.POST.get('subject_code', '').strip()
        syllabus_text = request.POST.get('syllabus_text', '').strip()
        branch = request.POST.get('branch', '').strip()
        semester_raw = request.POST.get('semester', '').strip()
        semester = int(semester_raw) if semester_raw.isdigit() else None

        is_ajax = (
            request.headers.get('x-requested-with') == 'XMLHttpRequest'
            or 'application/json' in request.headers.get('accept', '')
            or request.content_type == 'application/json'
            or request.POST.get('async') == '1'
        )

        if not subject_name:
            if is_ajax:
                return JsonResponse({
                    'success': False,
                    'status': 'FAILED',
                    'error': 'Please enter a subject name to generate the syllabus.'
                }, status=400)
            messages.error(request, "Please enter a subject name to generate the syllabus.")
            return redirect('/')

        # Duplicate protection: If subject already exists with 5 modules, redirect immediately
        existing_subject = None
        if subject_code:
            existing_subject = Subject.objects.filter(code__iexact=subject_code, name__iexact=subject_name).first()
        if not existing_subject:
            existing_subject = Subject.objects.filter(name__iexact=subject_name).first()

        if existing_subject and existing_subject.modules.exists():
            redirect_url = reverse('curriculum:subject_detail', kwargs={'slug': existing_subject.slug})
            logger.info("Subject '%s' already exists with %d modules. Returning existing record.", existing_subject.name, existing_subject.modules.count())
            if is_ajax:
                return JsonResponse({
                    'success': True,
                    'task_id': None,
                    'status': 'COMPLETED',
                    'progress': 100,
                    'step': 4,
                    'total_steps': 4,
                    'redirect_url': redirect_url,
                    'subject_name': existing_subject.name,
                    'subject_code': existing_subject.code,
                    'subject_slug': existing_subject.slug,
                    'message': f"Subject '{existing_subject.name}' ({existing_subject.code}) is already available with {existing_subject.modules.count()} modules."
                })
            messages.info(request, f"Subject '{existing_subject.name}' already exists.")
            return redirect(redirect_url)

        # Background generation via PostgreSQL GenerationJob for interactive web client
        if is_ajax:
            if not is_worker_accepting_jobs():
                logger.error("Curriculum generation rejected: in-process worker is offline or unavailable.")
                return JsonResponse({
                    'success': False,
                    'status': 'FAILED',
                    'error': 'Curriculum generation is temporarily unavailable. Please try again later.'
                }, status=503)

            user = request.user if request.user.is_authenticated else None
            session_key = ""
            if not user:
                if not request.session.session_key:
                    request.session.save()
                session_key = request.session.session_key or ""

            with transaction.atomic():
                job = GenerationJob.objects.create(
                    user=user,
                    session_key=session_key,
                    subject_name=subject_name,
                    subject_code=subject_code,
                    syllabus_text=syllabus_text,
                    branch=branch,
                    semester=semester,
                    status=GenerationJob.Status.PENDING,
                    progress=0,
                    step=1,
                    total_steps=4,
                    progress_message=f"Job queued and awaiting worker pickup for '{subject_name}'."
                )
                transaction.on_commit(notify_new_job)

            if not user:
                request.session[f"allowed_job_{job.id}"] = True
                request.session.modified = True

            check_url = reverse('curriculum:generation_status', kwargs={'task_id': str(job.id)})
            return JsonResponse({
                'success': True,
                'task_id': str(job.id),
                'status': 'PENDING',
                'progress': 0,
                'step': 1,
                'total_steps': 4,
                'message': f"Curriculum generation initiated for '{subject_name}'.",
                'check_url': check_url,
            })

        # Synchronous fallback for legacy non-AJAX POST (preserves test suite and backwards compatibility)
        try:
            subject = generate_curriculum_from_prompt(
                subject_name=subject_name,
                subject_code=subject_code,
                syllabus_text=syllabus_text,
                branch=branch,
                semester=semester
            )
            messages.success(
                request,
                f"Syllabus for '{subject.name}' ({subject.code}) generated successfully with {subject.modules.count()} modules!"
            )
            return redirect('curriculum:subject_detail', slug=subject.slug)
        except AIAuthError as e:
            logger.error("Authentication error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "AI service authentication error. Please verify that your GROQ_API_KEY is configured correctly."
            )
        except AIPermissionError as e:
            logger.error("Permission error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "AI service permission denied. Please verify your Groq account access and permissions."
            )
        except AIModelNotFoundError as e:
            logger.error("Model not found generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "The configured Groq AI model is not available. Please verify your model settings (default: openai/gpt-oss-120b)."
            )
        except AIRateLimitError as e:
            logger.warning("Rate limit exceeded generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "Groq API rate limit reached. Please wait a moment before trying again."
            )
        except AINetworkError as e:
            logger.error("Network error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "Network error communicating with Groq API. Please check your network connection and try again."
            )
        except (AIServerError, AITransientError) as e:
            logger.warning("Transient error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "The AI curriculum service is temporarily experiencing high server demand. Please wait a few moments and try again."
            )
        except AIInvalidRequestError as e:
            logger.error("Invalid request generating curriculum for '%s': %s", subject_name, e)
            messages.error(request, f"Unable to process curriculum request: {e}")
        except CurriculumValidationError as ve:
            logger.warning("Curriculum validation error for '%s': %s", subject_name, ve)
            messages.error(
                request,
                f"Syllabus validation failed: {ve} Please verify that your syllabus contains 5 clear module headings."
            )
        except CurriculumParseError as e:
            logger.warning("Parse error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "The AI generated an incomplete curriculum structure. Please try again or provide detailed syllabus text."
            )
        except CurriculumGenerationError as e:
            logger.error("Curriculum generation failure for '%s': %s", subject_name, e)
            messages.error(request, f"Curriculum generation error: {e}")
        except ValueError as ve:
            logger.warning("Validation error for '%s': %s", subject_name, ve)
            messages.error(request, str(ve))
        except Exception as e:
            logger.exception("Unexpected error in generate_subject_ai for '%s': %s", subject_name, e)
            messages.error(
                request,
                "An unexpected error occurred while generating the syllabus. Please try again."
            )

    return redirect('/')


def generation_status(request, task_id):
    """
    Returns current execution status of a background curriculum generation task.
    States: PENDING, RUNNING, COMPLETED, FAILED.
    """
    if not task_id:
        return JsonResponse({
            'success': False,
            'status': 'FAILED',
            'error': 'No task identifier provided.'
        }, status=400)

    try:
        job_uuid = uuid.UUID(str(task_id))
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({
            'success': False,
            'status': 'FAILED',
            'error': 'Invalid task identifier format. Expected UUID.'
        }, status=400)

    try:
        job = GenerationJob.objects.select_related('subject').get(id=job_uuid)
    except GenerationJob.DoesNotExist:
        return JsonResponse({
            'success': False,
            'status': 'FAILED',
            'error': 'Generation job not found.'
        }, status=404)

    # Ownership authorization check:
    # 1. Authenticated job: only the authenticated owner can access it
    if job.user_id:
        if not request.user.is_authenticated or job.user_id != request.user.id:
            return JsonResponse({
                'success': False,
                'status': 'FAILED',
                'error': 'Permission denied.'
            }, status=403)
    # 2. Anonymous job: only the original anonymous creator's session can access it
    else:
        current_session_key = request.session.session_key or ""
        session_has_job = request.session.get(f"allowed_job_{job.id}", False)
        is_session_owner = (job.session_key and current_session_key == job.session_key) or session_has_job
        if not is_session_owner:
            return JsonResponse({
                'success': False,
                'status': 'FAILED',
                'error': 'Permission denied.'
            }, status=403)

    redirect_url = None
    subject_slug = None
    if job.subject:
        redirect_url = reverse('curriculum:subject_detail', kwargs={'slug': job.subject.slug})
        subject_slug = job.subject.slug

    is_success = (job.status != GenerationJob.Status.FAILED)

    return JsonResponse({
        'success': is_success,
        'task_id': str(job.id),
        'status': job.status,
        'progress': job.progress,
        'step': job.step,
        'total_steps': job.total_steps,
        'message': job.progress_message,
        'redirect_url': redirect_url,
        'subject_slug': subject_slug,
        'error': job.error_message or None,
    })


def subject_detail(request, slug):
    """
    Renders the syllabus view for a given subject.
    Prefetches modules, topics, keywords, notes, exam questions, and progress records.
    Calculates overall syllabus completion metrics.
    """
    subject = get_object_or_404(
        Subject.objects.prefetch_related(
            'modules__topics__keywords',
            'modules__topics__progress',  # <-- Prefetches progress records
            'modules__notes',
            'modules__exam_questions__topic'
        ),
        slug=slug
    )

    selected_topic_slug = request.GET.get('topic')
    active_topic = None

    all_topics = [topic for module in subject.modules.all() for topic in module.topics.all()]

    if selected_topic_slug:
        active_topic = next((t for t in all_topics if t.slug == selected_topic_slug), None)

    if not active_topic and all_topics:
        active_topic = all_topics[0]

    active_module = active_topic.module if active_topic else (subject.modules.first() if subject.modules.exists() else None)
    active_module_questions = list(active_module.exam_questions.all()) if active_module else []
    active_module_repeated_count = sum(1 for q in active_module_questions if q.is_repeated)

    # Calculate Syllabus Progress Metrics
    total_topics = len(all_topics)
    completed_topics = 0
    for t in all_topics:
        # Check if topic has a progress record marked COMPLETED
        if hasattr(t, 'progress') and t.progress.status == TopicProgress.Status.COMPLETED:
            completed_topics += 1

    progress_percent = int((completed_topics / total_topics) * 100) if total_topics > 0 else 0

    context = {
        'subject': subject,
        'active_topic': active_topic,
        'active_module': active_module,
        'active_module_questions': active_module_questions,
        'active_module_repeated_count': active_module_repeated_count,
        'note_form': ModuleNoteForm(),
        'total_topics': total_topics,
        'completed_topics': completed_topics,
        'progress_percent': progress_percent,
    }
    return render(request, 'curriculum/subject_detail.html', context)


def upload_module_note(request, module_id):
    """
    Handles user file uploads for a specific syllabus module.
    Saves the PDF/doc note and redirects back to the subject page with a success message.
    """
    module = get_object_or_404(Module.objects.select_related('subject'), id=module_id)

    if request.method == 'POST':
        form = ModuleNoteForm(request.POST, request.FILES)
        if form.is_valid():
            note = form.save(commit=False)
            note.module = module  # Attach the note to this specific module
            note.save()
            messages.success(request, f"Note '{note.title}' uploaded successfully for Module {module.module_number}!")
        else:
            messages.error(request, "Failed to upload note. Please verify the file format.")

    # Redirect back to the subject's page
    return redirect('curriculum:subject_detail', slug=module.subject.slug)


def update_topic_progress(request, topic_id):
    """
    Updates the learning progress status and personal notes for a topic.
    """
    topic = get_object_or_404(Topic.objects.select_related('module__subject'), id=topic_id)
    if request.method == 'POST':
        new_status = request.POST.get('status')
        notes_content = request.POST.get('personal_notes', '')
        # Retrieve existing progress or create a new one
        progress, created = TopicProgress.objects.get_or_create(topic=topic)
        
        if new_status in TopicProgress.Status.values:
            progress.status = new_status
        
        progress.personal_notes = notes_content
        progress.save()
        messages.success(request, f"Updated progress for '{topic.title}'!")
    # Redirect back to the syllabus view with this topic selected
    return redirect(f"/subjects/{topic.module.subject.slug}/?topic={topic.slug}")


def _serve_note_file(request, note_id, as_attachment=False):
    """
    Internal file streamer that masks Cloudinary/storage URLs with clean, branded domain URLs.
    Delivers custom academic filenames (e.g. 'CS701_Module_1_Notes.pdf') and proper MIME headers.
    """
    note = get_object_or_404(
        ModuleNote.objects.select_related('module__subject'),
        id=note_id
    )

    if not note.file:
        raise Http404("Note file not found.")

    ext = (note.file_extension or 'pdf').lower()
    subject_code = getattr(note.module.subject, 'code', 'NOTE').strip().upper()
    title_slug = slugify(note.title) or f"module_{note.module.module_number}_note"
    clean_filename = f"{subject_code}_{title_slug}.{ext}"

    content_type, _ = mimetypes.guess_type(note.file.name)
    if not content_type:
        if ext == 'pdf':
            content_type = 'application/pdf'
        elif ext in ('doc', 'docx'):
            content_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        elif ext in ('ppt', 'pptx'):
            content_type = 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
        else:
            content_type = 'application/octet-stream'

    disposition_type = "attachment" if as_attachment else "inline"
    content_disposition = f'{disposition_type}; filename="{clean_filename}"'

    file_url = getattr(note.file, 'url', '')
    raw_name = getattr(note.file, 'name', '')

    # Determine if this file is located on remote storage (Cloudinary or CDN)
    target_remote_url = None
    if file_url.startswith(('http://', 'https://')):
        target_remote_url = file_url
    elif raw_name.startswith(('http://', 'https://')):
        target_remote_url = raw_name
    elif 'res.cloudinary.com' in file_url:
        idx = file_url.find('http')
        if idx != -1:
            target_remote_url = file_url[idx:]
    elif 'res.cloudinary.com' in raw_name:
        idx = raw_name.find('http')
        if idx != -1:
            target_remote_url = raw_name[idx:]

    # If the file is stored remotely (Cloudinary or CDN), stream it to hide Cloudinary origin
    if target_remote_url:
        try:
            req = requests.get(target_remote_url, stream=True, timeout=20)
            if req.status_code != 200:
                logger.error("Failed to fetch remote note file (%s): %s", req.status_code, target_remote_url)
                raise Http404("Unable to fetch note from cloud storage.")

            def stream_chunks():
                for chunk in req.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        yield chunk

            response = StreamingHttpResponse(stream_chunks(), content_type=content_type)
            if 'Content-Length' in req.headers:
                response['Content-Length'] = req.headers['Content-Length']
            response['Content-Disposition'] = content_disposition
            response['Cache-Control'] = 'public, max-age=86400'
            return response
        except requests.RequestException as e:
            logger.error("Error streaming note file %s: %s", note_id, e)
            raise Http404("Error retrieving note file from storage.")

    # Local filesystem storage (development)
    try:
        response = FileResponse(note.file.open('rb'), content_type=content_type)
        response['Content-Disposition'] = content_disposition
        response['Cache-Control'] = 'public, max-age=86400'
        return response
    except Exception as e:
        logger.error("Error opening local note file %s: %s", note_id, e)
        raise Http404("Local note file could not be read.")


def view_module_note(request, note_id):
    """
    Renders note PDF/doc inline in a new tab under the website domain, masking Cloudinary.
    """
    return _serve_note_file(request, note_id, as_attachment=False)


def download_module_note(request, note_id):
    """
    Prompts immediate file download with a clean, branded academic filename.
    """
    return _serve_note_file(request, note_id, as_attachment=True)


def toggle_subject_pin(request):
    """
    AJAX endpoint allowing anonymous and guest students to pin or unpin
    a subject to/from their personal study desk without an account.
    Synchronizes state across session and long-lived cookie.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed. Use POST.'}, status=405)

    slug = (request.POST.get('slug') or '').strip()
    if not slug:
        import json
        try:
            body = json.loads(request.body.decode('utf-8'))
            slug = (body.get('slug') or '').strip()
        except Exception:
            slug = ''

    if not slug:
        return JsonResponse({'success': False, 'error': 'Subject slug is required.'}, status=400)

    subject = get_object_or_404(Subject, slug=slug)

    pinned = set(request.session.get('studyhub_pinned', []))
    cookie_pinned = request.COOKIES.get('studyhub_pinned', '')
    if cookie_pinned:
        for s in cookie_pinned.split(','):
            s_clean = s.strip()
            if s_clean:
                pinned.add(s_clean)

    is_pinned = False
    if slug in pinned:
        pinned.remove(slug)
        is_pinned = False
    else:
        pinned.add(slug)
        is_pinned = True

    pinned_list = sorted(list(pinned))
    request.session['studyhub_pinned'] = pinned_list
    request.session.modified = True

    response = JsonResponse({
        'success': True,
        'slug': slug,
        'is_pinned': is_pinned,
        'total_pinned': len(pinned_list),
        'pinned_slugs': pinned_list,
        'message': f"Subject '{subject.name}' {'pinned to' if is_pinned else 'removed from'} your desk."
    })
    response.set_cookie(
        'studyhub_pinned',
        ','.join(pinned_list),
        max_age=365 * 24 * 60 * 60,
        samesite='Lax'
    )
    return response


def generate_module_questions_view(request, module_id):
    """
    AJAX endpoint to generate or fetch high-yield VTU exam questions for a module.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required.'}, status=405)

    module = get_object_or_404(Module.objects.select_related('subject'), id=module_id)
    try:
        questions = generate_vtu_questions_for_module(module)
        return JsonResponse({
            'success': True,
            'module_id': module.id,
            'module_number': module.module_number,
            'questions': [
                {
                    'id': q.id,
                    'question_text': q.question_text,
                    'marks': q.marks,
                    'frequency': q.frequency,
                    'is_repeated': q.is_repeated,
                    'years_asked': q.years_asked,
                    'scheme': q.scheme,
                    'has_blueprint': bool(q.answer_blueprint)
                } for q in questions
            ]
        })
    except Exception as e:
        logger.exception("Error generating questions for module %s: %s", module_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def get_question_blueprint_view(request, question_id):
    """
    AJAX endpoint returning the VTU 10-Mark Answer Architect blueprint for an exam question.
    Generates and caches the blueprint via AI if not previously cached.
    """
    question = get_object_or_404(
        ExamQuestion.objects.select_related('module__subject', 'topic'),
        id=question_id
    )

    try:
        blueprint = generate_vtu_answer_blueprint(
            question_text=question.question_text,
            marks=question.marks,
            subject_name=question.module.subject.name,
            module_title=question.module.title,
            topic_title=question.topic.title if question.topic else "",
            exam_question=question
        )
        return JsonResponse({
            'success': True,
            'question_id': question.id,
            'question_text': question.question_text,
            'marks': question.marks,
            'blueprint': blueprint
        })
    except Exception as e:
        logger.exception("Error generating blueprint for question %s: %s", question_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def generate_topic_blueprint_view(request, topic_id):
    """
    AJAX endpoint to generate a full VTU Answer Blueprint for any given syllabus topic.
    Allows instant preparation for topics directly from the syllabus tree.
    """
    topic = get_object_or_404(
        Topic.objects.select_related('module__subject'),
        id=topic_id
    )
    raw_marks = request.GET.get('marks') or request.POST.get('marks') or '10'
    try:
        marks = int(raw_marks)
    except (ValueError, TypeError):
        marks = 10

    # Check if this topic already has an ExamQuestion with a blueprint
    existing_eq = topic.exam_questions.filter(answer_blueprint__isnull=False).first()
    if existing_eq and existing_eq.answer_blueprint:
        return JsonResponse({
            'success': True,
            'topic_id': topic.id,
            'topic_title': topic.title,
            'question_id': existing_eq.id,
            'blueprint': existing_eq.answer_blueprint
        })

    question_prompt = f"Explain {topic.title} in detail with neat labeled diagrams, working mechanism, and practical applications. [{marks}M]"
    try:
        blueprint = generate_vtu_answer_blueprint(
            question_text=question_prompt,
            marks=marks,
            subject_name=topic.module.subject.name,
            module_title=topic.module.title,
            topic_title=topic.title
        )
        return JsonResponse({
            'success': True,
            'topic_id': topic.id,
            'topic_title': topic.title,
            'question_id': None,
            'blueprint': blueprint
        })
    except Exception as e:
        logger.exception("Error generating topic blueprint for %s: %s", topic_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def get_module_cheat_sheet_view(request, module_id):
    """
    AJAX endpoint to retrieve or generate the 1-page revision cheat sheet for a module.
    """
    module = get_object_or_404(Module.objects.select_related('subject'), id=module_id)
    force_refresh = (request.GET.get('refresh') == '1' or request.POST.get('refresh') == '1')
    try:
        data = generate_module_cheat_sheet(module, force_refresh=force_refresh)
        return JsonResponse({
            'success': True,
            'module_id': module.id,
            'cheat_sheet': data
        })
    except Exception as e:
        logger.exception("Error generating cheat sheet for module %s: %s", module_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def get_module_quiz_view(request, module_id):
    """
    AJAX endpoint to retrieve or generate the 5-minute Micro-Quiz and Flashcards for a module.
    """
    module = get_object_or_404(Module.objects.select_related('subject'), id=module_id)
    force_refresh = (request.GET.get('refresh') == '1' or request.POST.get('refresh') == '1')
    try:
        data = generate_module_quiz(module, force_refresh=force_refresh)
        return JsonResponse({
            'success': True,
            'module_id': module.id,
            'quiz_data': data
        })
    except Exception as e:
        logger.exception("Error generating quiz for module %s: %s", module_id, e)
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


