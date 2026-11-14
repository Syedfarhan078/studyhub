import logging
import uuid
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.urls import reverse
from django.contrib import messages
from .models import Subject, Module, Topic, TopicProgress, GenerationJob
from .forms import ModuleNoteForm
from .services.ai_curriculum import (
    generate_curriculum_from_prompt,
    CurriculumGenerationError,
    GeminiTransientError,
    GeminiAuthError,
    GeminiInvalidRequestError,
    GeminiModelNotFoundError,
    CurriculumParseError,
    CurriculumValidationError,
)

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
                    'message': f"Subject '{existing_subject.name}' ({existing_subject.code}) is already available with {existing_subject.modules.count()} modules."
                })
            messages.info(request, f"Subject '{existing_subject.name}' already exists.")
            return redirect(redirect_url)

        # Background generation via PostgreSQL GenerationJob for interactive web client
        if is_ajax:
            user = request.user if request.user.is_authenticated else None
            session_key = ""
            if not user:
                if not request.session.session_key:
                    request.session.save()
                session_key = request.session.session_key or ""

            job = GenerationJob.objects.create(
                user=user,
                session_key=session_key,
                subject_name=subject_name,
                subject_code=subject_code,
                syllabus_text=syllabus_text,
                status=GenerationJob.Status.PENDING,
                progress=0,
                step=1,
                total_steps=4,
                progress_message=f"Job queued and awaiting worker pickup for '{subject_name}'."
            )
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
                syllabus_text=syllabus_text
            )
            messages.success(
                request,
                f"Syllabus for '{subject.name}' ({subject.code}) generated successfully with {subject.modules.count()} modules!"
            )
            return redirect('curriculum:subject_detail', slug=subject.slug)
        except GeminiTransientError as e:
            logger.warning("Transient error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "The AI curriculum service is temporarily experiencing high server demand. Please wait a few moments and try again."
            )
        except GeminiAuthError as e:
            logger.error("Authentication error generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "AI service authentication error. Please verify that your GEMINI_API_KEY is configured correctly."
            )
        except GeminiInvalidRequestError as e:
            logger.error("Invalid request generating curriculum for '%s': %s", subject_name, e)
            messages.error(request, f"Unable to process curriculum request: {e}")
        except GeminiModelNotFoundError as e:
            logger.error("Model not found generating curriculum for '%s': %s", subject_name, e)
            messages.error(
                request,
                "The configured AI model is not available. Please verify your Gemini model configuration."
            )
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
    if job.subject:
        redirect_url = reverse('curriculum:subject_detail', kwargs={'slug': job.subject.slug})

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
        'error': job.error_message or None,
    })


def subject_detail(request, slug):
    """
    Renders the syllabus view for a given subject.
    Prefetches modules, topics, keywords, notes, and progress records.
    Calculates overall syllabus completion metrics.
    """
    subject = get_object_or_404(
        Subject.objects.prefetch_related(
            'modules__topics__keywords',
            'modules__topics__progress',  # <-- Prefetches progress records
            'modules__notes'
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