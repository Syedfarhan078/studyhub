import logging
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from .models import Subject, Module, Topic, TopicProgress
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
    Accepts subject name, course code, and optional syllabus text.
    Calls Gemini API with exponential backoff and fallback model support,
    then redirects directly to the new subject's interactive detail page.
    """
    if request.method == 'POST':
        subject_name = request.POST.get('subject_name', '').strip()
        subject_code = request.POST.get('subject_code', '').strip()
        syllabus_text = request.POST.get('syllabus_text', '').strip()

        if not subject_name:
            messages.error(request, "Please enter a subject name to generate the syllabus.")
            return redirect('/')

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