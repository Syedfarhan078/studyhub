from django.urls import path
from . import views

app_name = 'curriculum'

urlpatterns = [
    path('generate-ai/', views.generate_subject_ai, name='generate_subject_ai'),
    path('generation-status/<str:task_id>/', views.generation_status, name='generation_status'),
    path('notes/<int:note_id>/', views.view_module_note, name='view_module_note'),
    path('notes/<int:note_id>/download/', views.download_module_note, name='download_module_note'),
    path('modules/<int:module_id>/upload-note/', views.upload_module_note, name='upload_module_note'),
    path('topics/<int:topic_id>/update-progress/', views.update_topic_progress, name='update_topic_progress'),
    path('<slug:slug>/', views.subject_detail, name='subject_detail'),
]
