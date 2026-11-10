from django.urls import path
from . import views

app_name = 'resources'

urlpatterns = [
    path('topics/<int:topic_id>/videos/', views.topic_videos, name='topic_videos'),
    path('topics/<int:topic_id>/articles/', views.topic_articles, name='topic_articles'),
]