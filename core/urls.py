from django.urls import path
from . import views

app_name = 'core'

urlpatterns = [
    path('', views.home, name='home'),
    path('seed-data/', views.seed_data, name='seed_data'),
]
