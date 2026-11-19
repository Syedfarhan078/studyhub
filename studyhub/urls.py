"""
URL configuration for studyhub project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.views.static import serve
from django.contrib.sitemaps.views import sitemap
from curriculum.sitemaps import SubjectSitemap, StaticViewSitemap
from core.views import robots_txt

sitemaps = {
    'static': StaticViewSitemap,
    'subjects': SubjectSitemap,
}

urlpatterns = [
    path('admin/', admin.site.urls),
    path('robots.txt', robots_txt, name='robots_txt'),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}, name='django.contrib.sitemaps.views.sitemap'),
    path('subjects/', include('curriculum.urls')),
    path('resources/', include('resources.urls')),
    path('', include('core.urls')),
    # Media file serving fallback
    re_path(r'^media/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
]
