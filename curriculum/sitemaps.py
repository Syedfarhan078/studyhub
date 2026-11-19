from django.contrib.sitemaps import Sitemap
from django.urls import reverse
from .models import Subject


class SubjectSitemap(Sitemap):
    changefreq = 'weekly'
    priority = 0.9

    def items(self):
        return Subject.objects.all().order_by('order', 'name')

    def location(self, item):
        return item.get_absolute_url()


class StaticViewSitemap(Sitemap):
    priority = 1.0
    changefreq = 'daily'

    def items(self):
        return ['core:home']

    def location(self, item):
        return reverse(item)
