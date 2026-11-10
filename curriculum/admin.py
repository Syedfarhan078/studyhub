from django.contrib import admin
from .models import Subject, Module, Topic, TopicKeyword, ModuleNote, TopicProgress
# Register your models here.

class TopicKeywordInline(admin.TabularInline):
    model = TopicKeyword
    extra = 3

@admin.register(Subject)
class SubjectAdmin(admin.ModelAdmin):
    list_display = ('code', 'name', 'order', 'module_count')
    prepopulated_fields = {'slug': ('name',)}
    search_fields = ('name', 'code')
    ordering = ('order', 'name')

    def module_count(self, obj):
        return obj.modules.count()
    module_count.short_description = "Modules"

@admin.register(Module)
class ModuleAdmin(admin.ModelAdmin):
    list_display = ('subject', 'module_number', 'title', 'topic_count')
    list_filter = ('subject',)
    search_fields = ('title', 'subject__name', 'subject__code')
    ordering = ('subject', 'module_number')
    def topic_count(self, obj):
        return obj.topics.count()
    topic_count.short_description = "Topics"

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    list_display = ('title', 'module', 'order', 'keyword_count')
    list_filter = ('module__subject', 'module')
    search_fields = ('title', 'description', 'keywords__keyword')
    prepopulated_fields = {'slug': ('title',)}
    inlines = [TopicKeywordInline]
    def keyword_count(self, obj):
        return obj.keywords.count()
    keyword_count.short_description = "Keywords"

@admin.register(ModuleNote)
class ModuleNoteAdmin(admin.ModelAdmin):
    list_display = ('title', 'module', 'file_extension', 'file_size_display', 'uploaded_at')
    list_filter = ('module__subject', 'module')
    search_fields = ('title', 'description')


@admin.register(TopicProgress)
class TopicProgressAdmin(admin.ModelAdmin):
    list_display = ('topic', 'status', 'updated_at')
    list_filter = ('status', 'topic__module__subject')