from django.contrib import admin
from .models import Subject, Module, Topic, TopicKeyword, ModuleNote, TopicProgress, GenerationJob
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
    list_display = ('title', 'module', 'get_extension', 'get_file_size', 'uploaded_at')
    list_filter = ('module__subject', 'module')
    search_fields = ('title', 'description')

    def get_extension(self, obj):
        try:
            return obj.file_extension
        except Exception:
            return "FILE"
    get_extension.short_description = "Ext"

    def get_file_size(self, obj):
        try:
            return obj.file_size_display
        except Exception:
            return "Unknown"
    get_file_size.short_description = "Size"


@admin.register(TopicProgress)
class TopicProgressAdmin(admin.ModelAdmin):
    list_display = ('topic', 'status', 'updated_at')
    list_filter = ('status', 'topic__module__subject')


@admin.register(GenerationJob)
class GenerationJobAdmin(admin.ModelAdmin):
    list_display = ('id', 'subject_name', 'subject_code', 'status', 'progress', 'attempts', 'created_at', 'finished_at')
    list_filter = ('status', 'created_at')
    search_fields = ('subject_name', 'subject_code', 'id')
    readonly_fields = ('id', 'created_at', 'started_at', 'finished_at', 'locked_at', 'locked_by', 'claim_token', 'session_key')
