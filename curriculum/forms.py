from django import forms
from .models import ModuleNote


class ModuleNoteForm(forms.ModelForm):
    class Meta:
        model = ModuleNote
        fields = ['title', 'file', 'description']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control bg-dark border-secondary-subtle text-light',
                'placeholder': 'e.g. Module 1 Handwritten Class Notes'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control bg-dark border-secondary-subtle text-light',
                'accept': '.pdf,.docx,.doc,.ppt,.pptx,.txt'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control bg-dark border-secondary-subtle text-light',
                'rows': 2,
                'placeholder': 'Optional notes, author name, or key topics covered'
            }),
        }