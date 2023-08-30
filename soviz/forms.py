from django import forms

class RepoForm(forms.Form):
    repo = forms.CharField(label='Github Link', max_length=1000)
