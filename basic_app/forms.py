from django import forms
from django.forms import (CharField, TextInput, PasswordInput)


class LoginForm(forms.Form):
    username = CharField(max_length=30, widget=TextInput(attrs={'class': 'form-control', 'placeholder':'Username'}))
    password = CharField(widget=PasswordInput(attrs={'class': 'form-control', 'placeholder':'Password'}))
