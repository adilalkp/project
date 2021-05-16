from django import forms
from django.forms import (CharField, TextInput, PasswordInput)


class LoginForm(forms.Form):
    username = CharField(max_length=30, widget=TextInput(attrs={'class': 'form__input', 'placeholder':'Username', 'id':'username'}))
    password = CharField(widget=PasswordInput(attrs={'class': 'form__input', 'placeholder':'Password', 'id':'password'}))
