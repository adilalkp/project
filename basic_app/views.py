from django.shortcuts import render, redirect
from django.contrib.auth import (authenticate, login, logout)
from django.contrib import messages

from .forms import LoginForm
# Create your views here.
def index(request):
    form = LoginForm()
    if request.method=='GET':
        return render(request, 'index.html', {'form':form})
    else:
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('basic_app:dashboard')
        
        messages.error(request, "Invalid username or password")
        return redirect('basic_app:index')

def dashboard(request):
    return render(request, 'dashboard.html')



