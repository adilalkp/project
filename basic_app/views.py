from .ml_module import ml_run_final
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from django.contrib.auth import (authenticate, login, logout)
from django.contrib import messages
from time import sleep
from django.contrib.auth.models import User

# from numpy.lib.arraysetops import ediff1d
from .forms import LoginForm
from .models import Image
from pathlib import Path
import uuid
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# #ml imports









#views
def index(request):
    form = LoginForm()
    if request.method=='GET':
        return render(request, 'index.html', {'form':form})
    else:
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('basic_app:dashboard')
        messages.error(request, "Invalid username or password")
        return redirect('basic_app:index')

def logout_view(request):
    logout(request)
    return redirect('basic_app:index')

@login_required(login_url="basic_app:index")
def dashboard(request):
    if request.method=="GET":
        POST = 0
        return render(request, 'dashboard.html', {'post':POST})
    else:
        username = str(request.user)
        POST = 1 # for get post differentiation in template for conditional rendering
        images = request.FILES.getlist('images')
        for image in images:
            file_name = username + str(uuid.uuid4()) + ".jpg"
            image._name = file_name
            imgobj = Image(owner=username, imagefile=image)
            imgobj.save()        

            returned_list = ml_run_final(username)

        Image.objects.filter(owner=username).delete()
        return render(request, 'dashboard.html', {'plates':returned_list[0], 'crashed':returned_list[1], 'post':POST})





        




