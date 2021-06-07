from .ml_module import run_job
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from celery import shared_task


from django.contrib.auth import (authenticate, login, logout)
from django.contrib import messages
from time import sleep
from django.contrib.auth.models import User

#mail
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode

# from numpy.lib.arraysetops import ediff1d
from .forms import LoginForm
from .models import Video, Job, VehicleRecord
from pathlib import Path
import uuid
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent




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




@login_required
def dashboard(request):
    if request.method=="GET":
        return render(request, 'dashboard.html', {'post':0})
    else:
         username = str(request.user)
         video = request.FILES.get('video')
         job_name = request.POST.get('job_name')
         job_code = str(uuid.uuid4())
         filename = str(username) + "-" + job_code + ".mp4"
         video._name = filename
         obj = Video(owner=username, videofile=video, job_code=job_code)
         obj.save()
         job_obj = Job(owner=username, job_name=job_name, job_code=job_code, status="pending")
         job_obj.save()

         #get some stuff for mail
         current_site = get_current_site(request)
         domain = current_site.domain
         user_email = request.user.email

         #run
         run_job.delay(username, job_code, domain, user_email)

         return render(request, 'dashboard.html', {'post':1})



def list_jobs(request):
    username = str(request.user)
    jobs = Job.objects.filter(owner=username)
    return render(request, 'jobs.html', {'joblist':jobs, 'username':username})

def individual_job(request, job_code):
    vehi_records = VehicleRecord.objects.filter(job_code=job_code)
    nums = len(vehi_records)
    return render(request, 'job.html', {'vehicle_records':vehi_records, 'nums':nums})


# for styling
