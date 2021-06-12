from basic_app.ml.ml_module import run_job
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from celery import shared_task
from django.utils import timezone
from django.contrib.postgres.search import SearchVector


from django.contrib.auth import (authenticate, login, logout)
from django.contrib import messages
from time import sleep
from django.contrib.auth.models import User

#mail
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMultiAlternatives
from django.utils.encoding import force_bytes, force_text
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.http import HttpResponse
from django.template.loader import render_to_string
from weasyprint import HTML
import tempfile
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



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
         job_obj = Job(owner=username, job_name=job_name, job_code=job_code, status="pending", created_on=timezone.now())
         job_obj.save()

         #get some stuff for mail
         current_site = get_current_site(request)
         domain = current_site.domain
         user_email = request.user.email

         #run
         run_job.delay(username, job_code, domain, user_email)

         return render(request, 'dashboard.html', {'post':1})



def list_jobs(request):
    jobs = Job.objects.filter(owner=request.user).order_by('-created_on')
    return render(request, 'jobs.html', {'joblist':jobs})


def individual_job(request, job_code):
    if request.method=='GET':
        vehi_records = VehicleRecord.objects.filter(job_code=job_code)
        nums = len(vehi_records)
        if len==0:
            blank = 'yes'
        else:
            blank = 'no'
        return render(request, 'job.html', {'vehicle_records':vehi_records, 'nums':nums, 'job_code':job_code, 'blank':blank})
    else:   
        key = request.POST['key']
        vehi_records = VehicleRecord.objects.filter(job_code=job_code)
        vehi_records = vehi_records.annotate(search=SearchVector('license_plate','colour','vehicle_type'),).filter(search=key)
        nums = len(vehi_records)
        if nums==0:
            blank = 'yes'
        else:
            blank = 'no'
        return render(request, 'job.html', {'vehicle_records':vehi_records, 'nums':nums, 'job_code':job_code, 'blank':blank, 'key':key})



def report(request):
    return render(request, 'report.html')







def generate(request, job_code):

    vehi_records = VehicleRecord.objects.filter(job_code=job_code)
    html_string = render_to_string('report.html', {'records':vehi_records, 'username':request.user, 'email':request.user.email, 'date':timezone.now()})
    html = HTML(string=html_string, base_url=request.build_absolute_uri())
    result = html.write_pdf()

    # Creating http response
    response = HttpResponse(content_type='application/pdf;')
    response['Content-Disposition'] = 'inline; filename=list_people.pdf'
    response['Content-Transfer-Encoding'] = 'binary'
    with tempfile.NamedTemporaryFile(delete=True) as output:
        output.write(result)
        output.flush()
        output = open(output.name, 'rb')
        response.write(output.read())

    return response

def generate_key(request, job_code, key):
    vehi_records = VehicleRecord.objects.filter(job_code=job_code)
    vehi_records = vehi_records.annotate(search=SearchVector('license_plate','colour','vehicle_type'),).filter(search=key)
    dates = timezone.now()
    html_string = render_to_string('report.html', {'records':vehi_records, 'username':request.user, 'email':request.user.email, 'date':dates})
    html = HTML(string=html_string, base_url=request.build_absolute_uri())
    result = html.write_pdf()

    # Creating http response
    response = HttpResponse(content_type='application/pdf;')
    response['Content-Disposition'] = 'inline; filename=list_people.pdf'
    response['Content-Transfer-Encoding'] = 'binary'
    with tempfile.NamedTemporaryFile(delete=True) as output:
        output.write(result)
        output.flush()
        output = open(output.name, 'rb')
        response.write(output.read())

    return response