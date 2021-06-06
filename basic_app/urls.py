from django.urls import path

from basic_app import views
app_name = 'basic_app'

urlpatterns = [
    path('', views.index, name="index"),
    path('dashboard', views.dashboard, name="dashboard"),
    path('logout', views.logout_view, name="logout"),
    path('list_jobs', views.list_jobs, name="list_jobs"),
    path('job/<uuid:job_code>', views.individual_job, name="ind_job"),
    path('base', views.base),
    path('jobs', views.jobs),
    path('job', views.job)

]