from django.urls import path

from basic_app import views
app_name = 'basic_app'

urlpatterns = [
    path('', views.index, name="index"),
    path('dashboard', views.dashboard, name="dashboard"),
    path('logout', views.logout_view, name="logout")
]