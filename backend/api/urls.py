from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file),
    path('conversate/', views.conversate),
]
