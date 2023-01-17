from django.urls import path, include

from . import views     # it means - 'from all import views'
from django.contrib import admin

urlpatterns = [
    path('',views.index, name='index'),
    path('add',views.addition, name='add'),

]

# whenever 'localhost:8000' will be called, function named 'index' will be called that is present in 'views.py' file
# This will happen for all other routes as well