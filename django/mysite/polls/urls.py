from django.urls import path
from . import views # . means current directory

urlpatterns = [
	path('',views.index,name = 'index')
]