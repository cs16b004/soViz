from django.urls import path
from . import views
urlpatterns = [
	path('',views.index,name = 'index'),
	path('repo/',views.get_repo,name  = 'repo'),
	path('repo/thanks/',views.th,name = 'thanks')
]
