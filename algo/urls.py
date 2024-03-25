from django.urls import path
from . import views

urlpatterns = [
    path('calculate-distances/', views.calculate_distances, name='calculate_distances'),
    path('tsp-solution/', views.tsp_solution, name='tsp_solution'),
    path('ant-system-solution/', views.ant_system_solution, name='ant_system_solution'),
    path('upload/', views.upload_and_read_excel, name='upload_excel'),
    path('ask/', views.ask_about_algorithms, name='ask_about_algorithms'),
     path('', views.index, name='index'),

]
