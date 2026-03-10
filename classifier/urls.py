from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify_document, name='classify'),
    path('classify-zip/', views.classify_zip, name='classify_zip'),
    path('rebuild-db/', views.rebuild_database, name='rebuild_db'),
    path('add-files/', views.add_new_files, name='add_files'),
    path('db-stats/', views.get_db_stats, name='db_stats'),
]
