from django.urls import path
from . import views

app_name = 'weaviate_classifier'

urlpatterns = [
    path('', views.index, name='index'),
    path('classify/', views.classify_document, name='classify_document'),
    path('rebuild/', views.rebuild_database, name='rebuild_database'),
    path('add-files/', views.add_new_files, name='add_new_files'),
    path('stats/', views.get_db_stats, name='get_db_stats'),
    path('health/', views.health_check, name='health_check'),
]