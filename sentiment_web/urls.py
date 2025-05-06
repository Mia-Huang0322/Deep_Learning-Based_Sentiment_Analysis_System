from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
urlpatterns = [
    # Admin panel routes
    path('admin_panel/', views.admin_dashboard, name='admin_dashboard'),
    path('admin_panel/users/', views.user_list, name='user_list'),
    path('admin_panel/feature_usage/', views.feature_usage, name='feature_usage'),
    path('admin_panel/page_views/', views.page_views, name='page_views'),
    path('text_sentiment/', views.text_sentiment, name='index'),
    path('visual_sentiment/', views.face_detection, name='face_detection'),
    path('combined_sentiment/', views.combined_sentiment, name='combined_sentiment'),
    path('send_code/', views.send_code, name='send_code'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('', views.home, name='home'),
    path('admin_panel/disable_user/<int:user_id>/', views.disable_user, name='disable_user'),
    path('admin_panel/delete_user/<int:user_id>/', views.delete_user, name='delete_user'),
    path('admin_login/', views.admin_login, name='admin_login'),



]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
