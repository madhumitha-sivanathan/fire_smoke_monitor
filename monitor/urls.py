from django.urls import path

from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("devices/add/", views.device_create, name="device_create"),
    path("devices/<int:pk>/delete/", views.device_delete, name="device_delete"),
    path("devices/<int:pk>/rules/", views.configure_rules, name="configure_rules"),
    path("api/devices/", views.api_devices, name="api_devices"),
    path("api/devices/<int:pk>/live/", views.api_live, name="api_live"),
    path("api/devices/<int:pk>/predictions/", views.api_predictions, name="api_predictions"),
]
