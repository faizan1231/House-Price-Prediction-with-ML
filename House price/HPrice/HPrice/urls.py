from django.contrib import admin
from django.urls import path
from .views import predict_price

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", predict_price, name="predict_price"),
]

