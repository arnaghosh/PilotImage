
from django.conf.urls import patterns,url
from django.views.generic import TemplateView

from rango import views

urlpatterns = patterns('',
                url(r'^$', views.index, name='index'),
				url(r'^ocr/', views.ocr, name='ocr'),
            )