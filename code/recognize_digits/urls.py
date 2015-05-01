from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'recognize_digits.views.home', name='home'),
    url(r'^predict/+', 'recognize_digits.views.predict', name='predict'),
    
)
