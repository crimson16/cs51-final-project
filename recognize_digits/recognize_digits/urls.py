from django.conf.urls import patterns, include, url
from django.contrib import admin

urlpatterns = patterns('',
    # Examples:
    url(r'^$', 'recognize_digits.views.home', name='home'),
    # url(r'^data/', 'recognize_digits.views.get_data', name='get_data'),
    url(r'^predict/+', 'recognize_digits.views.predict', name='predict'),
    # url(r'^blog/', include('blog.urls')),
)
