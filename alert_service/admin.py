from django.contrib import admin
from .models import Alert, AlertStock, AreaOfInterest

admin.site.register(Alert)
admin.site.register(AlertStock)
admin.site.register(AreaOfInterest)
