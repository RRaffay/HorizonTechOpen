from django.contrib import admin

# Register your models here.
from .models import GDELTEvent, GDELTCall, GDELTEntry

admin.site.register(GDELTEvent)
admin.site.register(GDELTCall)
admin.site.register(GDELTEntry)
