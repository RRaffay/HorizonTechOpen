from django.contrib import admin

# Register your models here.
from .models import ProcessedSECData, ProcessedGDELTData

admin.site.register(ProcessedSECData)
admin.site.register(ProcessedGDELTData)
