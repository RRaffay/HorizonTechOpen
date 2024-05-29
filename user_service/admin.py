# Register your models here.
from django.contrib import admin
from .models import Stock, Portfolio, PortfolioStock, Profile

admin.site.register(Stock)
admin.site.register(Portfolio)
admin.site.register(PortfolioStock)
admin.site.register(Profile)
