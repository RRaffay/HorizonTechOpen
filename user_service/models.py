from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.conf import settings
from django.apps import apps


from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User


class Stock(models.Model):
    name = models.CharField(max_length=255)
    ticker = models.CharField(max_length=10, unique=True)
    info = models.TextField(default="")

    def __str__(self):
        return self.ticker


class Portfolio(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    stocks = models.ManyToManyField(
        Stock, blank=True
    )  # Allow blank for an empty portfolio

    def __str__(self):
        return f"{self.user.username}'s Portfolio"

    def remove_stock(self, stock):
        self.stocks.remove(stock)
        AlertStock = apps.get_model("alert_service", "AlertStock")
        AlertStock.objects.filter(alert__user=self.user, stock=stock).delete()
        PortfolioStock.objects.filter(portfolio=self, stock=stock).delete()


class PortfolioStock(models.Model):
    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)
    shares = models.IntegerField(default=0)
    purchase_price = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    area_of_interest = models.TextField(default="")
    processed_area_of_interest = models.TextField(default="")

    def save(self, *args, **kwargs):
        if (
            not self.pk
            and PortfolioStock.objects.filter(
                portfolio=self.portfolio, stock=self.stock
            ).exists()
        ):
            raise ValidationError(
                "A PortfolioStock entry with this portfolio and stock already exists."
            )
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.portfolio.user.username}'s {self.stock.ticker}"


class Profile(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    is_paying_customer = models.BooleanField(default=False)
    alert_status = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username}'s Profile"


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)
