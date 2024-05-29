from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from user_service.models import Stock, Portfolio  # Adjust the import path as necessary


class Frequency(models.TextChoices):
    DAILY = "D", ("Daily")
    WEEKLY = "W", ("Weekly")


class Alert(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    frequency = models.CharField(
        max_length=1, choices=Frequency.choices, default=Frequency.DAILY
    )
    last_sent = models.DateTimeField(auto_now=True)

    @property
    def is_active(self):
        return self.user.profile.alert_status

    def __str__(self):
        frequency_display = self.get_frequency_display()
        return f"Alert for {self.user.username} - {frequency_display}"


class AlertStock(models.Model):
    alert = models.ForeignKey(
        Alert, on_delete=models.CASCADE, related_name="alert_stocks"
    )
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)

    def __str__(self):
        return f"Alert for {self.stock.ticker}"

    def clean(self):
        # Check if the stock is in the user's portfolio
        if not Portfolio.objects.filter(
            user=self.alert.user, stocks=self.stock
        ).exists():
            raise ValidationError("The stock is not in the user's portfolio.")
        # Ensure that no more than 5 areas of interest are associated with this stock in the alert
        if self.areas_of_interest.count() > 5:
            raise ValidationError(
                "You can have a maximum of 5 areas of interest per stock."
            )

    class Meta:
        unique_together = ("alert", "stock")


class AreaOfInterest(models.Model):
    alert_stock = models.ForeignKey(
        AlertStock, on_delete=models.CASCADE, related_name="areas_of_interest"
    )
    description = models.CharField(max_length=255)

    def __str__(self):
        return self.description

    # Run full_clean on the model before saving to enforce the clean method
    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)
