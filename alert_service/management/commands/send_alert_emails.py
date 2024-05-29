from django.core.management.base import BaseCommand
from django.core.mail import send_mail
from django.conf import settings
from alert_service.models import Alert, AlertStock, AreaOfInterest
from nlp_service.alert_processing import alert_analysis


import logging

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Send analysis emails to users"

    def get_analysis(self, stock_analysis_data):
        analysis = alert_analysis(stock_analysis_data)

        combined_analysis = "\n".join(
            f"{key}\n{value}" for key, value in analysis.items()
        )

        return combined_analysis

    def handle(self, *args, **kwargs):
        users_with_alerts = Alert.objects.filter(is_active=True).select_related("user")

        for alert in users_with_alerts:
            user = alert.user
            alert_stocks = AlertStock.objects.filter(alert=alert).prefetch_related(
                "areas_of_interest"
            )

            stock_analysis_data = {}
            for alert_stock in alert_stocks:
                areas = alert_stock.areas_of_interest.all()
                stock_analysis_data[alert_stock.stock.name] = [
                    area.description for area in areas
                ]

            # Passing in a dictionary of stock objects and their areas of interest as a list
            # TODO: Implement the get_analysis function
            # For each stock, get the analysis for an event of interest and then combine them
            analysis_result = self.get_analysis(stock_analysis_data)

            send_mail(
                "Your Stock Analysis",
                analysis_result,
                settings.DEFAULT_FROM_EMAIL,
                [user.email],
                fail_silently=False,
            )
            # TODO: Log the email being sent
            logger.info(f"Email sent to {user.email}")
