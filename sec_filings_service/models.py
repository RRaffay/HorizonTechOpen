from django.db import models


class SECFiling(models.Model):
    accessionNo = models.CharField(max_length=100, unique=True, default="Not found")
    stock = models.ForeignKey("user_service.Stock", on_delete=models.CASCADE)
    cik = models.CharField(max_length=15)
    ticker = models.CharField(max_length=10)
    companyName = models.CharField(max_length=200)
    companyNameLong = models.CharField(max_length=500)
    formType = models.CharField(max_length=20)
    description = models.TextField()
    linkToFilingDetails = models.URLField()
    linkToTxt = models.URLField()
    linkToHtml = models.URLField()
    linkToXbrl = models.URLField(blank=True, null=True)
    filedAt = models.DateTimeField()
    periodOfReport = models.DateField(blank=True, null=True)
    effectivenessDate = models.DateField(blank=True, null=True)
    registrationForm = models.CharField(max_length=20, blank=True, null=True)
    referenceAccessionNo = models.CharField(max_length=100, blank=True, null=True)

    # Other fields can be added as required

    def __str__(self):
        return self.ticker + " - " + self.formType + " - " + self.description
