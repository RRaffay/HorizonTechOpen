from django import forms
from .models import Alert, AlertStock, AreaOfInterest


class AlertForm(forms.ModelForm):
    class Meta:
        model = Alert
        fields = ["frequency"]


class AlertStockForm(forms.ModelForm):
    class Meta:
        model = AlertStock
        fields = ["stock"]
        widgets = {"stock": forms.CheckboxSelectMultiple}


class AreaOfInterestForm(forms.ModelForm):
    description = forms.CharField(
        widget=forms.TextInput(
            attrs={
                "placeholder": "Add new area of interest. Be as specific as possible.",
                "class": "form-control",
            }
        )
    )

    class Meta:
        model = AreaOfInterest
        fields = ["description"]


class BaseAreaOfInterestFormSet(forms.BaseInlineFormSet):
    def clean(self):
        """Adds validation to check that no two areas of interest are the same."""
        if any(self.errors):
            return  # Don't bother validating the formset unless each form is valid on its own
        descriptions = []
        for form in self.forms:
            description = form.cleaned_data.get("description")
            if description in descriptions:
                raise forms.ValidationError("Areas of interest must be unique.")
            descriptions.append(description)
