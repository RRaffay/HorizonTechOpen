# Create your views here.
from django.contrib.auth import login
from .forms import CustomUserCreationForm  # make sure to import the custom form
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Stock, Portfolio, PortfolioStock, Profile


from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from django.contrib import messages

from django.core.mail import send_mail
from django.contrib.auth.tokens import default_token_generator
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes

from django.utils.http import urlsafe_base64_decode
from django.contrib.auth.models import User
from django.conf import settings
from .forms import FeedbackForm, UserForm, ProfileForm


import logging

logger = logging.getLogger(__name__)


def verify_email(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode()
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        # Now we can log the user in
        user.backend = "django.contrib.auth.backends.ModelBackend"
        login(request, user)
        return redirect("home")  # Redirect to home page or other success page
    else:
        return render(
            request, "email_verification_invalid.html"
        )  # Verification link was invalid


def signup(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False  # Prevent the user from logging in until they've verified their email
            user.save()
            logger.info(f"User created: {user}")
            # Send verification email
            current_site = get_current_site(request)
            subject = "Verify your account"
            message = f"""
Hi {user.username},

Please confirm your email address by clicking the link below:

https://{current_site.domain}/verify-email/{urlsafe_base64_encode(force_bytes(user.pk))}/{default_token_generator.make_token(user)}

Thank You.
            """
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,  # Use the default FROM email set in your settings
                [user.email],
                fail_silently=False,
            )

            return render(
                request, "user_service/email_verification_sent.html"
            )  # Redirect to a page that instructs the user to check their email
        else:
            logger.info(f"User creation failed: {form.errors}")
    else:
        form = CustomUserCreationForm()
    return render(request, "user_service/signup.html", {"form": form})


@login_required
def home(request):
    # Create User's profile if it doesn't exist
    _ = Profile.objects.get_or_create(user=request.user)

    portfolio, created = Portfolio.objects.get_or_create(user=request.user)
    return render(request, "user_service/home.html", {"portfolio": portfolio})


@login_required
def add_stock(request):
    if request.method == "POST":
        stock_id = request.POST.get("stock_id")
        area_of_interest = request.POST.get("area_of_interest")

        # Get the stock and the user's portfolio
        stock = get_object_or_404(Stock, pk=stock_id)
        portfolio = get_object_or_404(Portfolio, user=request.user)

        # Create a new PortfolioStock entry
        try:
            portfolio_stock = PortfolioStock.objects.create(
                portfolio=portfolio,
                area_of_interest=area_of_interest,
                stock=stock,
            )
            portfolio_stock.save()
            portfolio.stocks.add(stock)
            portfolio.save()
            messages.success(request, "Stock added to your portfolio!")
            logging.info(f"Stock added to portfolio: {stock}")
        except Exception as e:
            # Handle specific exceptions if needed, e.g. IntegrityError
            logging.error(f"Failed to add stock: {e}")
            messages.error(request, f"Failed to add stock: {e}")

        return redirect("home")
    else:
        available_stocks = Stock.objects.all().exclude(portfolio__user=request.user)
        return render(
            request, "user_service/add_stock.html", {"stocks": available_stocks}
        )


@login_required
def remove_stock(request, stock_id):
    stock = Stock.objects.get(id=stock_id)
    portfolio = Portfolio.objects.get(user=request.user)
    portfolio.remove_stock(stock)
    return redirect("home")


@login_required
def feedback(request):
    if request.method == "POST":
        form = FeedbackForm(request.POST)
        if form.is_valid():
            logger.info(f"Feedback received: {form.cleaned_data}")
            return redirect("/home")
    else:
        form = FeedbackForm()

    return render(request, "user_service/feedback.html", {"form": form})


@login_required
def account_management(request):
    if request.method == "POST":
        user_form = UserForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=request.user.profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect("home")
    else:
        user_form = UserForm(instance=request.user)
        profile_form = ProfileForm(instance=request.user.profile)
    return render(
        request,
        "user_service/account_management.html",
        {"user_form": user_form, "profile_form": profile_form},
    )
