import pytest
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from user_service.models import Stock, Portfolio, PortfolioStock


@pytest.mark.django_db
def test_stock_creation():
    stock = Stock.objects.create(name="Test Stock", ticker="TST")
    assert Stock.objects.count() == 1
    assert stock.name == "Test Stock"
    assert stock.ticker == "TST"


@pytest.mark.django_db
def test_portfolio_creation():
    user = User.objects.create_user(username="testuser", password="12345")
    portfolio = Portfolio.objects.create(user=user)
    assert Portfolio.objects.count() == 1
    assert portfolio.user == user


@pytest.mark.django_db
def test_portfolio_stock_creation():
    user = User.objects.create_user(username="testuser", password="12345")
    portfolio = Portfolio.objects.create(user=user)
    stock = Stock.objects.create(name="Test Stock", ticker="TST")
    portfolio_stock = PortfolioStock.objects.create(
        portfolio=portfolio, stock=stock, shares=10, purchase_price=100.00
    )
    assert PortfolioStock.objects.count() == 1
    assert portfolio_stock.portfolio == portfolio
    assert portfolio_stock.stock == stock
    assert portfolio_stock.shares == 10
    assert portfolio_stock.purchase_price == 100.00


@pytest.mark.django_db
def test_portfolio_stock_creation_duplicate():
    user = User.objects.create_user(username="testuser", password="12345")
    portfolio = Portfolio.objects.create(user=user)
    stock = Stock.objects.create(name="Test Stock", ticker="TST")
    PortfolioStock.objects.create(
        portfolio=portfolio, stock=stock, shares=10, purchase_price=100.00
    )
    with pytest.raises(ValidationError):
        PortfolioStock.objects.create(
            portfolio=portfolio, stock=stock, shares=10, purchase_price=100.00
        )
