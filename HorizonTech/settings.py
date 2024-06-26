"""
Django settings for HorizonTech project.

Generated by 'django-admin startproject' using Django 4.2.5.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os
from decouple import config


__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

SECRET_KEY = os.getenv("SECRET_KEY")


LANGCHAIN_TRACING_V2 = config("LANGCHAIN_TRACING_V2", default=False)
LANGCHAIN_ENDPOINT = config("LANGCHAIN_ENDPOINT", default="")
LANGCHAIN_API_KEY = config("LANGCHAIN_API_KEY", default="")
LANGCHAIN_PROJECT = config("LANGCHAIN_PROJECT", default="")

SEC_API_KEY = config("SEC_API_KEY", default="")

OPENAI_API_KEY = config("OPENAI_API_KEY", default="DummyKey")
COHERE_API_KEY = config("COHERE_API_KEY", default="")


GOOGLE_API_KEY = config("GOOGLE_API_KEY", default="")
GOOGLE_CSE_ID = config("GOOGLE_CSE_ID", default="")

ALPHAVANTAGE_API_KEY = config("ALPHAVANTAGE_API_KEY", default="")

CLIENT_ID = config("CLIENT_ID", default="")

CLIENT_SECRET = config("CLIENT_SECRET", default="")

AZURE_POSTGRESQL_CONNECTIONSTRING = config(
    "AZURE_POSTGRESQL_CONNECTIONSTRING", default=""
)
AZURE_REDIS_CONNECTIONSTRING = config("AZURE_REDIS_CONNECTIONSTRING", default="")


# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Using CORS to allow requests from React frontend

if "WEBSITE_HOSTNAME" in os.environ:
    DEBUG = False
    ALLOWED_HOSTS = [os.environ["WEBSITE_HOSTNAME"], "169.254.129.4"]

else:
    ALLOWED_HOSTS = []


SITE_ID = 1


CSRF_TRUSTED_ORIGINS = (
    [
        "https://" + os.environ["WEBSITE_HOSTNAME"],
    ]
    if "WEBSITE_HOSTNAME" in os.environ
    else []
)

# Application definition

EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "accounts@horizontech.app"
EMAIL_HOST_PASSWORD = config("GMAIL_PASSWORD", default="")
DEFAULT_FROM_EMAIL = "accounts@horizontech.app"


INSTALLED_APPS = [
    # Added for user_service
    "user_service.apps.UserServiceConfig",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "bootstrap5",
    # Added for stock_news_service
    "stock_news_service.apps.StockNewsServiceConfig",
    # added for SEC filings service
    "sec_filings_service.apps.SecFilingsServiceConfig",
    # Added for NLP service
    "nlp_service.apps.NlpServiceConfig",
    # Added for Retriever Processing
    "retriever_processing.apps.RetrieverProcessingConfig",
    # Added for Alert Service
    "alert_service.apps.AlertServiceConfig",
    # Added for allauth
    "django.contrib.sites",
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
]

SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "SCOPE": [
            "profile",
            "email",
        ],
        "AUTH_PARAMS": {
            "access_type": "online",
        },
    }
}

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django_ratelimit.middleware.RatelimitMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


ROOT_URLCONF = "HorizonTech.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # Not sure how this will work in production
        "DIRS": [os.path.join(BASE_DIR, "templates")],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]


WSGI_APPLICATION = "HorizonTech.wsgi.application"


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

SESSION_ENGINE = "django.contrib.sessions.backends.cache"
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"
STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")


conn_str = AZURE_POSTGRESQL_CONNECTIONSTRING

if conn_str == "":
    conn_str = config("AZURE_POSTGRESQL_CONNECTIONSTRING", default="")
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": "my_horizon",
            "HOST": "localhost",
            "USER": "horizonh",
            "PASSWORD": "Horizon123@",
            "PORT": "5432",
        }
    }

    CACHES = {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": "redis://127.0.0.1:6379/1",
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
            },
        }
    }

else:
    conn_str_params = {
        pair.split("=")[0]: pair.split("=")[1] for pair in conn_str.split(" ")
    }
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": conn_str_params["dbname"],
            "HOST": conn_str_params["host"],
            "USER": conn_str_params["user"],
            "PASSWORD": conn_str_params["password"],
        }
    }

    CACHES = {
        "default": {
            "BACKEND": "django_redis.cache.RedisCache",
            "LOCATION": AZURE_REDIS_CONNECTIONSTRING,
            "OPTIONS": {
                "CLIENT_CLASS": "django_redis.client.DefaultClient",
                "COMPRESSOR": "django_redis.compressors.zlib.ZlibCompressor",
            },
        }
    }

LOGIN_URL = "login"
LOGOUT_REDIRECT_URL = "login"


# Password validation
# https://docs.djangoproject.com/en/4.2/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = "static/"

STATICFILES_DIRS = [
    BASE_DIR / "static",
]

STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")


# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# settings.py
GOOGLE_CREDENTIALS_PATH = os.path.join(
    BASE_DIR, "stock_news_service", "gdelt-api-horizon-tech-8ca179ba5108.json"
)


# LOGGING = {
#     "version": 1,
#     "disable_existing_loggers": False,
#     "formatters": {
#         "verbose": {
#             "format": "{levelname} {asctime} {funcName} {message}",
#             "style": "{",
#         },
#     },
#     "handlers": {
#         "console": {
#             "class": "logging.StreamHandler",
#             "level": "INFO",
#             "formatter": "verbose",
#         },
#         "file": {
#             "level": "WARNING",
#             "class": "logging.FileHandler",
#             "filename": "errors.log",
#             "formatter": "verbose",
#         },
#     },
#     "root": {
#         "handlers": ["console", "file"],
#         "level": "DEBUG",
#     },
# }


AUTHENTICATION_BACKENDS = (
    "django.contrib.auth.backends.ModelBackend",
    "allauth.account.auth_backends.AuthenticationBackend",
)

LOGIN_DIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/"


USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
