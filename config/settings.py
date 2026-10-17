import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-insecure-key-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "monitor.apps.MonitorConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.template.context_processors.media",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
        # Camera threads and web requests write concurrently; wait instead of failing.
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.environ.get("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Fire / smoke model
# ---------------------------------------------------------------------------
# Put your trained YOLO weights at  <project>/models/model.pt
# (or point FIRE_SMOKE_MODEL_PATH at them with an environment variable).
FIRE_SMOKE_MODEL_PATH = Path(os.environ.get("FIRE_SMOKE_MODEL_PATH", BASE_DIR / "models" / "model.pt"))
FIRE_SMOKE_MODEL_CONF = 0.25   # detector confidence floor; alarm thresholds cannot go below this
FIRE_SMOKE_MODEL_IOU = 0.45
FIRE_SMOKE_MODEL_IMGSZ = 640

# ---------------------------------------------------------------------------
# Logging (the RTSP polling logger writes to the console and a daily-rotated file)
# ---------------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"rtsp": {"format": "[RTSP] %(asctime)s %(levelname)s %(message)s"}},
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "rtsp"},
        "file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "filename": str(LOG_DIR / "rtsp.log"),
            "when": "midnight",
            "backupCount": 1,
            "encoding": "utf-8",
            "formatter": "rtsp",
        },
    },
    "loggers": {"rtsp": {"handlers": ["console", "file"], "level": "INFO", "propagate": False}},
}
