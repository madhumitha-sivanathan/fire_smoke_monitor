import os
import sys
import threading

from django.apps import AppConfig


def _should_start_polling() -> bool:
    """Start camera threads exactly once per server process.

    * runserver: only in the reloader's child process (RUN_MAIN) or with --noreload.
    * gunicorn / uvicorn / anything else: set FIREWATCH_START_POLLING=1.
    * Never for migrate, test, shell, etc.
    """
    if os.environ.get("FIREWATCH_START_POLLING") == "1":
        return True
    if "runserver" in sys.argv:
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv
    return False


class MonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "monitor"

    def ready(self):
        if not _should_start_polling():
            return

        from .rtsp import start_rtsp_polling

        # Delay slightly so Django finishes starting up before we touch the database.
        timer = threading.Timer(3.0, start_rtsp_polling)
        timer.daemon = True
        timer.start()
