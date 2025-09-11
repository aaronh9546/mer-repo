import os
from celery import Celery

# Create a Celery instance, getting the Redis URL from environment variables.
# This URL will be provided by Render.
celery_app = Celery(
    "worker",
    broker=os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
)

# This line is important so Celery can find the tasks defined in tasks.py
celery_app.conf.imports = ("tasks",)