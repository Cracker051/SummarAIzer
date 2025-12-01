import os

DEBUG = os.getenv("DEBUG") == "True"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
