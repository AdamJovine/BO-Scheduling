import os

basedir = os.path.abspath(os.path.dirname(__file__))
print("basedir , ", basedir)


class Config:
    """Base configuration class"""

    # Database configuration - keep your existing setup
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL"
    ) or "sqlite:///" + os.path.join(basedir, "schedules.db")

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Add engine configuration for better connection management
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,  # Verify connections before use
        "pool_recycle": 3600,  # Recycle connections every hour
        "pool_timeout": 20,  # Timeout for getting connection
        "max_overflow": 10,  # Extra connections beyond pool_size
        "echo": False,  # Set to True for SQL debugging
    }

    # Application paths - keep your existing logic
    SAVE_PATH = os.environ.get("SAVE_PATH") or os.path.join(basedir, "data")

    # Application settings - keep your existing values
    NUM_SLOTS = int(os.environ.get("NUM_SLOTS", 24))
    CORS_ORIGINS = ["*"]
    SECRET_KEY = os.environ.get("SECRET_KEY") or "dev-secret-key-change-in-production"

    # Add debug flag
    DEBUG = False

    # Keep your existing prints
    print("SAVE_PATH:", SAVE_PATH)
    print("UI_PATH:", UI_PATH)
    print("Database:", SQLALCHEMY_DATABASE_URI)
