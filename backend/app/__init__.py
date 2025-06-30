from flask import Flask, request
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)

    # Skip all extensions for now - just test basic Flask
    @app.route("/")
    def root():
        return {"message": "Hello from App Runner!", "status": "ok"}

    @app.route("/health")
    def health():
        return {"status": "healthy"}

    return app
