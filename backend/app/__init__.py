# app/__init__.py - USE SAME DATABASE PATH

from flask import Flask
from .extensions import db
from pathlib import Path
from sqlalchemy import text
from .connection import DatabaseConnection
from flask_cors import CORS


def create_app():

    app = Flask(__name__)

    # Use the SAME path as your connection.py
    import os

    if os.path.exists("/app"):  # Docker environment
        db_path = "sqlite:////app/schedules.db"
    else:  # Local development
        db_path = "sqlite:////app/schedules.db"

    print(f"🔧 Flask using database: {db_path}")
    container_data_dir = Path(__file__).resolve().parents[1] / "data"
    app.config["SQLALCHEMY_DATABASE_URI"] = db_path
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    CORS(app)
    # Init database
    db.init_app(app)

    # Import models so they exist
    from . import models

    # Create tables if they don't exist
    with app.app_context():
        # DatabaseConnection.execute_query("DROP TABLE IF EXISTS slider_recordings")
        db.create_all()
        print("✅ Database ready")

        # Check what tables exist
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"📋 Flask sees tables: {tables}")

    # Register your routes
    from .helpers.routes import helpers_bp
    from .slider_survey.routes import survey_bp
    from .pinned.routes import pinned_bp
    from .upload.routes import upload_bp

    app.register_blueprint(helpers_bp, url_prefix="/api")
    app.register_blueprint(survey_bp, url_prefix="/api")
    app.register_blueprint(pinned_bp, url_prefix="/api")
    app.register_blueprint(upload_bp, url_prefix="/api")

    return app

    @app.before_request
    def log_request():
        print(f"🔍 Request: {request.method} {request.url}")
        print(f"🔍 Headers: {dict(request.headers)}")
