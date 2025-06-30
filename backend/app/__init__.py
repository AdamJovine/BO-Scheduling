from flask import Flask, request  # Add 'request' to the import
from config import Config
from .extensions import db, cors, migrate

# You already have cors imported
from .helpers.routes import helpers_bp
from .slider_survey.routes import survey_bp
from .pinned.routes import pinned_bp

def create_app(config_class=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)

    # init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Configure CORS to allow your React app
    cors.init_app(app,
                  origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                  supports_credentials=True,
                  allow_headers=["Content-Type", "Authorization"],
                  methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])

    # Add request logging (the function causing the error)
    @app.before_request
    def log_request_info():
        print(f"🔍 INCOMING REQUEST: {request.method} {request.url}")

    # register blueprints with URL prefixes
    app.register_blueprint(helpers_bp, url_prefix='/api')
    app.register_blueprint(survey_bp, url_prefix='/api')
    app.register_blueprint(pinned_bp, url_prefix='/api')

    return app