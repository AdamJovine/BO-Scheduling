from flask import Flask, request
from config import Config
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app(config_class=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)

    # Initialize extensions with error handling
    db = None
    migrate = None
    cors = None

    try:
        from .extensions import db, cors, migrate

        # Try to initialize database
        try:
            db.init_app(app)
            logger.info("Database initialized successfully")

            # Try to initialize migrations
            try:
                migrate.init_app(app, db)
                logger.info("Database migrations initialized successfully")
            except Exception as e:
                logger.warning(f"Migration initialization failed: {e}")

        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
            db = None

        # Try to initialize CORS
        try:
            cors_origins = app.config.get("CORS_ORIGINS", ["*"])
            cors.init_app(
                app,
                origins=cors_origins,
                supports_credentials=True,
                allow_headers=["Content-Type", "Authorization"],
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            )

            logger.info("CORS initialized successfully")
        except Exception as e:
            logger.warning(f"CORS initialization failed: {e}")

    except ImportError as e:
        logger.warning(f"Failed to import extensions: {e}")
        logger.info("App will run without database and CORS")

    # Add request logging with error handling
    @app.before_request
    def log_request_info():
        try:
            print(f"🔍 INCOMING REQUEST: {request.method} {request.url}")
        except Exception as e:
            print(f"Request logging failed: {e}")

    # Register blueprints with error handling
    blueprints_to_register = [
        (".helpers.routes", "helpers_bp"),
        (".slider_survey.routes", "survey_bp"),
        (".pinned.routes", "pinned_bp"),
    ]

    for module_path, blueprint_name in blueprints_to_register:
        try:
            module = __import__(module_path, fromlist=[blueprint_name])
            blueprint = getattr(module, blueprint_name)
            app.register_blueprint(blueprint, url_prefix="/api")
            logger.info(f"Successfully registered blueprint: {blueprint_name}")
        except ImportError as e:
            logger.warning(
                f"Failed to import blueprint {blueprint_name} from {module_path}: {e}"
            )
        except AttributeError as e:
            logger.warning(
                f"Blueprint {blueprint_name} not found in {module_path}: {e}"
            )
        except Exception as e:
            logger.warning(f"Failed to register blueprint {blueprint_name}: {e}")

    # Add a basic health check route that doesn't require database
    @app.route("/health")
    def health_check():
        return {
            "status": "healthy",
            "database": "connected" if db else "disconnected",
            "message": "App is running",
        }

    # Add a basic root route
    @app.route("/")
    def root():
        return {
            "message": "BO-Scheduling API is running",
            "status": "ok",
            "database": "connected" if db else "disconnected",
        }

    # Create database tables if database is available
    if db:
        try:
            with app.app_context():
                db.create_all()
                logger.info("Database tables created/verified successfully")
        except Exception as e:
            logger.warning(f"Could not create database tables: {e}")

    logger.info("Flask app created successfully")
    return app
