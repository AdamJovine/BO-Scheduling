from flask import Flask
from config import Config
from .extensions import db, cors, migrate
from .helpers.routes       import helpers_bp
from .slider_survey.routes import survey_bp
from .pinned.routes import pinned_bp

def create_app(config_class=Config):
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_class)

    # init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app)
    

    # register blueprints with URL prefixes
    
    app.register_blueprint(helpers_bp,      url_prefix='/api')
    app.register_blueprint(survey_bp,       url_prefix='/api/slider-configs')
    app.register_blueprint(pinned_bp,       url_prefix='/api')

    return app
