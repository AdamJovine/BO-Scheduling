from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import logging

logger = logging.getLogger(__name__)

# Initialize extensions (but don't bind to app yet)
db = SQLAlchemy()
migrate = Migrate()

