import os
import time
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

basedir = os.path.abspath(os.path.dirname(__file__))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_connection(database_url, max_retries=5):
    """
    Test database connection with retries and exponential backoff
    Returns True if connection successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Testing database connection, attempt {attempt + 1}/{max_retries}"
            )
            engine = create_engine(database_url, pool_pre_ping=True)

            # Test the connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")

            logger.info("Database connection successful!")
            engine.dispose()
            return True

        except (OperationalError, SQLAlchemyError) as e:
            logger.warning(
                f"Database connection attempt {attempt + 1} failed: {str(e)}"
            )

            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Database connection failed after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error during database connection: {str(e)}")
            return False

    return False


class Config:
    # Database configuration with retries and fallback
    database_url = None
    database_available = False

    # Check if we have RDS environment variables
    if all(
        [
            os.environ.get("RDS_HOSTNAME"),
            os.environ.get("RDS_USERNAME"),
            os.environ.get("RDS_PASSWORD"),
            os.environ.get("RDS_PORT"),
            os.environ.get("RDS_DB_NAME"),
        ]
    ):
        # Construct RDS connection string
        rds_url = f"postgresql://{os.environ.get('RDS_USERNAME')}:{os.environ.get('RDS_PASSWORD')}@{os.environ.get('RDS_HOSTNAME')}:{os.environ.get('RDS_PORT')}/{os.environ.get('RDS_DB_NAME')}"

        logger.info("RDS environment variables found, testing connection...")
        if test_database_connection(rds_url):
            database_url = rds_url
            database_available = True
            logger.info("Using RDS PostgreSQL database")
        else:
            logger.warning("RDS connection failed, falling back to SQLite")

    # Fallback to SQLite if RDS is not available or connection failed
    if not database_available:
        # Ensure the data directory exists
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)

        sqlite_url = f"sqlite:///{os.path.join(data_dir, 'app.db')}"

        # Test SQLite connection (should always work)
        if test_database_connection(sqlite_url, max_retries=1):
            database_url = sqlite_url
            database_available = True
            logger.info("Using SQLite database for local development/fallback")
        else:
            # Last resort - in-memory SQLite
            database_url = "sqlite:///:memory:"
            database_available = True
            logger.warning("Using in-memory SQLite database - data will not persist!")

    SQLALCHEMY_DATABASE_URI = database_url
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Additional database configuration for better reliability
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
        "connect_args": (
            {
                "connect_timeout": 60,
            }
            if "postgresql" in database_url
            else {}
        ),
    }

    # Other configuration
    SAVE_PATH = os.environ.get("SAVE_PATH") or os.path.join(basedir, "data")
    UI_PATH = os.environ.get("UI_PATH") or "/home/asj53/BOScheduling/UI/pages/plots"
    NUM_SLOTS = int(os.environ.get("NUM_SLOTS", 24))
    CORS_ORIGINS = ["*"]

    # Ensure SAVE_PATH directory exists
    os.makedirs(SAVE_PATH, exist_ok=True)

    print("SAVE_PATH", SAVE_PATH)
    print("UI_PATH", UI_PATH)
    print("DATABASE_URL", database_url)
    print("DATABASE_AVAILABLE", database_available)
