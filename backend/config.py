import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config:
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
        # Use RDS
        SQLALCHEMY_DATABASE_URI = f"postgresql://{os.environ.get('RDS_USERNAME')}:{os.environ.get('RDS_PASSWORD')}@{os.environ.get('RDS_HOSTNAME')}:{os.environ.get('RDS_PORT')}/{os.environ.get('RDS_DB_NAME')}"
    else:
        # Use SQLite for local development
        SQLALCHEMY_DATABASE_URI = (
            f"sqlite:///{os.path.join(os.path.dirname(__file__), 'data', 'app.db')}"
        )

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SAVE_PATH = os.environ.get("SAVE_PATH") or os.path.join(basedir, "data")
    UI_PATH = "/home/asj53/BOScheduling/UI/pages/plots"  # os.environ.get('UI_PATH')   or os.path.join(basedir, 'ui')
    NUM_SLOTS = int(os.environ.get("NUM_SLOTS", 24))
    CORS_ORIGINS = ["*"]
    print("SAVE_PATH", SAVE_PATH)
    print("UI_PATH", UI_PATH)
