from flask_migrate import upgrade
from app import create_app

if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        upgrade()
    print("Database migrations completed")
