from app import create_app, db
import os


def import_data():
    app = create_app()
    with app.app_context():
        # Create tables first
        db.create_all()

        # Import data from SQL file
        sql_file = os.path.join(os.path.dirname(__file__), "schedules.sql")
        if os.path.exists(sql_file):
            with open(sql_file, "r") as f:
                sql_commands = f.read()

            # Split and execute SQL commands
            for command in sql_commands.split(";"):
                if command.strip():
                    try:
                        db.engine.execute(command)
                    except Exception as e:
                        print(f"Skipping command due to error: {e}")

            db.session.commit()
            print("Data imported successfully")
        else:
            print("No SQL file found, creating empty database")


if __name__ == "__main__":
    import_data()
