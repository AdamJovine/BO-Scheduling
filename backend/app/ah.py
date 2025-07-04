from sqlalchemy import create_engine, text

# Check your LOCAL database (where you ran the scripts)
engine = create_engine("sqlite:///schedules.db", echo=False)

with engine.connect() as conn:
    # Check what tables exist
    result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
    tables = [row[0] for row in result.fetchall()]
    print(f"LOCAL DB tables: {tables}")

    # Check row counts
    for table in tables:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  {table}: {count} rows")
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
