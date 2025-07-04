# test_database.py - Simple database test script
import os
import sqlite3
from sqlalchemy import create_engine, text

# Database path - use the Docker mounted path
DB_PATH = "/app/data/schedules.db"
DB_URL = f"sqlite:///{DB_PATH}"


def test_database_simple():
    """Simple SQLite test without SQLAlchemy"""
    print(f"🔍 Testing database at: {DB_PATH}")
    print(f"🔍 Database exists: {os.path.exists(DB_PATH)}")

    try:
        # Simple SQLite connection
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Test 1: List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"✅ Tables found: {tables}")

        # Test 2: Check if our main tables exist
        expected_tables = ["schedules", "metrics", "slots", "schedule_details"]
        for table in expected_tables:
            if table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✅ {table}: {count} rows")
            else:
                print(f"❌ {table}: table missing")

        # Test 3: Insert and read a test record
        cursor.execute(
            """
            INSERT OR REPLACE INTO schedules (schedule_id, display_name, max_slot) 
            VALUES ('test_schedule_001', 'Test Schedule', 24)
        """
        )

        cursor.execute(
            "SELECT * FROM schedules WHERE schedule_id = 'test_schedule_001'"
        )
        test_row = cursor.fetchone()
        print(f"✅ Test insert/select: {test_row}")

        conn.commit()
        conn.close()

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_database_sqlalchemy():
    """Test with SQLAlchemy (like your Flask app uses)"""
    print(f"\n🔍 Testing with SQLAlchemy...")

    try:
        engine = create_engine(DB_URL, echo=False)

        with engine.begin() as conn:
            # Test basic query
            result = conn.execute(
                text("SELECT COUNT(*) as total FROM sqlite_master WHERE type='table'")
            )
            table_count = result.fetchone().total
            print(f"✅ SQLAlchemy connection: {table_count} tables found")

            # Test specific table queries
            try:
                result = conn.execute(text("SELECT COUNT(*) FROM schedules"))
                count = result.fetchone()[0]
                print(f"✅ schedules table: {count} rows")
            except Exception as e:
                print(f"❌ schedules table error: {e}")

            try:
                result = conn.execute(text("SELECT COUNT(*) FROM metrics"))
                count = result.fetchone()[0]
                print(f"✅ metrics table: {count} rows")
            except Exception as e:
                print(f"❌ metrics table error: {e}")

            # Test a sample query like your app uses
            try:
                result = conn.execute(
                    text(
                        """
                    SELECT DISTINCT m.schedule_id 
                    FROM metrics m
                    WHERE m.schedule_id LIKE :prefix 
                    ORDER BY m.schedule_id
                    LIMIT 5
                """
                    ),
                    {"prefix": "20250620%"},
                )

                schedules = [row.schedule_id for row in result]
                print(f"✅ Sample schedule query: {schedules}")

            except Exception as e:
                print(f"❌ Sample query error: {e}")

        return True

    except Exception as e:
        print(f"❌ SQLAlchemy test failed: {e}")
        return False


def fix_slider_recordings_table():
    """Fix the typos in slider_recordings table"""
    print(f"\n🔧 Fixing slider_recordings table...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Drop the broken table
        cursor.execute("DROP TABLE IF EXISTS slider_recordings")

        # Create the correct table
        cursor.execute(
            """
            CREATE TABLE slider_recordings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                slider_key TEXT NOT NULL,
                value FLOAT,
                min_value FLOAT,
                max_value FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, slider_key)
            )
        """
        )

        # Also create slider_configs if missing
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS slider_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                thresholds TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

        print("✅ slider_recordings table fixed")
        return True

    except Exception as e:
        print(f"❌ Failed to fix table: {e}")
        return False


def create_sample_data():
    """Create some sample data for testing"""
    print(f"\n📊 Creating sample data...")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Sample schedule
        cursor.execute(
            """
            INSERT OR REPLACE INTO schedules (schedule_id, display_name, max_slot) 
            VALUES ('20250620_120000i1test', 'Sample Schedule 1', 24)
        """
        )

        # Sample metrics
        cursor.execute(
            """
            INSERT OR REPLACE INTO metrics (
                schedule_id, conflicts, quints, quads, four_in_five, 
                avg_max, semester
            ) VALUES (
                '20250620_120000i1test', 5, 2, 3, 1, 
                15.5, 'sp25'
            )
        """
        )

        # Sample slots
        for slot in range(1, 25):
            present = 1 if slot <= 20 else 0  # First 20 slots are used
            cursor.execute(
                """
                INSERT OR REPLACE INTO slots (schedule_id, slot_number, present) 
                VALUES (?, ?, ?)
            """,
                ("20250620_120000i1test", slot, present),
            )

        conn.commit()
        conn.close()

        print("✅ Sample data created")
        return True

    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Database Test Script")
    print("=" * 50)

    # Run tests
    test_database_simple()
    test_database_sqlalchemy()
    fix_slider_recordings_table()
    create_sample_data()

    # Final verification
    print("\n🔍 Final verification...")
    test_database_sqlalchemy()
