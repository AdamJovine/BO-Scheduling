# backend/app/connection.py - FIXED WITH CORRECT PATH

from sqlalchemy import create_engine, text
import os


class DatabaseConnection:
    _engine = None

    @classmethod
    def get_database_path(cls):
        if os.path.exists("/app"):  # Docker environment
            db_path = "sqlite:////app/schedules.db"
        else:  # Local development
            db_path = "sqlite:////app/schedules.db"
        return db_path

    @classmethod
    def check_connection(cls):
        """Actually test the connection"""
        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"❌ Connection check failed: {e}")
            return False

    @classmethod
    def table_exists(cls, table_name):
        """Check if a specific table exists"""
        try:
            result = cls.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name = :table_name",
                {"table_name": table_name},
                fetch_all=False,
            )
            return result is not None
        except:
            return False

    @classmethod
    def get_engine(cls):
        """Get the correct SQLAlchemy engine for schedules.db"""
        if cls._engine is None:
            db_url = cls.get_database_path()
            print(f"🔧 Using database: {db_url}")

            cls._engine = create_engine(db_url, echo=False)

            # Test the connection immediately
            try:
                with cls._engine.connect() as conn:
                    # Check if database has any tables
                    result = conn.execute(
                        text("SELECT name FROM sqlite_master WHERE type='table'")
                    )
                    tables = [row[0] for row in result.fetchall()]

                    if tables:
                        print(f"✅ Connected to database with tables: {tables}")

                        # Show row counts
                        for table in tables:
                            try:
                                count_result = conn.execute(
                                    text(f"SELECT COUNT(*) FROM {table}")
                                )
                                count = count_result.scalar()
                                print(f"  📊 {table}: {count} rows")
                            except:
                                pass
                    else:
                        print("⚠️ Connected to EMPTY database - no tables found!")

            except Exception as e:
                print(f"❌ Error connecting to database: {e}")
                raise

        return cls._engine

    @classmethod
    def execute_query(cls, query, params=None, fetch_all=True):
        """Execute a query safely and return results for SELECT queries"""
        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                # Check if this is a SELECT query that returns rows
                if query.strip().upper().startswith("SELECT"):
                    # For SELECT queries, fetch and return results
                    if fetch_all:
                        return result.fetchall()
                    else:
                        return result.fetchone()
                else:
                    # For INSERT, UPDATE, DELETE queries, just commit and return None
                    conn.commit()
                    return None

        except Exception as e:
            print(f"❌ Query failed: {e}")
            print(f"   Query: {query}")
            print(f"   Params: {params}")
            raise

    @classmethod
    def execute_insert(cls, query, params=None):
        """Execute an INSERT query and return number of affected rows"""
        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                conn.commit()  # Commit the transaction
                return result.rowcount
        except Exception as e:
            print(f"❌ Insert failed: {e}")
            print(f"   Query: {query}")
            print(f"   Params: {params}")
            raise

    @classmethod
    def execute_delete(cls, query, params=None):
        """Execute a DELETE query and return number of affected rows"""
        try:
            engine = cls.get_engine()
            with engine.connect() as conn:
                if params:
                    result = conn.execute(text(query), params)
                else:
                    result = conn.execute(text(query))

                conn.commit()  # Commit the transaction
                return result.rowcount
        except Exception as e:
            print(f"❌ Delete failed: {e}")
            print(f"   Query: {query}")
            print(f"   Params: {params}")
            raise


# Test connection when module is imported
print("🚀 Testing database connection on import...")
try:
    if DatabaseConnection.check_connection():
        print("✅ Database connection verified on import")
    else:
        print("❌ Database connection failed on import")
except Exception as e:
    print(f"❌ Import test failed: {e}")
