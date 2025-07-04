import os, glob, re, json, subprocess
from functools import lru_cache
import pandas as pd
from typing import List
import logging

# from config import Config
from .connection import DatabaseConnection

logger = logging.getLogger(__name__)

# Configuration constants
SAVE_PATH = ""  # Config.SAVE_PATH
DATA_PATH = ""  # os.path.join(BASE_DIR, "data")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# now point at data/plots under the project root:
# UI_PATH = os.path.join(BASE_DIR, "app", "data", "plots")
NUM_SLOTS = 24  # Config.NUM_SLOTS
SEMESTER = os.environ.get("SEMESTER", "sp25")  # Default to 'sp25' if not set


import numpy as np
import pandas as pd
import math
import os


def extract_i_number(path: str) -> int:
    """Extract iteration number from filename."""
    fname = os.path.basename(path)
    m = re.search(r"i(\d+)", fname)
    if not m:
        raise ValueError(f"No 'i<digits>' segment found in {path!r}")
    return int(m.group(1))


def run_one_iteration(prefs: list):
    """
    Save prefs, build and submit a SLURM script.
    """
    try:
        prefs_path = os.path.join(SAVE_PATH, "pending_prefs.json")
        with open(prefs_path, "w") as f:
            json.dump(prefs, f)

        slurm_txt = f"""#!/bin/bash
#SBATCH -J aEIUU_1iter
#SBATCH -o /home/asj53/aEIUU_%j.out
#SBATCH -e /home/asj53/aEIUU_%j.err
#SBATCH --partition=frazier
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=200G
#SBATCH -t 23:00:00
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/asj53/BOScheduling/optimization

set -x
source ~/.bashrc
conda activate research_env

python -u run_EIUU.py \\
    --prefs {prefs_path} \\
    --n_iterations 1
"""
        script_path = os.path.join(SAVE_PATH, "submit_one_iter.slurm")
        with open(script_path, "w") as f:
            f.write(slurm_txt)

        res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Error submitting SLURM job: {res.stderr.strip()}")

        logger.info(f"SLURM job submitted successfully: {res.stdout.strip()}")
        return res.stdout.strip()

    except Exception as e:
        logger.error(f"Error in run_one_iteration: {e}")
        raise


def extract_schedule_id_from_filename(filepath: str) -> str:
    """
    Extract schedule ID from CSV or PNG filename.

    Handles formats like:
    - 20250623_180131i23thompson-33ae10f23812ffc1ed173820dd5dc015.csv
    - 20250624_045838i10thompson-3ec4820f0c1b823b7bb009fdf3b66f22_dist.png

    Returns: 20250623_180131i23thompson-33ae10f23812ffc1ed173820dd5dc015
    """
    filename = os.path.basename(filepath)

    # Remove file extensions
    if filename.endswith(".csv"):
        schedule_id = filename[:-4]  # Remove .csv
    elif filename.endswith(".png"):
        schedule_id = filename[:-4]  # Remove .png
        # Remove distribution suffix if present
        if schedule_id.endswith("_dist"):
            schedule_id = schedule_id[:-5]  # Remove _dist
        elif schedule_id.endswith("_distribution"):
            schedule_id = schedule_id[:-13]  # Remove _distribution
    else:
        # Just remove the extension
        schedule_id = os.path.splitext(filename)[0]

    return schedule_id


def debug_database_tables():
    """Debug function to inspect database tables and structure"""
    try:
        logger.info("=== DATABASE DEBUG INFO ===")

        # Check database connection
        if not DatabaseConnection.check_connection():
            logger.error("❌ Database connection failed!")
            return

        # 1. List all tables
        tables = DatabaseConnection.execute_query(
            """
            SELECT name FROM sqlite_master WHERE type='table'
        """
        )
        table_names = [row.name for row in tables]
        logger.info(f"Available tables: {table_names}")

        # 2. If metrics table exists, show its structure
        if "metrics" in table_names:
            logger.info("✓ metrics table EXISTS")

            # Get table schema
            columns_info = DatabaseConnection.execute_query(
                "PRAGMA table_info(metrics)"
            )
            columns = [
                (row[1], row[2]) for row in columns_info
            ]  # (column_name, data_type)
            logger.info(f"metrics table columns: {columns}")

            # Get row count
            count_result = DatabaseConnection.execute_query(
                "SELECT COUNT(*) as count FROM metrics", fetch_all=False
            )
            count = count_result.count if count_result else 0
            logger.info(f"metrics table row count: {count}")

            # Show sample data if any exists
            if count > 0:
                sample_rows = DatabaseConnection.execute_query(
                    "SELECT * FROM metrics LIMIT 3"
                )
                logger.info("Sample rows:")
                for i, row in enumerate(sample_rows):
                    logger.info(f"  Row {i+1}: {dict(row._mapping)}")

            # Show sample schedule_ids to help with debugging
            sample_schedules = DatabaseConnection.execute_query(
                """
                SELECT DISTINCT schedule_id FROM metrics 
                ORDER BY schedule_id LIMIT 5
            """
            )
            schedule_samples = [row.schedule_id for row in sample_schedules]
            logger.info(f"Sample schedule IDs: {schedule_samples}")

        else:
            logger.error("✗ metrics table does NOT exist")

        # 3. Check other important tables
        for table_name in ["schedule_details", "schedules", "slots"]:
            if table_name in table_names:
                count_result = DatabaseConnection.execute_query(
                    f"SELECT COUNT(*) as count FROM {table_name}", fetch_all=False
                )
                count = count_result.count if count_result else 0
                logger.info(f"✓ {table_name} table exists with {count} rows")
            else:
                logger.warning(f"✗ {table_name} table does NOT exist")

        # 4. Check database file info
        database_list = DatabaseConnection.execute_query("PRAGMA database_list")
        for db_info in database_list:
            logger.info(f"Connected database: {dict(db_info._mapping)}")

        logger.info("=== END DATABASE DEBUG ===")

    except Exception as e:
        logger.error(f"Database debug failed: {e}")


def get_schedule_files(
    date_prefix: str, metrics_dir: str = None, semester: str = SEMESTER
) -> List[str]:
    """Get schedule IDs from database that match the given date prefix."""
    logger.info(f"Getting schedules for prefix: {date_prefix}")
    debug_database_tables()

    try:
        if semester:
            # Get schedules that have metrics with the specified semester
            result = DatabaseConnection.execute_query(
                """
                SELECT DISTINCT m.schedule_id 
                FROM metrics m
                WHERE m.schedule_id LIKE :prefix 
                AND m.semester = :semester
                ORDER BY m.schedule_id
            """,
                {"prefix": f"{date_prefix}_%", "semester": semester},
            )
        else:
            # Get all schedules from metrics, regardless of semester
            result = DatabaseConnection.execute_query(
                """
                SELECT DISTINCT schedule_id 
                FROM metrics
                WHERE schedule_id LIKE :prefix 
                ORDER BY schedule_id
            """,
                {"prefix": f"{date_prefix}_%"},
            )

        schedule_ids = [row.schedule_id for row in result]
        logger.info(
            f"Found {len(schedule_ids)} schedules from database matching prefix '{date_prefix}_'"
        )

        if not schedule_ids:
            # Debug: show what schedules actually exist
            all_schedules = DatabaseConnection.execute_query(
                """
                SELECT DISTINCT schedule_id FROM metrics 
                ORDER BY schedule_id LIMIT 10
            """
            )
            sample_ids = [row.schedule_id for row in all_schedules]
            logger.warning(f"No schedules found for prefix '{date_prefix}_'")
            logger.warning(f"Sample existing schedule IDs: {sample_ids}")

            # Check if it's a semester issue
            if semester:
                no_semester_result = DatabaseConnection.execute_query(
                    """
                    SELECT DISTINCT schedule_id 
                    FROM metrics
                    WHERE schedule_id LIKE :prefix 
                    ORDER BY schedule_id
                """,
                    {"prefix": f"{date_prefix}_%"},
                )
                no_semester_ids = [row.schedule_id for row in no_semester_result]
                if no_semester_ids:
                    logger.warning(
                        f"Found {len(no_semester_ids)} schedules for prefix without semester filter"
                    )
                    logger.warning("This might be a semester mismatch issue")

        return schedule_ids

    except Exception as e:
        logger.error(f"Database query failed: {e}")
        if not DatabaseConnection.table_exists("metrics"):
            logger.error("❌ Metrics table does not exist! Wrong database?")
        return []


@lru_cache(maxsize=1)
def generate_plots_for_files(date_prefix: str):
    """Generate missing schedule and distribution plots for all files matching prefix."""
    try:
        schedule_ids = get_schedule_files(date_prefix)
        logger.info(
            f"Generating plots for {len(schedule_ids)} schedules: {schedule_ids}"
        )

        for schedule_id in schedule_ids:
            # Check and generate regular schedule plot
            # if not plot_exists_on_disk(schedule_id):
            #    logger.info(f"Generating schedule plot for {schedule_id}")
            #    # Note: You'll need to import get_plot or implement the plotting logic
            #    get_plot(f"{schedule_id}.csv", schedule_id)
            #
            # Check and generate distribution plot
            # if not plot_exists_on_disk(schedule_id, "_dist"):
            #    logger.info(f"Generating distribution plot for {schedule_id}")
            #    # Note: You'll need to import last_day or implement the distribution plotting logic
            #    last_day(f"{schedule_id}.csv", schedule_id)

            logger.info(f"you should not be here: {date_prefix}")

    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise


@lru_cache(maxsize=1)
def load_schedule_data_basic(date_prefix: str) -> list[dict]:
    """Load schedule data from database instead of CSV files."""
    try:
        schedule_ids = get_schedule_files(date_prefix)

        if not schedule_ids:
            logger.warning(f"No schedules found for prefix: {date_prefix}")
            return []

        logger.info(f"Loading data for {len(schedule_ids)} schedules from database...")

        # Column mappings from database to expected format
        param_cols = [
            "size_cutoff",
            "reserved",
            "num_blocks",
            "large_block_size",
            "large_exam_weight",
            "large_block_weight",
            "large_size_1",
            "large_cutoff_freedom",
        ]

        # Map database column names to display names for metrics
        metrics_mapping = {
            "conflicts": "conflicts",
            "quints": "quints",
            "quads": "quads",
            "four_in_five": "four in five slots",
            "three_in_four": "three in four slots",
            "two_in_three": "two in three slots",
            "singular_late": "singular late exam",
            "two_large_gap": "two exams, large gap",
            "avg_max": "avg_max",
        }

        data = []

        for schedule_id in schedule_ids:
            try:
                # Get metrics data from database
                result = DatabaseConnection.execute_query(
                    """
                    SELECT m.*, s.display_name, s.max_slot
                    FROM metrics m
                    LEFT JOIN schedules s ON m.schedule_id = s.schedule_id
                    WHERE m.schedule_id = :schedule_id
                """,
                    {"schedule_id": schedule_id},
                    fetch_all=False,
                )

                if not result:
                    logger.warning(f"No metrics found for {schedule_id}")
                    continue

                # Extract iteration number for display
                try:
                    idx = extract_i_number(schedule_id)
                    display = f"Schedule {idx}"
                except ValueError:
                    display = f"Schedule {schedule_id[:20]}..."  # Fallback display name

                # Build metrics dict with display names
                metrics = {}
                for db_col, display_name in metrics_mapping.items():
                    if hasattr(result, db_col) and getattr(result, db_col) is not None:
                        metrics[display_name] = getattr(result, db_col)

                # Add computed metrics
                triple_24h = getattr(result, "triple_in_24h", 0) or 0
                triple_same_day = getattr(result, "triple_in_same_day", 0) or 0
                evening_morning_b2b = getattr(result, "evening_morning_b2b", 0) or 0
                other_b2b = getattr(result, "other_b2b", 0) or 0

                metrics["reschedules"] = triple_24h + triple_same_day
                metrics["back_to_back"] = evening_morning_b2b + other_b2b

                # Build params dict
                params = {}
                for col in param_cols:
                    if hasattr(result, col) and getattr(result, col) is not None:
                        params[col] = getattr(result, col)

                # Get slot data from database
                slot_rows = DatabaseConnection.execute_query(
                    """
                    SELECT slot_number, present
                    FROM slots
                    WHERE schedule_id = :schedule_id
                    ORDER BY slot_number
                """,
                    {"schedule_id": schedule_id},
                )

                # Build columns dict from slot data
                columns = {
                    i: 0 for i in range(1, NUM_SLOTS + 1)
                }  # Initialize all slots to 0
                logger.debug(f"Initialized columns: {columns}")

                if slot_rows:
                    # Use database slot data
                    logger.debug(f"Found {len(slot_rows)} slot rows for {schedule_id}")
                    for slot_row in slot_rows:
                        slot_num = int(slot_row.slot_number)
                        logger.debug(
                            f"Processing slot {slot_num}, present: {slot_row.present}"
                        )
                        if (
                            slot_num in columns
                        ):  # Only include slots within NUM_SLOTS range
                            columns[slot_num] = 1 if slot_row.present else 0
                else:
                    # Fallback to CSV if no slot data in database
                    logger.warning(
                        f"No slot data in database for {schedule_id}, trying CSV fallback..."
                    )
                    try:
                        fname = f"{schedule_id}.csv"
                        csv_path = os.path.join(SAVE_PATH, "schedules", fname)
                        df_sched = pd.read_csv(csv_path)
                        slots = sorted(df_sched["slot"].unique())
                        columns = {
                            i: (1 if i in slots else 0) for i in range(1, NUM_SLOTS + 1)
                        }
                        logger.info(f"Loaded slot data from CSV for {schedule_id}")
                    except FileNotFoundError:
                        logger.warning(
                            f"Schedule CSV not found for {schedule_id}: {csv_path}"
                        )
                        # columns already initialized to all 0s
                    except Exception as e:
                        logger.warning(
                            f"Error reading schedule CSV for {schedule_id}: {e}"
                        )
                        # columns already initialized to all 0s

                data.append(
                    {
                        "name": display,
                        "basename": schedule_id,
                        "metrics": metrics,
                        "params": params,
                        "columns": columns,
                    }
                )

                logger.debug(f"Successfully processed schedule {schedule_id}")

            except Exception as e:
                logger.error(f"Error processing schedule {schedule_id}: {e}")
                continue

        logger.info(f"Successfully loaded {len(data)} schedules from database")
        return data

    except Exception as e:
        logger.error(f"Error in load_schedule_data_basic: {e}")
        return []


# Additional utility functions for database management
def get_database_info():
    """Get comprehensive database information for debugging."""
    try:
        info = {
            "connection_healthy": DatabaseConnection.check_connection(),
            "tables": [],
        }

        if info["connection_healthy"]:
            tables = DatabaseConnection.execute_query(
                """
                SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
            """
            )

            for table in tables:
                table_name = table.name
                count_result = DatabaseConnection.execute_query(
                    f"SELECT COUNT(*) as count FROM {table_name}", fetch_all=False
                )
                count = count_result.count if count_result else 0

                info["tables"].append({"name": table_name, "row_count": count})

        return info

    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"connection_healthy": False, "error": str(e)}


def verify_schedule_exists(schedule_id: str) -> bool:
    """Verify that a schedule exists in the database."""
    try:
        result = DatabaseConnection.execute_query(
            """
            SELECT COUNT(*) as count FROM metrics WHERE schedule_id = :schedule_id
        """,
            {"schedule_id": schedule_id},
            fetch_all=False,
        )

        return result and result.count > 0

    except Exception as e:
        logger.error(f"Error verifying schedule {schedule_id}: {e}")
        return False
