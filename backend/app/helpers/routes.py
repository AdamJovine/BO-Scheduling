from flask import Blueprint, jsonify, request, send_from_directory, abort, make_response
import os
import sys
from pathlib import Path
from flask import abort, send_from_directory, Blueprint, current_app
from ..connection import DatabaseConnection
import logging

# from data.connection import DatabaseConnection
from ..helpers_logic import (
    get_schedule_files,
    generate_plots_for_files,
    load_schedule_data_basic,
    run_one_iteration,
)

# Configure logging to output to stdout for Docker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

helpers_bp = Blueprint("helpers", __name__)


@helpers_bp.route("/files/<date_prefix>", methods=["GET"])
def files(date_prefix):
    """Get schedule files for a given date prefix"""
    try:
        result = get_schedule_files(date_prefix)
        # Use both logging and print for Docker visibility
        message = f"Retrieved files for date prefix: {date_prefix}"
        logger.info(message)
        print(f"[INFO] {message}", flush=True)
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error getting files for {date_prefix}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e)}), 500


@helpers_bp.route("/schedules/<date_prefix>", methods=["GET"])
def schedules(date_prefix):
    """Get schedule data for a given date prefix"""
    try:
        # Optionally generate plots (uncomment if needed)
        # generate_plots_for_files(date_prefix)

        result = load_schedule_data_basic(date_prefix)
        message = f"Retrieved schedule data for date prefix: {date_prefix}"
        logger.info(message)
        print(f"[INFO] {message}", flush=True)
        return jsonify(result)
    except Exception as e:
        error_msg = f"Error getting schedules for {date_prefix}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e)}), 500


@helpers_bp.route("/run", methods=["POST"])
def run():
    """Run one iteration of the scheduling algorithm"""
    try:
        prefs = request.json.get("prefs", []) if request.json else []
        message = f"Running iteration with {len(prefs)} preferences"
        logger.info(message)
        print(f"[INFO] {message}", flush=True)

        msg = run_one_iteration(prefs)
        success_msg = f"Iteration completed: {msg}"
        logger.info(success_msg)
        print(f"[INFO] {success_msg}", flush=True)
        return jsonify({"status": "submitted", "detail": msg}), 202

    except Exception as e:
        error_msg = f"Error running iteration: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"status": "error", "detail": str(e)}), 500


@helpers_bp.route("/images/<filename>")
def serve_image(filename):
    # build the path to <project>/app/static/plots
    UI_PATH = Path(current_app.static_folder) / "plots"
    FALLBACK_IMAGE = "fallback-image.png"

    # Log the image request
    request_msg = f"Image request for: {filename}"
    debug_msg = f"Looking in directory: {UI_PATH} (exists? {UI_PATH.exists()})"
    logger.info(request_msg)
    logger.debug(debug_msg)
    print(f"[INFO] {request_msg}", flush=True)
    print(f"[DEBUG] {debug_msg}", flush=True)

    # only allow .png
    if not filename.lower().endswith(".png"):
        warning_msg = f"Invalid file type requested: {filename}, serving fallback"
        logger.warning(warning_msg)
        print(f"[WARNING] {warning_msg}", flush=True)
        # Serve fallback instead of 404
        fallback_path = UI_PATH / FALLBACK_IMAGE
        if fallback_path.is_file():
            return send_from_directory(str(UI_PATH), FALLBACK_IMAGE)
        else:
            abort(404)  # Only abort if fallback also missing

    # static/plots must exist
    if not UI_PATH.is_dir():
        error_msg = f"Static plots folder missing: {UI_PATH}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        abort(500)

    # Check if requested file exists
    file_path = UI_PATH / filename
    if not file_path.is_file():
        warning_msg = f"File not found: {file_path}, attempting fallback"
        logger.warning(warning_msg)
        print(f"[WARNING] {warning_msg}", flush=True)

        # Try to serve fallback image instead
        fallback_path = UI_PATH / FALLBACK_IMAGE
        if fallback_path.is_file():
            fallback_msg = f"Serving fallback image for: {filename}"
            logger.info(fallback_msg)
            print(f"[INFO] {fallback_msg}", flush=True)
            return send_from_directory(str(UI_PATH), FALLBACK_IMAGE)
        else:
            # Fallback image is also missing - this is a configuration error
            error_msg = f"Both requested file and fallback image missing: {filename}, {FALLBACK_IMAGE}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}", flush=True)
            abort(404)

    # serve the requested file
    success_msg = f"Serving image: {filename}"
    logger.info(success_msg)
    print(f"[INFO] {success_msg}", flush=True)
    return send_from_directory(str(UI_PATH), filename)


@helpers_bp.route("/images/debug")
def debug_images():
    """Debug endpoint to list available images."""
    try:
        UI_PATH = Path(current_app.static_folder) / "plots"

        if not UI_PATH.exists():
            error_msg = f"Plots directory does not exist: {UI_PATH}"
            logger.error(error_msg)
            print(f"[ERROR] {error_msg}", flush=True)
            return jsonify(
                {
                    "error": error_msg,
                    "plots_dir": str(UI_PATH),
                }
            )

        png_files = [f for f in os.listdir(UI_PATH) if f.endswith(".png")]
        info_msg = f"Found {len(png_files)} PNG files in plots directory"
        logger.info(info_msg)
        print(f"[INFO] {info_msg}", flush=True)

        return jsonify(
            {
                "plots_dir": str(UI_PATH),
                "total_png_files": len(png_files),
                "sample_files": png_files[:20],  # Show first 20 files
                "directory_exists": True,
            }
        )

    except Exception as e:
        error_msg = f"Error in debug_images: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e), "plots_dir": str(UI_PATH)})


@helpers_bp.route("/download/schedules/<schedule_id>")
def download_schedule(schedule_id):
    """Download schedule data as CSV from schedule_details table."""
    try:
        info_msg = f"Downloading schedule data for {schedule_id}"
        logger.info(info_msg)
        print(f"[INFO] {info_msg}", flush=True)

        # Query schedule_details table
        rows = DatabaseConnection.execute_query(
            """
            SELECT schedule_id, exam_id, slot, faculty, semester
            FROM schedule_details
            WHERE schedule_id = :schedule_id
            ORDER BY slot, exam_id
        """,
            {"schedule_id": schedule_id},
        )

        if not rows:
            warning_msg = f"No schedule data found for {schedule_id}"
            logger.warning(warning_msg)
            print(f"[WARNING] {warning_msg}", flush=True)
            abort(404, description=f"No schedule data found for {schedule_id}")

        # Create CSV content with faculty column
        csv_content = "schedule_id,exam_id,slot,faculty,semester\n"
        for row in rows:
            # Handle empty faculty field
            faculty = row.faculty if row.faculty else ""
            csv_content += (
                f"{row.schedule_id},{row.exam_id},{row.slot},{faculty},{row.semester}\n"
            )

        # Create response with CSV content
        response = make_response(csv_content)
        response.headers["Content-Type"] = "text/csv"
        response.headers["Content-Disposition"] = (
            f'attachment; filename="{schedule_id}.csv"'
        )

        success_msg = (
            f"Downloaded schedule {schedule_id} with {len(rows)} exam assignments"
        )
        logger.info(success_msg)
        print(f"[INFO] {success_msg}", flush=True)
        return response

    except Exception as e:
        error_msg = f"Error downloading schedule {schedule_id}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e)}), 500


@helpers_bp.route("/download/schedules/<schedule_id>/debug")
def debug_schedule_download(schedule_id):
    """Debug endpoint to see what data exists for a schedule."""
    try:
        info_msg = f"Debug schedule download for {schedule_id}"
        logger.info(info_msg)
        print(f"[INFO] {info_msg}", flush=True)

        # Check if schedule_details table exists
        table_exists = DatabaseConnection.table_exists("schedule_details")

        if not table_exists:
            warning_msg = "schedule_details table does not exist"
            logger.warning(warning_msg)
            print(f"[WARNING] {warning_msg}", flush=True)
            return jsonify(
                {
                    "schedule_id": schedule_id,
                    "error": "schedule_details table does not exist",
                    "table_exists": False,
                    "suggestion": "Create the schedule_details table first",
                }
            )

        # Count total rows for this schedule
        count_result = DatabaseConnection.execute_query(
            """
            SELECT COUNT(*) as total
            FROM schedule_details
            WHERE schedule_id = :schedule_id
        """,
            {"schedule_id": schedule_id},
            fetch_all=False,
        )

        total_count = count_result.total if count_result else 0

        # Get sample data
        sample_rows = DatabaseConnection.execute_query(
            """
            SELECT schedule_id, exam_id, slot, faculty, semester
            FROM schedule_details
            WHERE schedule_id = :schedule_id
            ORDER BY slot, exam_id
            LIMIT 10
        """,
            {"schedule_id": schedule_id},
        )

        # Get database info
        db_info = (
            "Connected" if DatabaseConnection.check_connection() else "Disconnected"
        )

        sample_data = []
        for row in sample_rows:
            sample_data.append(
                {
                    "schedule_id": row.schedule_id,
                    "exam_id": row.exam_id,
                    "slot": row.slot,
                    "faculty": row.faculty,
                    "semester": row.semester,
                }
            )

        success_msg = f"Debug complete for {schedule_id}: {total_count} rows found"
        logger.info(success_msg)
        print(f"[INFO] {success_msg}", flush=True)

        return jsonify(
            {
                "schedule_id": schedule_id,
                "total_rows": total_count,
                "has_data": total_count > 0,
                "sample_data": sample_data,
                "database_connection": db_info,
                "table_exists": table_exists,
            }
        )

    except Exception as e:
        error_msg = f"Error in debug_schedule_download for {schedule_id}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify(
            {
                "schedule_id": schedule_id,
                "error": str(e),
                "table_exists": DatabaseConnection.table_exists("schedule_details"),
                "database_connection": (
                    "Connected"
                    if DatabaseConnection.check_connection()
                    else "Disconnected"
                ),
                "suggestion": "Check database connection and table structure",
            }
        )


# Additional helper endpoints for better database management
@helpers_bp.route("/database/health", methods=["GET"])
def database_health():
    """Check database health and connection status"""
    try:
        db_healthy = DatabaseConnection.check_connection()

        # Check important tables
        tables_to_check = ["schedule_details", "pinned_schedules"]
        table_status = {}

        for table in tables_to_check:
            table_status[table] = DatabaseConnection.table_exists(table)

        plots_dir = Path(current_app.static_folder) / "plots"
        health_msg = (
            f"Database health check: {'healthy' if db_healthy else 'unhealthy'}"
        )
        logger.info(health_msg)
        print(f"[INFO] {health_msg}", flush=True)

        return jsonify(
            {
                "database_connected": db_healthy,
                "tables": table_status,
                "status": "healthy" if db_healthy else "unhealthy",
                "plots_directory": {
                    "path": str(plots_dir),
                    "exists": plots_dir.exists(),
                },
            }
        ), (200 if db_healthy else 503)

    except Exception as e:
        error_msg = f"Database health check failed: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return (
            jsonify(
                {"database_connected": False, "error": str(e), "status": "unhealthy"}
            ),
            503,
        )


@helpers_bp.route("/database/tables", methods=["GET"])
def list_database_tables():
    """List all tables in the database"""
    try:
        tables = DatabaseConnection.execute_query(
            """
            SELECT name, type 
            FROM sqlite_master 
            WHERE type='table'
            ORDER BY name
        """
        )

        table_list = []
        for table in tables:
            # Get row count for each table
            try:
                count_result = DatabaseConnection.execute_query(
                    f"SELECT COUNT(*) as count FROM {table.name}", fetch_all=False
                )
                row_count = count_result.count if count_result else 0
            except Exception:
                row_count = "unknown"

            table_list.append(
                {"name": table.name, "type": table.type, "row_count": row_count}
            )

        info_msg = f"Listed {len(table_list)} database tables"
        logger.info(info_msg)
        print(f"[INFO] {info_msg}", flush=True)

        return jsonify({"tables": table_list, "total_tables": len(table_list)})

    except Exception as e:
        error_msg = f"Error listing database tables: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e)}), 500


@helpers_bp.route("/schedules/<schedule_id>/summary", methods=["GET"])
def get_schedule_summary(schedule_id):
    """Get a summary of a specific schedule"""
    try:
        # Check if schedule exists
        schedule_exists = DatabaseConnection.execute_query(
            """
            SELECT COUNT(*) as count 
            FROM schedule_details 
            WHERE schedule_id = :schedule_id
        """,
            {"schedule_id": schedule_id},
            fetch_all=False,
        )

        if not schedule_exists or schedule_exists.count == 0:
            warning_msg = f"Schedule {schedule_id} not found"
            logger.warning(warning_msg)
            print(f"[WARNING] {warning_msg}", flush=True)
            return jsonify({"error": f"Schedule {schedule_id} not found"}), 404

        # Get schedule statistics
        stats = DatabaseConnection.execute_query(
            """
            SELECT 
                COUNT(*) as total_exams,
                COUNT(DISTINCT slot) as total_slots,
                COUNT(DISTINCT faculty) as total_faculty,
                COUNT(DISTINCT semester) as total_semesters,
                MIN(slot) as min_slot,
                MAX(slot) as max_slot
            FROM schedule_details 
            WHERE schedule_id = :schedule_id
        """,
            {"schedule_id": schedule_id},
            fetch_all=False,
        )

        # Get slot distribution
        slot_distribution = DatabaseConnection.execute_query(
            """
            SELECT slot, COUNT(*) as exam_count
            FROM schedule_details 
            WHERE schedule_id = :schedule_id
            GROUP BY slot
            ORDER BY slot
        """,
            {"schedule_id": schedule_id},
        )

        slot_dist_dict = {row.slot: row.exam_count for row in slot_distribution}

        summary = {
            "schedule_id": schedule_id,
            "statistics": {
                "total_exams": stats.total_exams,
                "total_slots": stats.total_slots,
                "total_faculty": stats.total_faculty,
                "total_semesters": stats.total_semesters,
                "slot_range": f"{stats.min_slot} - {stats.max_slot}",
            },
            "slot_distribution": slot_dist_dict,
        }

        success_msg = f"Generated summary for schedule {schedule_id}"
        logger.info(success_msg)
        print(f"[INFO] {success_msg}", flush=True)
        return jsonify(summary)

    except Exception as e:
        error_msg = f"Error getting schedule summary for {schedule_id}: {e}"
        logger.error(error_msg)
        print(f"[ERROR] {error_msg}", flush=True)
        return jsonify({"error": str(e)}), 500
