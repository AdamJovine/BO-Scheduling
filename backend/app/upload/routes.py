# app/upload/routes.py
from flask import Blueprint, request, current_app, jsonify
from ..extensions import db
import pandas as pd
import io
from werkzeug.utils import secure_filename
from datetime import datetime
import logging
import os
from sqlalchemy import text

# Create blueprint
upload_bp = Blueprint("upload", __name__)

logger = logging.getLogger(__name__)

# Metrics mapping from your existing code
METRICS_MAP = {
    "conflicts": "conflicts",
    "quints": "quints",
    "quads": "quads",
    "four in five slots": "four_in_five",
    "triple in 24h (no gaps)": "triple_in_24h",
    "triple in same day (no gaps)": "triple_in_same_day",
    "three in four slots": "three_in_four",
    "evening/morning b2b": "evening_morning_b2b",
    "other b2b": "other_b2b",
    "two in three slots": "two_in_three",
    "singular late exam": "singular_late",
    "two exams, large gap": "two_large_gap",
    "avg_max": "avg_max",
    "lateness": "lateness",
    "size_cutoff": "size_cutoff",
    "reserved": "reserved",
    "num_blocks": "num_blocks",
    "alpha": "alpha",
    "gamma": "gamma",
    "delta": "delta",
    "vega": "vega",
    "theta": "theta",
    "large_block_size": "large_block_size",
    "large_exam_weight": "large_exam_weight",
    "large_block_weight": "large_block_weight",
    "large_size_1": "large_size_1",
    "large_cutoff_freedom": "large_cutoff_freedom",
    "tradeoff": "tradeoff",
    "flpens": "flpens",
    "semester": "semester",
}

# Integer columns (everything else is FLOAT)
INTEGER_COLUMNS = {
    "conflicts",
    "quints",
    "quads",
    "four_in_five",
    "triple_in_24h",
    "triple_in_same_day",
    "three_in_four",
    "evening_morning_b2b",
    "other_b2b",
    "two_in_three",
    "singular_late",
    "two_large_gap",
    "lateness",
    "size_cutoff",
    "reserved",
    "num_blocks",
}


@upload_bp.route("/upload-metrics", methods=["POST"])
def upload_metrics():
    """
    Endpoint to receive metrics CSV files from G2 cluster and update database
    Expected: multipart/form-data with CSV files OR a single combined CSV
    """
    try:
        # Check if request has files
        if "files" not in request.files:
            return {"error": "No files provided"}, 400

        files = request.files.getlist("files")
        if not files or files[0].filename == "":
            return {"error": "No files selected"}, 400

        # API key authentication
        # api_key = request.headers.get("X-API-Key")
        # if api_key != current_app.config.get("UPLOAD_API_KEY"):
        #    logger.warning(f"Invalid API key attempt from {request.remote_addr}")
        #    return {"error": "Invalid API key"}, 401

        # Get semester from request (default to current)
        semester = request.form.get("semester", "sp25")  # Default semester

        processed_files = []
        errors = []
        total_metrics_inserted = 0

        for file in files:
            try:
                # Validate file type
                if not file.filename.endswith(".csv"):
                    errors.append(f"Invalid file type: {file.filename}")
                    continue

                # Read CSV into pandas DataFrame
                csv_content = file.read().decode("utf-8")
                df = pd.read_csv(io.StringIO(csv_content))

                logger.info(f"Processing {file.filename}: {len(df)} rows")

                # Process metrics CSV
                result = process_metrics_csv(df, file.filename, semester)

                processed_files.append(
                    {
                        "filename": file.filename,
                        "rows_processed": len(df),
                        "metrics_inserted": result.get("metrics_inserted", 0),
                        "status": "success",
                    }
                )

                total_metrics_inserted += result.get("metrics_inserted", 0)
                logger.info(
                    f"Successfully processed {file.filename}: {result.get('metrics_inserted', 0)} metrics inserted"
                )

            except Exception as e:
                error_msg = f"Error processing {file.filename}: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)

        # Return summary
        response = {
            "status": "completed",
            "processed_files": processed_files,
            "total_metrics_inserted": total_metrics_inserted,
            "semester": semester,
            "errors": errors,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Metrics upload completed: {total_metrics_inserted} total metrics inserted, {len(errors)} errors"
        )

        return response, 200 if not errors else 207  # 207 = partial success

    except Exception as e:
        logger.error(f"Critical error in upload_metrics: {str(e)}")
        return {"error": "Internal server error"}, 500


def process_metrics_csv(df, filename, semester):
    """
    Process metrics CSV and insert into metrics table
    Handles both individual CSV files and combined CSV files
    """
    try:
        # Ensure metrics table exists
        metrics_inserted = 0

        # If this is a combined CSV (has schedule_id column), process all rows
        if "schedule_id" in df.columns:
            logger.info(f"Processing combined metrics CSV with {len(df)} schedules")
            for _, row in df.iterrows():
                if process_single_metrics_row(row, semester, row["schedule_id"]):
                    metrics_inserted += 1
        else:
            # Single metrics file - use filename as schedule_id
            if len(df) > 0:
                schedule_id = os.path.splitext(os.path.basename(filename))[0]
                row = df.iloc[0]  # Take first row only
                if process_single_metrics_row(row, semester, schedule_id):
                    metrics_inserted += 1

        # Commit all changes
        db.session.commit()

        logger.info(f"Successfully inserted {metrics_inserted} metrics records")

        return {"success": True, "metrics_inserted": metrics_inserted}

    except Exception as e:
        db.session.rollback()
        logger.error(f"Database error in process_metrics_csv: {str(e)}")
        raise e


def process_single_metrics_row(row, semester, schedule_id):
    """
    Process a single row of metrics data
    Returns True if successfully inserted, False otherwise
    """
    try:
        # Build data dictionary
        data = {"schedule_id": schedule_id, "semester": semester}

        for csv_col, sql_col in METRICS_MAP.items():
            if csv_col not in row or pd.isna(row[csv_col]):
                continue

            val = row[csv_col]

            # Convert to appropriate type
            if sql_col in INTEGER_COLUMNS:
                try:
                    data[sql_col] = int(float(val))  # Convert to float first, then int
                except (ValueError, TypeError):
                    logger.warning(
                        f"Skipping invalid integer value for {csv_col}: {val}"
                    )
                    continue
            else:
                try:
                    data[sql_col] = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping invalid float value for {csv_col}: {val}")
                    continue

        # Insert into database if we have data beyond just schedule_id and semester
        if len(data) > 2:
            cols = ", ".join(data.keys())
            params = ", ".join(f":{c}" for c in data)
            stmt = text(f"INSERT OR REPLACE INTO metrics ({cols}) VALUES ({params})")

            db.session.execute(stmt, data)
            logger.debug(f"Inserted metrics for {schedule_id} ({len(data)-2} metrics)")
            return True
        else:
            logger.warning(f"No valid metrics found for {schedule_id}")
            return False

    except Exception as e:
        logger.error(f"Error processing metrics row for {schedule_id}: {e}")
        return False


@upload_bp.route("/upload-status", methods=["GET"])
def upload_status():
    """Health check endpoint for upload API"""
    try:
        # Check database connection
        db.session.execute(text("SELECT 1"))
        db_status = True

        # Check metrics table
        result = db.session.execute(text("SELECT COUNT(*) FROM metrics"))
        metrics_count = result.scalar()

    except Exception as e:
        db_status = False
        metrics_count = 0
        logger.error(f"Database connection failed: {e}")

    return {
        "status": "healthy" if db_status else "unhealthy",
        "database_connected": db_status,
        "metrics_count": metrics_count,
        "timestamp": datetime.utcnow().isoformat(),
        "upload_folder": str(current_app.config.get("UPLOAD_FOLDER", "")),
        "max_file_size_mb": (current_app.config.get("MAX_CONTENT_LENGTH") or 0)
        // (1024 * 1024),
    }


@upload_bp.route("/tables", methods=["GET"])
def list_tables():
    """List available database tables (for debugging)"""
    try:
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()

        # Get row counts for each table
        table_info = {}
        for table in tables:
            try:
                result = db.session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.scalar()
                table_info[table] = {"row_count": count}
            except Exception as e:
                table_info[table] = {"error": str(e)}

        return {"tables": table_info, "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Error listing tables: {e}")
        return {"error": "Failed to list tables"}, 500


@upload_bp.route("/metrics/<semester>", methods=["GET"])
def get_metrics(semester):
    """Get metrics for a specific semester (for testing/verification)"""
    try:
        result = db.session.execute(
            text("SELECT COUNT(*) FROM metrics WHERE semester = :semester"),
            {"semester": semester},
        )
        count = result.scalar()

        return {
            "semester": semester,
            "metrics_count": count,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting metrics for {semester}: {e}")
        return {"error": "Failed to get metrics"}, 500


# _______________________________________________________________________________________________________
# SCHEDULES
# _______________________________________________________________________________________________________

from flask import request, jsonify
from sqlalchemy import create_engine, text
import csv
import io
import os
from typing import List, Dict, Tuple


@upload_bp.route("/upload-schedules", methods=["POST"])
def upload_schedules():
    """
    API endpoint to upload and process schedule CSV files
    Expects multiple CSV files via form data with 'files' key
    """
    try:
        # Check authentication
        # api_key = request.headers.get("X-API-Key")
        # expected_key = os.environ.get("UPLOAD_API_KEY", "your-secret-api-key-here")
        # if not api_key or api_key != expected_key:
        #    return jsonify({"error": "Invalid or missing API key"}), 401

        # Check if files were uploaded
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No files selected"}), 400

        # Get semester from form data or default
        semester = request.form.get("semester", "sp25")

        # Process each uploaded CSV
        results = process_schedule_csvs(files, semester)

        # Determine response status
        has_errors = any(result.get("error") for result in results["processed_files"])
        status_code = 207 if has_errors else 200  # 207 = Multi-Status (partial success)

        return jsonify(results), status_code

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def process_schedule_csvs(files: List, semester: str) -> Dict:
    """
    Process uploaded schedule CSV files and insert into database
    """
    # Database setup
    engine = create_engine("sqlite:///schedules.db", echo=False)

    # SQL statements
    delete_sql = text("DELETE FROM schedule_details WHERE schedule_id = :schedule_id")
    insert_sql = text(
        """
        INSERT OR REPLACE INTO schedule_details
          (schedule_id, exam_id, slot, faculty, semester)
        VALUES
          (:schedule_id, :exam_id, :slot, :faculty, :semester)
    """
    )

    create_table_sql = text(
        """
        CREATE TABLE IF NOT EXISTS schedule_details (
          schedule_id   TEXT    NOT NULL,
          exam_id       TEXT    NOT NULL,
          slot          INTEGER,
          faculty       TEXT,
          semester      TEXT,
          PRIMARY KEY (schedule_id, exam_id)
        )
    """
    )

    processed_files = []
    total_records = 0

    try:
        with engine.begin() as conn:
            # Ensure table exists
            conn.execute(create_table_sql)

            for file in files:
                if not file.filename.endswith(".csv"):
                    processed_files.append(
                        {
                            "filename": file.filename,
                            "error": "File is not a CSV",
                            "records_processed": 0,
                        }
                    )
                    continue

                try:
                    result = process_single_schedule_csv(
                        file, semester, conn, delete_sql, insert_sql
                    )
                    processed_files.append(result)
                    total_records += result.get("records_processed", 0)

                except Exception as e:
                    processed_files.append(
                        {
                            "filename": file.filename,
                            "error": f"Error processing file: {str(e)}",
                            "records_processed": 0,
                        }
                    )

    except Exception as e:
        return {
            "error": f"Database error: {str(e)}",
            "processed_files": processed_files,
            "total_records": total_records,
        }

    return {
        "message": f"Processed {len(files)} files, {total_records} total records",
        "processed_files": processed_files,
        "total_records": total_records,
    }


def process_single_schedule_csv(
    file, semester: str, conn, delete_sql, insert_sql
) -> Dict:
    """
    Process a single schedule CSV file
    """
    filename = file.filename
    schedule_id = os.path.splitext(filename)[0]

    # Read file content
    content = file.read().decode("utf-8")
    file.seek(0)  # Reset file pointer for potential re-reading

    # Parse CSV
    csv_reader = csv.DictReader(io.StringIO(content))

    # Find required columns
    exam_col = None
    slot_col = None

    for col in csv_reader.fieldnames or []:
        if "exam" in col.lower() and "group" in col.lower():
            exam_col = col
        elif "slot" in col.lower():
            slot_col = col

    if not exam_col or not slot_col:
        return {
            "filename": filename,
            "error": f"Required columns not found. Available: {csv_reader.fieldnames}",
            "records_processed": 0,
        }

    # Remove existing records for this schedule
    conn.execute(delete_sql, {"schedule_id": schedule_id})

    # Prepare batch data
    batch = []
    csv_reader = csv.DictReader(io.StringIO(content))  # Re-create reader

    for row in csv_reader:
        exam_id = row.get(exam_col, "").strip()
        if not exam_id:
            continue

        try:
            slot = int(row.get(slot_col, 0))
        except (ValueError, TypeError):
            continue

        batch.append(
            {
                "schedule_id": schedule_id,
                "exam_id": exam_id,
                "slot": slot,
                "faculty": "",  # Can be updated later if needed
                "semester": semester,
            }
        )

    # Insert batch
    if batch:
        conn.execute(insert_sql, batch)
        return {
            "filename": filename,
            "schedule_id": schedule_id,
            "records_processed": len(batch),
            "success": True,
        }
    else:
        return {
            "filename": filename,
            "schedule_id": schedule_id,
            "error": "No valid records found in CSV",
            "records_processed": 0,
        }


@upload_bp.route("/schedule-upload-status", methods=["GET"])
def get_schedule_upload_status():
    """
    Health check endpoint for schedule uploads
    """
    try:
        engine = create_engine("sqlite:///schedules.db", echo=False)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM schedule_details"))
            count = result.scalar()

            return (
                jsonify(
                    {
                        "status": "healthy",
                        "total_schedule_records": count,
                        "message": "Schedule upload service is operational",
                    }
                ),
                200,
            )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": f"Database connection failed: {str(e)}"}
            ),
            500,
        )


# ________________________________________________________________________________________________________
# IMAGES
# ________________________________________________________________________________________________________


from flask import request, jsonify, current_app
import os
from werkzeug.utils import secure_filename
from PIL import Image
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
import logging

# Configuration
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_FILES_PER_REQUEST = 10

logger = logging.getLogger(__name__)


def get_static_directory():
    """Get the correct static directory path"""
    # Option 1: Use environment variable (recommended)
    static_dir = os.environ.get("STATIC_DIRECTORY")
    if static_dir:
        return static_dir

    # Option 2: Check if we're in Docker vs local development
    if os.path.exists("/app"):  # Docker environment
        # Use the backend structure that matches your volume mount
        return "/app/backend/app/static"
    else:  # Local development
        # Use relative path to backend directory
        return os.path.join(
            os.path.dirname(current_app.root_path), "backend", "app", "static"
        )


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image_file(file) -> Tuple[bool, str]:
    """Validate that the uploaded file is actually a valid image"""
    try:
        # Try to open with PIL to verify it's a valid image
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)  # Reset stream position
        return True, ""
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"


def get_image_info(filepath: str) -> Dict:
    """Get information about an uploaded image"""
    try:
        with Image.open(filepath) as img:
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": os.path.getsize(filepath),
            }
    except Exception as e:
        return {"error": f"Could not read image info: {str(e)}"}


@upload_bp.route("/upload-images", methods=["POST"])
def upload_images():
    """
    API endpoint to upload image files
    POST /api/upload-images
    """
    try:
        # Check authentication
        # api_key = request.headers.get("X-API-Key")
        # expected_key = os.environ.get("UPLOAD_API_KEY", "your-secret-api-key-here")
        # if not api_key or api_key != expected_key:
        #    return jsonify({"error": "Invalid or missing API key"}), 401

        # Check if files were uploaded
        if "files" not in request.files:
            return jsonify({"error": "No files provided"}), 400

        files = request.files.getlist("files")
        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No files selected"}), 400

        # Limit number of files
        if len(files) > MAX_FILES_PER_REQUEST:
            return (
                jsonify(
                    {
                        "error": f"Too many files. Maximum {MAX_FILES_PER_REQUEST} files per request"
                    }
                ),
                400,
            )

        # Get optional subdirectory from form data
        subdirectory = request.form.get("subdirectory", "").strip()
        if subdirectory:
            subdirectory = secure_filename(subdirectory)

        # Process each uploaded image
        results = process_image_uploads(files, subdirectory)

        # Determine response status
        has_errors = any(result.get("error") for result in results["processed_files"])
        status_code = 207 if has_errors else 200  # 207 = Multi-Status (partial success)

        return jsonify(results), status_code

    except Exception as e:
        logger.error(f"Unexpected error in image upload: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


def process_image_uploads(files: List, subdirectory: str = "") -> Dict:
    """Process multiple image uploads"""

    # Use the new static directory function
    static_dir = get_static_directory()

    if subdirectory:
        upload_dir = os.path.join(static_dir, subdirectory)
    else:
        upload_dir = static_dir

    # Create directory if it doesn't exist
    os.makedirs(upload_dir, exist_ok=True)

    logger.info(f"Uploading files to: {upload_dir}")

    processed_files = []
    total_size = 0

    for file in files:
        if not file.filename:
            processed_files.append(
                {
                    "filename": "Unknown",
                    "error": "Empty filename",
                    "uploaded_path": None,
                }
            )
            continue

        try:
            result = process_single_image_upload(file, upload_dir, subdirectory)
            processed_files.append(result)

            if result.get("size_bytes"):
                total_size += result["size_bytes"]

        except Exception as e:
            processed_files.append(
                {
                    "filename": file.filename,
                    "error": f"Error processing file: {str(e)}",
                    "uploaded_path": None,
                }
            )

    return {
        "message": f"Processed {len(files)} files, total size: {total_size:,} bytes",
        "processed_files": processed_files,
        "total_files": len(files),
        "total_size_bytes": total_size,
        "upload_directory": subdirectory or "static (root)",
        "actual_upload_path": upload_dir,  # Debug info
    }


def process_single_image_upload(file, upload_dir: str, subdirectory: str = "") -> Dict:
    """Process a single image upload"""

    original_filename = file.filename

    # Check file extension
    if not allowed_file(original_filename):
        return {
            "filename": original_filename,
            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
            "uploaded_path": None,
        }

    # Check file size
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning

    if file_size > MAX_FILE_SIZE:
        return {
            "filename": original_filename,
            "error": f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f}MB",
            "uploaded_path": None,
        }

    if file_size == 0:
        return {
            "filename": original_filename,
            "error": "Empty file",
            "uploaded_path": None,
        }

    # Validate that it's actually an image
    is_valid, error_msg = validate_image_file(file)
    if not is_valid:
        return {
            "filename": original_filename,
            "error": error_msg,
            "uploaded_path": None,
        }

    # Save file with original filename
    filepath = os.path.join(upload_dir, original_filename)

    # Save the file
    try:
        file.save(filepath)
        logger.info(f"Saved file to: {filepath}")

        # Get image information
        image_info = get_image_info(filepath)

        # Construct the web-accessible path
        if subdirectory:
            web_path = f"/static/{subdirectory}/{original_filename}"
        else:
            web_path = f"/static/{original_filename}"

        return {
            "filename": original_filename,
            "uploaded_filename": original_filename,
            "uploaded_path": web_path,
            "local_path": filepath,
            "success": True,
            **image_info,
        }

    except Exception as e:
        # Clean up file if it was partially created
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass

        return {
            "filename": original_filename,
            "error": f"Failed to save file: {str(e)}",
            "uploaded_path": None,
        }


@upload_bp.route("/images", methods=["GET"])
def list_uploaded_images():
    """
    API endpoint to list uploaded images
    GET /api/images
    """
    try:
        subdirectory = request.args.get("subdirectory", "").strip()
        if subdirectory:
            subdirectory = secure_filename(subdirectory)

        # Use the new static directory function
        static_dir = get_static_directory()
        logger.info(f"Listing images from static_dir: {static_dir}")

        if subdirectory:
            search_dir = os.path.join(static_dir, subdirectory)
        else:
            search_dir = static_dir

        if not os.path.exists(search_dir):
            return (
                jsonify(
                    {
                        "images": [],
                        "directory": subdirectory or "static (root)",
                        "message": "Directory does not exist",
                        "search_dir": search_dir,  # Debug info
                    }
                ),
                200,
            )

        images = []
        for filename in os.listdir(search_dir):
            filepath = os.path.join(search_dir, filename)

            # Check if it's a file and has allowed extension
            if os.path.isfile(filepath) and allowed_file(filename):
                # Get file stats
                stat = os.stat(filepath)

                # Get image info
                image_info = get_image_info(filepath)

                # Construct web path
                if subdirectory:
                    web_path = f"/static/{subdirectory}/{filename}"
                else:
                    web_path = f"/static/{filename}"

                images.append(
                    {
                        "filename": filename,
                        "web_path": web_path,
                        "upload_time": datetime.fromtimestamp(
                            stat.st_mtime
                        ).isoformat(),
                        "file_size": stat.st_size,
                        **image_info,
                    }
                )

        # Sort by upload time (newest first)
        images.sort(key=lambda x: x["upload_time"], reverse=True)

        return (
            jsonify(
                {
                    "images": images,
                    "directory": subdirectory or "static (root)",
                    "total_images": len(images),
                    "total_size_bytes": sum(img["file_size"] for img in images),
                    "search_dir": search_dir,  # Debug info
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error listing images: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@upload_bp.route("/images/<filename>", methods=["DELETE"])
def delete_image():
    """
    API endpoint to delete an uploaded image
    DELETE /api/images/<filename>
    """
    try:
        # Check authentication
        api_key = request.headers.get("X-API-Key")
        expected_key = os.environ.get("UPLOAD_API_KEY", "your-secret-api-key-here")
        if not api_key or api_key != expected_key:
            return jsonify({"error": "Invalid or missing API key"}), 401

        filename = request.view_args.get("filename")
        if not filename:
            return jsonify({"error": "No filename provided"}), 400

        # Security: ensure filename is safe
        filename = secure_filename(filename)
        if not allowed_file(filename):
            return jsonify({"error": "Invalid file type"}), 400

        subdirectory = request.args.get("subdirectory", "").strip()
        if subdirectory:
            subdirectory = secure_filename(subdirectory)

        # Use the new static directory function
        static_dir = get_static_directory()

        if subdirectory:
            filepath = os.path.join(static_dir, subdirectory, filename)
        else:
            filepath = os.path.join(static_dir, filename)

        if not os.path.exists(filepath):
            return jsonify({"error": "File not found"}), 404

        # Delete the file
        os.remove(filepath)

        return (
            jsonify(
                {
                    "message": f"File {filename} deleted successfully",
                    "filename": filename,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error deleting image: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@upload_bp.route("/image-upload-status", methods=["GET"])
def get_image_upload_status():
    """
    Health check endpoint for image uploads
    GET /api/image-upload-status
    """
    try:
        # Use the new static directory function
        static_dir = get_static_directory()

        # Count total images
        total_images = 0
        total_size = 0

        if os.path.exists(static_dir):
            for root, dirs, files in os.walk(static_dir):
                for file in files:
                    if allowed_file(file):
                        filepath = os.path.join(root, file)
                        total_images += 1
                        total_size += os.path.getsize(filepath)

        return (
            jsonify(
                {
                    "status": "healthy",
                    "static_directory": static_dir,
                    "static_directory_exists": os.path.exists(static_dir),
                    "total_images": total_images,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "allowed_extensions": list(ALLOWED_EXTENSIONS),
                    "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
                    "message": "Image upload service is operational",
                }
            ),
            200,
        )

    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"Image service error: {str(e)}"}),
            500,
        )
