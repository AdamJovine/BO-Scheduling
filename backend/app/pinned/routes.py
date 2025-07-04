from flask import Blueprint, jsonify, request
from datetime import datetime
import json
import logging
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)
pinned_bp = Blueprint("pinned", __name__)


@pinned_bp.route("/pinned-schedules/<user_id>", methods=["GET"])
def get_pinned_schedules(user_id):
    """Get all pinned schedules for a user"""
    try:
        pins = DatabaseConnection.execute_query(
            """
            SELECT id, user_id, schedule_id, name, data, created 
            FROM pinned_schedules 
            WHERE user_id = :user_id 
            ORDER BY created DESC
            """,
            {"user_id": user_id},
        )

        pinned_schedules = []
        for pin in pins:
            pinned_schedules.append(
                {
                    "id": pin.id,
                    "user_id": pin.user_id,
                    "sched_id": pin.schedule_id,
                    "name": pin.name,
                    "data": json.loads(pin.data) if pin.data else None,
                    "created": pin.created,
                }
            )

        logger.info(
            f"Retrieved {len(pinned_schedules)} pinned schedules for user {user_id}"
        )
        return jsonify({"success": True, "pinned_schedules": pinned_schedules})

    except Exception as e:
        logger.error(f"Error in get_pinned_schedules for user {user_id}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@pinned_bp.route("/pinned-schedules/<user_id>/<schedule_id>", methods=["POST"])
def pin_schedule(user_id, schedule_id):
    """Pin a schedule for a user"""
    try:
        data = request.get_json()
        schedule_name = data.get("schedule_name", "") if data else ""
        schedule_data = json.dumps(data.get("schedule_data", {})) if data else "{}"
        created_time = datetime.now().isoformat()

        rows_affected = DatabaseConnection.execute_insert(
            """
            INSERT OR REPLACE INTO pinned_schedules 
            (user_id, schedule_id, name, data, created) 
            VALUES (:user_id, :schedule_id, :name, :data, :created)
            """,
            {
                "user_id": user_id,
                "schedule_id": schedule_id,
                "name": schedule_name,
                "data": schedule_data,
                "created": created_time,
            },
        )

        if rows_affected > 0:
            logger.info(
                f"Successfully pinned schedule {schedule_id} for user {user_id}"
            )
            return jsonify({"success": True}), 201
        else:
            logger.warning(
                f"No rows affected when pinning schedule {schedule_id} for user {user_id}"
            )
            return jsonify({"success": False, "error": "Failed to pin schedule"}), 500

    except Exception as e:
        logger.error(
            f"Error in pin_schedule for user {user_id}, schedule {schedule_id}: {e}"
        )
        return jsonify({"success": False, "error": str(e)}), 500


@pinned_bp.route("/pinned-schedules/<user_id>/<schedule_id>", methods=["DELETE"])
def unpin_schedule(user_id, schedule_id):
    """Unpin a schedule for a user"""
    try:
        # Check if the pin exists first
        pin = DatabaseConnection.execute_query(
            """
            SELECT id FROM pinned_schedules 
            WHERE user_id = :user_id AND schedule_id = :schedule_id
            """,
            {"user_id": user_id, "schedule_id": schedule_id},
            fetch_all=False,
        )

        if not pin:
            logger.warning(
                f"Attempted to unpin non-existent schedule {schedule_id} for user {user_id}"
            )
            return jsonify({"success": False, "error": "Schedule not found"}), 404

        # Delete the pin
        rows_affected = DatabaseConnection.execute_delete(
            """
            DELETE FROM pinned_schedules 
            WHERE user_id = :user_id AND schedule_id = :schedule_id
            """,
            {"user_id": user_id, "schedule_id": schedule_id},
        )

        if rows_affected > 0:
            logger.info(
                f"Successfully unpinned schedule {schedule_id} for user {user_id}"
            )
            return jsonify({"success": True}), 200
        else:
            logger.warning(
                f"No rows affected when unpinning schedule {schedule_id} for user {user_id}"
            )
            return jsonify({"success": False, "error": "Failed to unpin schedule"}), 500

    except Exception as e:
        logger.error(
            f"Error in unpin_schedule for user {user_id}, schedule {schedule_id}: {e}"
        )
        return jsonify({"success": False, "error": str(e)}), 500


@pinned_bp.route("/test-pinned", methods=["GET"])
def test_pinned():
    """Test the pinned schedules API and database connection"""
    try:
        # Test database connection
        db_connected = DatabaseConnection.check_connection()

        # Check if pinned_schedules table exists
        table_exists = DatabaseConnection.table_exists("pinned_schedules")

        # Get some basic stats if table exists
        pin_count = 0
        if table_exists:
            try:
                result = DatabaseConnection.execute_query(
                    "SELECT COUNT(*) as count FROM pinned_schedules", fetch_all=False
                )
                pin_count = result.count if result else 0
            except Exception as e:
                logger.warning(f"Could not get pin count: {e}")

        response_data = {
            "success": True,
            "message": "Pinned schedules API is working!",
            "database_connected": db_connected,
            "pinned_table_exists": table_exists,
            "total_pins": pin_count,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("Pinned schedules test endpoint called successfully")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in test_pinned endpoint: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# Optional: Add a health check route
@pinned_bp.route("/pinned-schedules/health", methods=["GET"])
def health_check():
    """Health check for pinned schedules service"""
    try:
        db_healthy = DatabaseConnection.check_connection()
        table_exists = DatabaseConnection.table_exists("pinned_schedules")

        status = "healthy" if (db_healthy and table_exists) else "unhealthy"

        return jsonify(
            {
                "status": status,
                "database": "connected" if db_healthy else "disconnected",
                "table_exists": table_exists,
                "timestamp": datetime.now().isoformat(),
            }
        ), (200 if status == "healthy" else 503)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            ),
            503,
        )
