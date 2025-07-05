from flask import Blueprint, jsonify, request
from sqlalchemy import text
import uuid
import json
import logging
from datetime import datetime
from flask_cors import cross_origin
from ..connection import DatabaseConnection

logger = logging.getLogger(__name__)
survey_bp = Blueprint("survey", __name__)


# Slider config routes
@survey_bp.route("/slider-configs", methods=["GET"])
@cross_origin()
def list_configs():
    """Get all saved configurations"""
    try:
        logger.info("📋 Listing slider configurations...")

        configs = DatabaseConnection.execute_query(
            """
            SELECT id, name, description, thresholds, timestamp 
            FROM slider_configs 
            ORDER BY timestamp DESC
            """
        )

        logger.info(f"📊 Found {len(configs)} slider configurations")

        config_list = []
        for i, config in enumerate(configs):
            try:
                config_dict = {
                    "id": config.id,
                    "name": config.name,
                    "description": config.description,
                    "thresholds": (
                        json.loads(config.thresholds) if config.thresholds else {}
                    ),
                    "created": config.timestamp,
                }
                config_list.append(config_dict)
                logger.debug(f"📝 Config {i+1}: {config.name}")
            except Exception as e:
                logger.warning(f"⚠️ Error processing config {i+1}: {e}")

        logger.info(f"✅ Successfully processed {len(config_list)} configurations")
        return jsonify({"configs": config_list})

    except Exception as e:
        logger.error(f"❌ Error listing configs: {e}")
        return jsonify({"error": str(e), "configs": []}), 500


@survey_bp.route("/slider-configs", methods=["POST"])
@cross_origin()
def save_config():
    """Save a new slider configuration"""
    try:
        data = request.get_json()
        name = data.get("name")
        thresholds = data.get("thresholds", {})
        description = data.get("description", "")

        if not name or not thresholds:
            logger.warning("⚠️ Missing required fields: name and thresholds")
            return jsonify({"error": "Name and thresholds are required"}), 400

        config_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        logger.info(f"💾 Saving slider configuration: {name}")

        DatabaseConnection.execute_query(
            """
            INSERT INTO slider_configs (id, name, description, thresholds, created)
            VALUES (:id, :name, :description, :thresholds, :created)
        """,
            {
                "id": config_id,
                "name": name,
                "description": description,
                "thresholds": json.dumps(thresholds),
                "created": timestamp,
            },
        )

        logger.info(f"✅ Successfully saved configuration: {name} (ID: {config_id})")

        return jsonify(
            {
                "success": True,
                "message": "Configuration saved successfully",
                "config_id": config_id,
                "name": name,
            }
        )

    except Exception as e:
        logger.error(f"❌ Error saving config: {e}")
        return jsonify({"error": str(e)}), 500


# Slider recording routes
@survey_bp.route("/slider-recordings", methods=["GET"])
@cross_origin()
def get_slider_recordings():
    """Get all slider recordings with optional filtering"""
    try:
        session_id = request.args.get("session_id")
        slider_key = request.args.get("slider_key")
        limit = request.args.get("limit", type=int)

        logger.info(
            f"📊 Getting slider recordings (session_id={session_id}, slider_key={slider_key}, limit={limit})"
        )

        # Build query with filters
        query = """
            SELECT id, session_id, slider_key, value, min_value, max_value, created
            FROM slider_recordings
        """

        conditions = []
        params = {}

        if session_id:
            conditions.append("session_id = :session_id")
            params["session_id"] = session_id
            logger.info(f"🔍 Filtering by session_id: {session_id}")

        if slider_key:
            conditions.append("slider_key = :slider_key")
            params["slider_key"] = slider_key
            logger.info(f"🔍 Filtering by slider_key: {slider_key}")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created DESC"

        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit
            logger.info(f"🔍 Limiting results to: {limit}")

        logger.info(f"🔍 Executing query: {query}")
        recordings = DatabaseConnection.execute_query(query, params)
        logger.info(f"📊 Query returned {len(recordings)} recordings")

        recording_list = []
        for i, recording in enumerate(recordings):
            try:
                recording_dict = {
                    "id": recording.id,
                    "session_id": recording.session_id,
                    "slider_key": recording.slider_key,
                    "value": recording.value,
                    "min_value": recording.min_value,
                    "max_value": recording.max_value,
                    "created": recording.timestamp,
                }
                recording_list.append(recording_dict)
                if i < 3:  # Log first 3
                    logger.debug(
                        f"📝 Recording {i+1}: {recording.slider_key}={recording.value}"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Error processing recording {i+1}: {e}")

        logger.info(f"✅ Successfully processed {len(recording_list)} recordings")
        return jsonify({"recordings": recording_list, "total": len(recording_list)})

    except Exception as e:
        logger.error(f"❌ Error getting slider recordings: {e}")
        return jsonify({"error": str(e)}), 500


@survey_bp.route("/slider-recordings", methods=["POST"])
@cross_origin()
def record_slider_interaction():
    """Record a single slider interaction"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        slider_key = data.get("slider_key")
        value = data.get("value")
        min_value = data.get("min_value")
        max_value = data.get("max_value")

        if not all([session_id, slider_key, value is not None]):
            logger.warning("⚠️ Missing required fields for slider recording")
            return (
                jsonify({"error": "session_id, slider_key, and value are required"}),
                400,
            )

        # Remove this line since we're letting SQLite auto-generate the ID
        # recording_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        logger.info(
            f"📊 Recording slider interaction: {slider_key}={value} (session: {session_id})"
        )

        # Remove 'id' from the INSERT statement - let SQLite auto-generate it
        # Don't assign result for INSERT operations
        DatabaseConnection.execute_query(
            """
            INSERT INTO slider_recordings (session_id, slider_key, value, min_value, max_value, created)
            VALUES (:session_id, :slider_key, :value, :min_value, :max_value, :created)
        """,
            {
                "session_id": session_id,
                "slider_key": slider_key,
                "value": value,
                "min_value": min_value,
                "max_value": max_value,
                "created": timestamp,
            },
        )

        # Get the auto-generated ID if you need it for the response
        # (This depends on how your DatabaseConnection.execute_query works)
        logger.info(f"✅ Successfully recorded slider interaction")

        return jsonify(
            {
                "success": True,
                "message": "Slider interaction recorded successfully",
                # "recording_id": recording_id,  # Remove this or use the auto-generated ID
            }
        )

    except Exception as e:
        logger.error(f"❌ Error recording slider interaction: {e}")
        return jsonify({"error": str(e)}), 500


@survey_bp.route("/slider-recordings/batch", methods=["POST"])
@cross_origin()
def record_slider_batch():
    """Record multiple slider interactions in a batch"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        recordings = data.get("recordings", [])

        if not session_id or not recordings:
            logger.warning("⚠️ Missing required fields for batch recording")
            return jsonify({"error": "session_id and recordings are required"}), 400

        logger.info(
            f"📊 Recording batch of {len(recordings)} slider interactions (session: {session_id})"
        )

        recording_ids = []
        timestamp = datetime.utcnow().isoformat()

        for recording in recordings:
            recording_id = str(uuid.uuid4())
            recording_ids.append(recording_id)
            DatabaseConnection.execute_query(
                """
                INSERT INTO slider_recordings (session_id, slider_key, value, min_value, max_value, created)
                VALUES (:session_id, :slider_key, :value, :min_value, :max_value, :created)
            """,
                {
                    # Remove "id": recording_id,
                    "session_id": session_id,
                    "slider_key": recording.get("slider_key"),
                    "value": recording.get("value"),
                    "min_value": recording.get("min_value"),
                    "max_value": recording.get("max_value"),
                    "created": timestamp,
                },
            )

        logger.info(
            f"✅ Successfully recorded {len(recording_ids)} slider interactions"
        )

        return jsonify(
            {
                "success": True,
                "message": f"Recorded {len(recording_ids)} slider interactions",
                "recording_ids": recording_ids,
            }
        )

    except Exception as e:
        logger.error(f"❌ Error recording slider batch: {e}")
        return jsonify({"error": str(e)}), 500


@survey_bp.route("/slider-recordings/sessions", methods=["GET"])
@cross_origin()
def get_recording_sessions():
    """Get all unique session IDs with recording counts"""
    try:
        logger.info("📊 Getting recording sessions summary...")

        sessions = DatabaseConnection.execute_query(
            """
            SELECT 
                session_id,
                COUNT(id) as recording_count,
                MIN(created) as first_recording,
                MAX(created) as last_recording
            FROM slider_recordings
            WHERE session_id IS NOT NULL
            GROUP BY session_id
            ORDER BY MAX(created) DESC
        """
        )

        logger.info(f"📊 Found {len(sessions)} unique sessions")

        session_list = []
        for i, session in enumerate(sessions):
            try:
                session_dict = {
                    "session_id": session.session_id,
                    "recording_count": session.recording_count,
                    "first_recording": session.first_recording,
                    "last_recording": session.last_recording,
                }
                session_list.append(session_dict)
                logger.debug(
                    f"📝 Session {i+1}: {session.session_id} ({session.recording_count} recordings)"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error processing session {i+1}: {e}")

        logger.info(f"✅ Successfully processed {len(session_list)} sessions")
        return jsonify({"sessions": session_list})

    except Exception as e:
        logger.error(f"❌ Error getting recording sessions: {e}")
        return jsonify({"error": str(e)}), 500


# Debug endpoints
@survey_bp.route("/debug/test", methods=["GET"])
@cross_origin()
def debug_test():
    """Test endpoint to verify API is working"""
    logger.info("🧪 Debug test endpoint called")
    return jsonify(
        {
            "message": "Survey blueprint is working!",
            "created": datetime.utcnow().isoformat(),
            "blueprint_name": "survey",
            "status": "healthy",
        }
    )


@survey_bp.route("/debug/all-recordings", methods=["GET"])
@cross_origin()
def debug_all_recordings():
    """Show all recordings in the database"""
    try:
        logger.info("📊 Getting all recordings for debug...")

        recordings = DatabaseConnection.execute_query(
            """
            SELECT id, session_id, slider_key, value, min_value, max_value, created
            FROM slider_recordings 
            ORDER BY created DESC
        """
        )

        logger.info(f"📊 Found {len(recordings)} total recordings")

        recording_list = []
        for i, recording in enumerate(recordings):
            try:
                recording_dict = {
                    "id": recording.id,
                    "session_id": recording.session_id,
                    "slider_key": recording.slider_key,
                    "value": recording.value,
                    "min_value": recording.min_value,
                    "max_value": recording.max_value,
                    "created": recording.timestamp,
                }
                recording_list.append(recording_dict)
                if i < 5:  # Log first 5
                    logger.debug(
                        f"📝 Recording {i+1}: {recording.slider_key}={recording.value}"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Error processing recording {i+1}: {e}")

        logger.info(f"✅ Successfully processed {len(recording_list)} recordings")

        return jsonify(
            {
                "total_recordings": len(recording_list),
                "recordings": recording_list,
            }
        )

    except Exception as e:
        logger.error(f"❌ Error getting all recordings: {e}")
        return jsonify({"error": str(e)}), 500


# Health check endpoint
@survey_bp.route("/slider-recordings/health", methods=["GET"])
@cross_origin()
def health_check():
    """Health check for survey service"""
    try:
        logger.info("🏥 Running health check...")

        db_healthy = DatabaseConnection.check_connection()
        logger.info(f"📊 Database connection: {'✅ OK' if db_healthy else '❌ FAILED'}")

        # Overall status
        status = "healthy" if db_healthy else "unhealthy"

        # Get table counts
        table_counts = {}
        for table_name in ["slider_configs", "slider_recordings", "metrics"]:
            try:
                count_result = DatabaseConnection.execute_query(
                    f"SELECT COUNT(*) as count FROM {table_name}", fetch_all=False
                )
                table_counts[table_name] = count_result.count if count_result else 0
                logger.info(f"📊 {table_name}: {table_counts[table_name]} rows")
            except Exception as e:
                table_counts[table_name] = f"error: {str(e)}"
                logger.warning(f"⚠️ Error checking {table_name}: {e}")

        logger.info(f"✅ Health check completed - Status: {status}")

        return jsonify(
            {
                "status": status,
                "database": "connected" if db_healthy else "disconnected",
                "table_counts": table_counts,
                "created": datetime.utcnow().isoformat(),
            }
        ), (200 if status == "healthy" else 503)

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return (
            jsonify(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "created": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )
