
from flask import Blueprint, jsonify, request
from .models import SliderConfig, SliderRecording
from ..extensions import db
from sqlalchemy import text
import uuid
from datetime import datetime

from flask_cors import cross_origin

survey_bp = Blueprint('survey', __name__)

# Add these routes to your survey/routes.py file

@survey_bp.route('/init-tables', methods=['POST'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def init_tables():
    """Initialize database tables for slider functionality"""
    try:
        print("🏗️ Creating database tables...")
        
        # Create all tables defined in models
        db.create_all()
        
        print("✅ Tables created successfully")
        
        return jsonify({
            'success': True,
            'message': 'Database tables initialized successfully',
            'timestamp': datetime.utcnow().isoformat()
        }), 201
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Failed to initialize database tables'
        }), 500

@survey_bp.route('/debug/check-tables', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def debug_check_tables():
    """Check if required tables exist"""
    try:
        # Import here to avoid circular imports
        from sqlalchemy import text
        
        with db.engine.begin() as conn:
            # Get all table names
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table'
                ORDER BY name
            """))
            tables = [row[0] for row in result.fetchall()]
        
        return jsonify({
            'success': True,
            'all_tables': tables,
            'slider_configs_exists': 'slider_configs' in tables,
            'slider_recordings_exists': 'slider_recordings' in tables,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Existing slider config routes
@survey_bp.route('/slider-configs', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def list_configs():
    """Get all saved configurations"""
    try:
        configs = SliderConfig.query.order_by(SliderConfig.timestamp.desc()).all()
        return jsonify({'configs': [c.to_dict() for c in configs]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@survey_bp.route('/slider-configs', methods=['POST'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def create_config():
    """Save a new configuration"""
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name') or not data.get('thresholds'):
            return jsonify({'message': 'Name and thresholds are required'}), 400
        
        # Check if name already exists
        existing = SliderConfig.query.filter_by(name=data['name']).first()
        if existing:
            return jsonify({'message': 'Configuration name already exists'}), 409
        
        config = SliderConfig(
            name=data['name'],
            description=data.get('description', ''),
            thresholds=data['thresholds']
        )
        
        db.session.add(config)
        db.session.commit()
        
        return jsonify(config.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@survey_bp.route('/slider-configs/<int:id>', methods=['PUT', 'DELETE'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def modify_config(id):
    """Update or delete a configuration"""
    try:
        config = SliderConfig.query.get_or_404(id)
        
        if request.method == 'PUT':
            data = request.get_json()
            if 'name' in data:
                config.name = data['name']
            if 'description' in data:
                config.description = data['description']
            if 'thresholds' in data:
                config.thresholds = data['thresholds']
            
            db.session.commit()
            return jsonify(config.to_dict())
        
        else:  # DELETE
            db.session.delete(config)
            db.session.commit()
            return '', 204
            
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# New routes for recording slider interactions
@survey_bp.route('/slider-recordings', methods=['POST'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def record_slider_interaction():
    """Record a single slider interaction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'message': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['slider_key', 'value', 'min_value', 'max_value']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({'message': f'Missing required fields: {missing_fields}'}), 400
        
        recording = SliderRecording(
            session_id=data.get('session_id'),
            slider_key=data['slider_key'],
            value=float(data['value']),
            min_value=float(data['min_value']),
            max_value=float(data['max_value'])
        )
        
        db.session.add(recording)
        db.session.commit()
        
        return jsonify(recording.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error saving recording', 'error': str(e)}), 500

@survey_bp.route('/slider-recordings/batch', methods=['POST'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def record_slider_batch():
    """Record multiple slider positions at once"""
    print("RECORD SLIDERER ER ")
    
    try:
        data = request.get_json()
        
        if not data or not data.get('recordings'):
            return jsonify({'message': 'recordings array is required'}), 400
        
        session_id = data.get('session_id') or str(uuid.uuid4())
        recordings_data = data.get('recordings', [])
        print('recordings_data. , ' , recordings_data )
        recordings = []
        for item in recordings_data:
            print("ITEM " , item)
            # Validate required fields for each recording
            required_fields = ['slider_key', 'value', 'min_value', 'max_value']
            if not all(field in item for field in required_fields):
                continue
                
            recording = SliderRecording(
                session_id=session_id,
                slider_key=item['slider_key'],
                value=float(item['value']),
                min_value=float(item['min_value']),
                max_value=float(item['max_value'])
            )
            recordings.append(recording)
        
        if recordings:
            db.session.add_all(recordings)
            db.session.commit()
        
        return jsonify({
            'message': f'Recorded {len(recordings)} slider interactions',
            'session_id': session_id,
            'recordings': [r.to_dict() for r in recordings]
        }), 201
            
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'message': 'Error saving batch recordings',
            'error': str(e)
        }), 500

@survey_bp.route('/slider-recordings', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def get_slider_recordings():
    """Get all slider recordings with optional filtering"""
    try:
        session_id = request.args.get('session_id')
        slider_key = request.args.get('slider_key')
        limit = request.args.get('limit', type=int)
        
        query = SliderRecording.query
        
        if session_id:
            query = query.filter_by(session_id=session_id)
        if slider_key:
            query = query.filter_by(slider_key=slider_key)
        
        query = query.order_by(SliderRecording.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        recordings = query.all()
        
        return jsonify({
            'recordings': [r.to_dict() for r in recordings],
            'total': len(recordings)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@survey_bp.route('/slider-recordings/sessions', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def get_recording_sessions():
    """Get all unique session IDs with recording counts"""
    try:
        from sqlalchemy import func
        
        sessions = db.session.query(
            SliderRecording.session_id,
            func.count(SliderRecording.id).label('recording_count'),
            func.min(SliderRecording.timestamp).label('first_recording'),
            func.max(SliderRecording.timestamp).label('last_recording')
        ).group_by(SliderRecording.session_id).all()
        
        return jsonify({
            'sessions': [{
                'session_id': s.session_id,
                'recording_count': s.recording_count,
                'first_recording': s.first_recording.isoformat() if s.first_recording else None,
                'last_recording': s.last_recording.isoformat() if s.last_recording else None
            } for s in sessions]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Debug endpoints for testing
@survey_bp.route('/debug/test', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def debug_test():
    """Test endpoint to verify API is working"""
    return jsonify({
        'message': 'Survey blueprint is working!',
        'timestamp': datetime.utcnow().isoformat(),
        'blueprint_name': 'survey'
    })

@survey_bp.route('/debug/db-test', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def debug_db_test():
    """Test database connection"""
    try:
        # Test database connection
        recording_count = SliderRecording.query.count()
        config_count = SliderConfig.query.count()
        
        # Test if we can create a simple record
        test_record = SliderRecording(
            session_id='debug-test',
            slider_key='test_key',
            value=0.5,
            min_value=0.0,
            max_value=1.0
        )
        
        db.session.add(test_record)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'recording_count': recording_count + 1,
            'config_count': config_count,
            'test_record_id': test_record.id,
            'message': 'Database is working correctly'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'status': 'error',
            'error': str(e),
            'message': 'Database connection failed'
        }), 500

@survey_bp.route('/debug/all-recordings', methods=['GET'])
@cross_origin(origins=['http://localhost:5173', 'http://127.0.0.1:5173'])
def debug_all_recordings():
    """Show all recordings in the database"""
    try:
        recordings = SliderRecording.query.order_by(SliderRecording.timestamp.desc()).all()
        return jsonify({
            'total_recordings': len(recordings),
            'recordings': [r.to_dict() for r in recordings]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500