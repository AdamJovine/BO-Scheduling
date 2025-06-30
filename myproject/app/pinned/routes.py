from flask import Blueprint, jsonify, request
from sqlalchemy import text
from datetime import datetime
import json
from ..helpers_logic import engine

pinned_bp = Blueprint('pinned', __name__)

@pinned_bp.route('/pinned-schedules/<user_id>', methods=['GET'])
def get_pinned_schedules(user_id):
    try:
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT id, user_id, schedule_id, name, data, created 
                FROM pinned_schedules 
                WHERE user_id = :user_id 
                ORDER BY created DESC
            """), {"user_id": user_id})
            
            pins = result.fetchall()
        
        pinned_schedules = []
        for pin in pins:
            pinned_schedules.append({
                'id': pin.id,
                'user_id': pin.user_id,
                'sched_id': pin.schedule_id,
                'name': pin.name,
                'data': json.loads(pin.data) if pin.data else None,
                'created': pin.created
            })
        
        return jsonify({
            'success': True,
            'pinned_schedules': pinned_schedules
        })
        
    except Exception as e:
        print(f"Error in get_pinned_schedules: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@pinned_bp.route('/pinned-schedules/<user_id>/<schedule_id>', methods=['POST'])
def pin_schedule(user_id, schedule_id):
    try:
        data = request.get_json()
        schedule_name = data.get('schedule_name', '') if data else ''
        schedule_data = json.dumps(data.get('schedule_data', {})) if data else '{}'
        created_time = datetime.now().isoformat()
        
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO pinned_schedules 
                (user_id, schedule_id, name, data, created) 
                VALUES (:user_id, :schedule_id, :name, :data, :created)
            """), {
                "user_id": user_id,
                "schedule_id": schedule_id,
                "name": schedule_name,
                "data": schedule_data,
                "created": created_time
            })
        
        return jsonify({'success': True}), 201
        
    except Exception as e:
        print(f"Error in pin_schedule: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@pinned_bp.route('/pinned-schedules/<user_id>/<schedule_id>', methods=['DELETE'])
def unpin_schedule(user_id, schedule_id):
    try:
        with engine.begin() as conn:
            # Check if the pin exists
            result = conn.execute(text("""
                SELECT id FROM pinned_schedules 
                WHERE user_id = :user_id AND schedule_id = :schedule_id
            """), {"user_id": user_id, "schedule_id": schedule_id})
            
            pin = result.fetchone()
            if not pin:
                return jsonify({
                    'success': False, 
                    'error': 'Schedule not found'
                }), 404
            
            # Delete the pin
            conn.execute(text("""
                DELETE FROM pinned_schedules 
                WHERE user_id = :user_id AND schedule_id = :schedule_id
            """), {"user_id": user_id, "schedule_id": schedule_id})
        
        return jsonify({'success': True}), 200
        
    except Exception as e:
        print(f"Error in unpin_schedule: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@pinned_bp.route('/test-pinned', methods=['GET'])
def test_pinned():
    try:
        # Test database connection and table existence
        with engine.begin() as conn:
            result = conn.execute(text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='pinned_schedules'
            """))
            table_exists = result.fetchone()
        
        return jsonify({
            'success': True,
            'message': 'Pinned schedules API is working!',
            'database_connected': True,
            'pinned_table_exists': bool(table_exists),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500