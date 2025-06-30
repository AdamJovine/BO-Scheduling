from flask import Blueprint, jsonify, request, send_from_directory, abort, make_response
import os
from ..extensions import db
from config import Config
from sqlalchemy import text
from ..helpers_logic import (get_schedule_files,
                             generate_plots_for_files,
                             load_schedule_data_basic,
                             run_one_iteration)
from ..helpers_logic import engine
helpers_bp = Blueprint('helpers', __name__)

# Define plots directory
PLOTS_DIR = '/home/asj53/BOScheduling/UI/pages/plots'

@helpers_bp.route('/files/<date_prefix>', methods=['GET'])
def files(date_prefix):
    return jsonify(get_schedule_files(date_prefix))

@helpers_bp.route('/schedules/<date_prefix>', methods=['GET'])
def schedules(date_prefix):
    #generate_plots_for_files(date_prefix)
    return jsonify(load_schedule_data_basic(date_prefix))

@helpers_bp.route('/run', methods=['POST'])
def run():
    prefs = request.json.get('prefs', [])
    try:
        msg = run_one_iteration(prefs)
        return jsonify({'status':'submitted','detail':msg}), 202
    except Exception as e:
        return jsonify({'status':'error','detail':str(e)}), 500

@helpers_bp.route('/images/<filename>')
def serve_image(filename):
    """Serve plot images from the plots directory."""
    print(f'Image request for: {filename}')
    print(f'Looking in directory: {PLOTS_DIR}')
    
    # Security check: only allow .png files
    if not filename.lower().endswith('.png'):
        print(f'Invalid file type: {filename}')
        abort(404)
    
    # Check if plots directory exists
    if not os.path.exists(PLOTS_DIR):
        print(f'Plots directory does not exist: {PLOTS_DIR}')
        abort(404)
    
    # Check if the specific file exists
    file_path = os.path.join(PLOTS_DIR, filename)
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        # List available files for debugging
        try:
            available_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
            print(f'Available PNG files: {available_files[:10]}...')  # Show first 10
        except Exception as e:
            print(f'Error listing directory: {e}')
        abort(404)
    
    print(f'Serving file: {file_path}')
    try:
        return send_from_directory(PLOTS_DIR, filename)
    except Exception as e:
        print(f'Error serving file: {e}')
        abort(500)

@helpers_bp.route('/images/debug')
def debug_images():
    """Debug endpoint to list available images."""
    try:
        if not os.path.exists(PLOTS_DIR):
            return jsonify({
                'error': f'Plots directory does not exist: {PLOTS_DIR}',
                'plots_dir': PLOTS_DIR
            })
        
        png_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')]
        
        return jsonify({
            'plots_dir': PLOTS_DIR,
            'total_png_files': len(png_files),
            'sample_files': png_files[:20],  # Show first 20 files
            'directory_exists': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'plots_dir': PLOTS_DIR
        })
@helpers_bp.route('/download/schedules/<schedule_id>')
def download_schedule(schedule_id):
    """Download schedule data as CSV from schedule_details table."""
    
    
    
    with engine.begin() as conn:
        # Query schedule_details table - now with faculty column
        result = conn.execute(text("""
            SELECT schedule_id, exam_id, slot, faculty, semester
            FROM schedule_details
            WHERE schedule_id = :schedule_id
            ORDER BY slot, exam_id
        """), {"schedule_id": schedule_id})
        
        rows = result.fetchall()
        
        if not rows:
            abort(404, description=f"No schedule data found for {schedule_id}")
        
        # Create CSV content with faculty column
        csv_content = "schedule_id,exam_id,slot,faculty,semester\n"
        for row in rows:
            # Handle empty faculty field
            faculty = row.faculty if row.faculty else ""
            csv_content += f"{row.schedule_id},{row.exam_id},{row.slot},{faculty},{row.semester}\n"
        
        # Create response with CSV content
        response = make_response(csv_content)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename="{schedule_id}.csv"'
        
        print(f"Downloaded schedule {schedule_id} with {len(rows)} exam assignments")
        return response

@helpers_bp.route('/download/schedules/<schedule_id>/debug')
def debug_schedule_download(schedule_id):
    """Debug endpoint to see what data exists for a schedule."""
    try:
        # Import the engine from helpers_logic to use the same database
        from ..helpers_logic import engine
        
        with engine.begin() as conn:
            # Check if schedule_details table exists and has data
            try:
                # Count total rows for this schedule
                count_result = conn.execute(text("""
                    SELECT COUNT(*) as total
                    FROM schedule_details
                    WHERE schedule_id = :schedule_id
                """), {"schedule_id": schedule_id})
                
                total_count = count_result.fetchone().total
                
                # Get sample data
                sample_result = conn.execute(text("""
                    SELECT schedule_id, exam_id, slot, faculty, semester
                    FROM schedule_details
                    WHERE schedule_id = :schedule_id
                    ORDER BY slot, exam_id
                    LIMIT 10
                """), {"schedule_id": schedule_id})
                
                sample_rows = sample_result.fetchall()
                
                return jsonify({
                    'schedule_id': schedule_id,
                    'total_rows': total_count,
                    'has_data': total_count > 0,
                    'sample_data': [
                        {
                            'schedule_id': row.schedule_id,
                            'exam_id': row.exam_id,
                            'slot': row.slot,
                            'faculty': row.faculty,
                            'semester': row.semester
                        } for row in sample_rows
                    ],
                    'database_engine': str(engine.url),
                    'table_exists': True
                })
                
            except Exception as table_error:
                # If table doesn't exist or other error
                return jsonify({
                    'schedule_id': schedule_id,
                    'error': str(table_error),
                    'table_exists': False,
                    'database_engine': str(engine.url),
                    'suggestion': 'Make sure schedule_details table exists and is populated'
                })
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'schedule_id': schedule_id,
            'suggestion': 'Check database connection and table structure'
        })