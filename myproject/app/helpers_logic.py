import os, glob, re, json, subprocess
from functools import lru_cache
import pandas as pd
from sqlalchemy import create_engine, text
from typing import List
from config import Config
#from UI.pages.schedule_plots import get_plot, last_day

SAVE_PATH = Config.SAVE_PATH
UI_PATH   = Config.UI_PATH
NUM_SLOTS = Config.NUM_SLOTS
def extract_i_number(path: str) -> int:
    fname = os.path.basename(path)
    m = re.search(r"i(\d+)", fname)
    if not m:
        raise ValueError(f"No 'i<digits>' segment found in {path!r}")
    return int(m.group(1))
# Database configuration
def get_db_path():
    """Get database path, handling both script and notebook contexts."""
    try:
        # When running as a script
        script_dir = os.path.dirname(__file__)
        return os.path.join(script_dir, '..', 'data', 'schedules.db')
    except NameError:
        # When running in notebook or interactive environment
        # Try different possible locations
        possible_paths = [
            'schedules.db',  # Current directory
            'data/schedules.db',  # data subdirectory
            '../data/schedules.db',  # parent/data
            '../../data/schedules.db',  # grandparent/data
            '/home/asj53/BOScheduling/schedules.db',  # Absolute path guess
            'myproject/data/schedules.db'  # Project structure
        ]
        
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path
        
        # If none found, return the first option
        return os.path.abspath(possible_paths[0])

DB_PATH = get_db_path()
DB_URL = f"sqlite:///{DB_PATH}"
SEMESTER = "sp25"  # Update as needed
engine = create_engine(DB_URL, echo=False)

def run_one_iteration(prefs: list):
    """
    Save prefs, build and submit a SLURM script.
    """
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

python -u run_EIUU.py \
    --prefs {prefs_path} \
    --n_iterations 1
"""
    script_path = os.path.join(SAVE_PATH, "submit_one_iter.slurm")
    with open(script_path, "w") as f:
        f.write(slurm_txt)

    res = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Error submitting SLURM job: {res.stderr.strip()}")
    return res.stdout.strip()

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
    if filename.endswith('.csv'):
        schedule_id = filename[:-4]  # Remove .csv
    elif filename.endswith('.png'):
        schedule_id = filename[:-4]  # Remove .png
        # Remove distribution suffix if present
        if schedule_id.endswith('_dist'):
            schedule_id = schedule_id[:-5]  # Remove _dist
        elif schedule_id.endswith('_distribution'):
            schedule_id = schedule_id[:-13]  # Remove _distribution
    else:
        # Just remove the extension
        schedule_id = os.path.splitext(filename)[0]
    
    return schedule_id

def get_schedule_files(date_prefix: str, metrics_dir: str = None, semester: str = SEMESTER) -> List[str]:
    """Get schedule IDs from database that match the given date prefix."""
    print('Getting schedules for prefix:', date_prefix)
    
    try:
        with engine.begin() as conn:
            if semester:
                # Get schedules that have metrics with the specified semester
                result = conn.execute(text("""
                    SELECT DISTINCT m.schedule_id 
                    FROM metrics m
                    WHERE m.schedule_id LIKE :prefix 
                    AND m.semester = :semester
                    ORDER BY m.schedule_id
                """), {
                    "prefix": f"{date_prefix}_%",
                    "semester": semester
                })
            else:
                # Get all schedules from metrics, regardless of semester
                result = conn.execute(text("""
                    SELECT DISTINCT schedule_id 
                    FROM metrics
                    WHERE schedule_id LIKE :prefix 
                    ORDER BY schedule_id
                """), {
                    "prefix": f"{date_prefix}_%"
                })
            
            schedule_ids = [row.schedule_id for row in result]
            print(f"Found {len(schedule_ids)} schedules from database matching prefix '{date_prefix}_'")
            return schedule_ids
            
    except Exception as e:
        print(f"Database query failed: {e}")
        print("Falling back to file system search...")
        
        # Fallback to original file system approach
        metrics_dir = metrics_dir or os.path.join(SAVE_PATH, 'metrics')
        pattern = os.path.join(metrics_dir, f"{date_prefix}*.csv")
        files = [extract_schedule_id_from_filename(p) for p in glob.glob(pattern)]
        print(f"Found {len(files)} schedules from file system matching prefix '{date_prefix}'")
        return files

def plot_exists_on_disk(schedule_id: str, plot_suffix: str = '', plots_dir: str = None) -> bool:
    """Check if plot file exists on disk."""
    if plots_dir is None:
        plots_dir = os.path.join(UI_PATH, 'pages', 'plots')  # Updated to match your structure
    
    plot_filename = f"{schedule_id}{plot_suffix}.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    return os.path.exists(plot_path)

@lru_cache(maxsize=1)
def generate_plots_for_files(date_prefix: str):
    """Generate missing schedule and distribution plots for all files matching prefix."""
    schedule_ids = get_schedule_files(date_prefix)
    print("PLOT SCHEDULE IDs:", schedule_ids)
    
    for schedule_id in schedule_ids:
        # Check and generate regular schedule plot
        if not plot_exists_on_disk(schedule_id):
            print(f"Generating schedule plot for {schedule_id}")
            # Note: You'll need to import get_plot or pass the filename instead of schedule_id
            # get_plot(f"{schedule_id}.csv", schedule_id)  # Assuming get_plot function exists
        
        # Check and generate distribution plot
        if not plot_exists_on_disk(schedule_id, '_dist'):
            print(f"Generating distribution plot for {schedule_id}")
            # Note: You'll need to import last_day or pass the filename instead of schedule_id
            # last_day(f"{schedule_id}.csv", schedule_id)  # Assuming last_day function exists

@lru_cache(maxsize=1)
def load_schedule_data_basic(date_prefix: str) -> list[dict]:
    """Load schedule data from database instead of CSV files."""
    schedule_ids = get_schedule_files(date_prefix)
    
    if not schedule_ids:
        print(f"No schedules found for prefix: {date_prefix}")
        return []
    
    print(f"Loading data for {len(schedule_ids)} schedules from database...")
    
    # Column mappings from database to expected format
    param_cols = ['size_cutoff', 'reserved', 'num_blocks', 'large_block_size',
                  'large_exam_weight', 'large_block_weight', 'large_size_1',
                  'large_cutoff_freedom']
    
    # Map database column names to display names for metrics
    metrics_mapping = {
        'conflicts': 'conflicts',
        'quints': 'quints',
        'quads': 'quads',
        'four_in_five': 'four in five slots',
        'three_in_four': 'three in four slots',
        'two_in_three': 'two in three slots',
        'singular_late': 'singular late exam',
        'two_large_gap': 'two exams, large gap',
        'avg_max': 'avg_max'
    }
    
    data = []

    with engine.begin() as conn:
        for schedule_id in schedule_ids:
            # Get metrics data from database
            result = conn.execute(text("""
                SELECT m.*, s.display_name, s.max_slot
                FROM metrics m
                LEFT JOIN schedules s ON m.schedule_id = s.schedule_id
                WHERE m.schedule_id = :schedule_id
            """), {"schedule_id": schedule_id})
            
            row = result.fetchone()
            if not row:
                print(f"Warning: No metrics found for {schedule_id}")
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
                if hasattr(row, db_col) and getattr(row, db_col) is not None:
                    metrics[display_name] = getattr(row, db_col)
            
            # Add computed metrics
            triple_24h = getattr(row, 'triple_in_24h', 0) or 0
            triple_same_day = getattr(row, 'triple_in_same_day', 0) or 0
            evening_morning_b2b = getattr(row, 'evening_morning_b2b', 0) or 0
            other_b2b = getattr(row, 'other_b2b', 0) or 0
            
            metrics['reschedules'] = triple_24h + triple_same_day
            metrics['back_to_back'] = evening_morning_b2b + other_b2b
            
            # Build params dict
            params = {}
            for col in param_cols:
                if hasattr(row, col) and getattr(row, col) is not None:
                    params[col] = getattr(row, col)
            
            # Get slot data from database
            slot_result = conn.execute(text("""
                SELECT slot_number, present
                FROM slots
                WHERE schedule_id = :schedule_id
                ORDER BY slot_number
            """), {"schedule_id": schedule_id})
            
            # Build columns dict from slot data
            columns = {i: 0 for i in range(1, NUM_SLOTS+1)}  # Initialize all slots to 0
            print('columns '  , columns)
            slot_rows = slot_result.fetchall()
            print("SLOT ROWS  , " , slot_rows)
            if slot_rows:
                # Use database slot data
                for slot_row in slot_rows:
                    slot_num = int(slot_row.slot_number)
                    print('slot_num , ' , slot_num) 
                    print('slot_row.present , ' , slot_row.present) 
                    if slot_num in columns:  # Only include slots within NUM_SLOTS range
                        columns[slot_num] = 1#int(slot_row.present)
            else:
                # Fallback to CSV if no slot data in database
                print(f"Warning: No slot data in database for {schedule_id}, trying CSV fallback...")
                try:
                    fname = f"{schedule_id}.csv"
                    df_sched = pd.read_csv(os.path.join(SAVE_PATH, 'schedules', fname))
                    slots = sorted(df_sched['slot'].unique())
                    columns = {str(i): (1 if i in slots else 0) for i in range(1, NUM_SLOTS+1)}
                except FileNotFoundError:
                    print(f"Warning: Schedule CSV also not found for {schedule_id}, using empty slots")
                    # columns already initialized to all 0s
                except Exception as e:
                    print(f"Warning: Error reading schedule CSV for {schedule_id}: {e}")
                    # columns already initialized to all 0s
            
            data.append({
                'name': display,
                'basename': schedule_id,
                'metrics': metrics,
                'params': params,
                'columns': columns,
            })
            

    
    print(f"Successfully loaded {len(data)} schedules from database")
    return data