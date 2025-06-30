import time
import datetime
import pandas as pd
import os
import sys
import timeit
from itertools import cycle
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import shutil
import glob

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import LICENSES, SAVE_PATH, SEMESTER, DATA_PATH
from .layercake import run_layer_cake
from .helpers import cleanup

#from layercake import run_layer_cake
#from helpers import calculate_metrics, save_block_assignment

def print_params(k, overlap, tradeoff, timelimit):
    """
    Prints the parameters used for the layer cake.
    """
    print('Algorithm starts:', datetime.datetime.now(), '\n\n')
    print('__________________________________________PARAMETERS______________________________________________________')
    print('Layer size, k:', k)
    print('Overlap:', overlap)
    print('Tradeoff params:', tradeoff)
    print('Time limit:', timelimit)
    print('__________________________________________________________________________________________________________\n\n\n')


def run_with_params(params):
    """
    Wrapper function to run layer cake with provided parameters
    
    Parameters:
        params: Tuple of parameters
    
    Returns:
        dict: Results metrics
    """
    (param_id, size_cutoff, num_reserved_for_frontloading, k_val, 
     num_blocks, license_info, timelimit)= params
    
    try:
        return run_layer_cake(
            param_id, size_cutoff, num_reserved_for_frontloading, k_val, num_blocks, 
            license_info, MODE = 'SLOW', timelimit = timelimit
        )
    except Exception as e:
        print(f"Error in run {param_id}: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            'param_id': param_id,
            'error': str(e)
        }

def run_parameter_sweep( size_cutoffs=None, k_values=None, frontloading_values=None, num_blocks_values=None, timelimit = 7200):
    """
    Run layer cake algorithm with multiple parameter combinations in parallel
    
    Parameters:
        semester: Semester identifier
        co_name: Name of conflict matrix file
        size_cutoffs: List of size thresholds for large exams
        k_values: List of layer sizes
        frontloading_values: List of numbers of blocks to reserve
        num_blocks_values: List of numbers of blocks to use
        data_dir: Directory containing input data
        output_dir: Directory for output files
        
    Returns:
        pd.DataFrame: Results of all parameter combinations
    """
    # Default parameter values if not provided
    if size_cutoffs is None:
        size_cutoffs = [200, 300]
    if k_values is None:
        k_values = [20000]
    if frontloading_values is None:
        frontloading_values = [2, 4, 5]
    if num_blocks_values is None:
        num_blocks_values = [20, 22, 24]
    
    # Generate parameter combinations
    param_combinations = []
    license_cycle = cycle(LICENSES)
    
    for size_cutoff in size_cutoffs:
        for k_val in k_values:
            for num_blocks in num_blocks_values:
                for frontloading in frontloading_values:
                    # Get the next license in the cycle
                    license_info = next(license_cycle)
                    param_id = f'size_cutoff{size_cutoff}frontloading{frontloading}k_val{k_val}num_blocks{num_blocks}'
                    
                    param_combinations.append((
                        param_id, size_cutoff, frontloading, k_val, num_blocks, 
                        license_info, timelimit
                    ))
    
    # Run parameter combinations in parallel
    start = timeit.default_timer()
    max_workers = min(len(LICENSES), len(param_combinations))
    print(f"Running {len(param_combinations)} parameter combinations with {max_workers} parallel processes")
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {executor.submit(run_with_params, params): params[0] 
                           for params in param_combinations}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_params):
            param_id = future_to_params[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed run {result['param_id']} of {len(param_combinations)}")
            except Exception as exc:
                print(f"Run {param_id} generated an exception: {exc}")
                import traceback
                print(traceback.format_exc())
    
    stop = timeit.default_timer()
    print(f'Total Runtime: {stop - start} seconds')
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{SAVE_PATH}/parameter_sweep_results.csv', index=False)
    
    return results_df



def count_conflicts(block_assignments, coenrollment_df):
    """
    Count the number of conflicts in a block assignment using coenrollment data.
    A conflict is when a student has two exams scheduled at the same slot.
    
    Args:
        block_assignments (pd.DataFrame): DataFrame with 'Exam Group' and 'Exam Block' columns
        coenrollment_df (pd.DataFrame): DataFrame with coenrollment information
    
    Returns:
        int: Number of conflicts
    """
    # Create a mapping from exam group to slot
    exam_to_block = dict(zip(block_assignments['Exam Group'], block_assignments['Exam Block']))
    #print('exam_to_block : ' , exam_to_block)
    # Count conflicts - students scheduled in the same slot
    conflicts = 0
    #print('coenrollment_df' , coenrollment_df)
    # For each pair of exams with shared students
    coenrollment_df = coenrollment_df.set_index('Unnamed: 0')
    #print('coenrollment_df.index: ' , coenrollment_df.index )
    #print(
    #    'exam_to_block ' , exam_to_block 
    #)
    #print('Exam Group' , block_assignments['Exam Group'].unique() )
    #print('ahhh' , coenrollment_df.index )
    #baa = set(block_assignments['Exam Group'].unique())
    #coo = set(coenrollment_df.index)
    #print('interseciton , ' , set(block_assignments['Exam Group'].unique()).intersection(coenrollment_df.index))
    #print('lenn , ' , len(set(block_assignments['Exam Group'].unique()).intersection(coenrollment_df.index)))
    #print('missing ', coo - baa.intersection(coo) )
    for e1 in coenrollment_df.index:
        for e2 in coenrollment_df.index:
            if coenrollment_df.at[e1,e2]>0:
                if exam_to_block[e1] == exam_to_block[e2]:
                    conflicts += coenrollment_df.at[e1,e2]
            
    return conflicts
def process_block_assignments(name=None):
    """
    Process block assignment files in the specified directory.
    Rename files with >5 conflicts by prepending 'bad_'.
    
    Returns:
        - If multiple files processed: List[bool], one entry per file processed,
          True if the file remained good (no rename), False if renamed bad.
        - If exactly one file processed: a single bool.
    """
    
    
    directory_path    = SAVE_PATH +"/blocks/"
    coenrollment_path = DATA_PATH + "/p_co.csv"
    coenrollment_df   = pd.read_csv(coenrollment_path)
    # find matching CSVs
    if name is None:
        pattern     = "*.csv"
    else:
        print('PBA name ; ' , name)
        pattern     = name
    block_files = glob.glob(os.path.join(directory_path, pattern))
    print('PATTERN : ' , pattern)
    print(f"Found {len(block_files)} block assignment files")

    results = []
    for file_path in block_files:
        #try: 
            basename = os.path.basename(file_path)
            # skip already bad files
            if basename.startswith("bad_"):
                print(f"Skipping already marked file: {basename}")
                continue

            print(f"Processing: {basename}")

            df       = pd.read_csv(file_path)
            if 'exam' in df.columns:
                df['Exam Group'] = df['exam']
            if 'slot' in df.columns:
                df['Exam Block'] = df['slot']
            df['Exam Group'] = df['Exam Group'].apply(cleanup)
            df.to_csv(file_path)
            print(
                'rewrote to ', file_path 
            )
            conflicts = 1000 
            try: 
                conflicts = count_conflicts(df, coenrollment_df)
            except: 

               print(f"  Found {conflicts} conflicts")

            if conflicts > 15:
                new_name     = f"bad_{basename}"
                new_path     = os.path.join(directory_path, new_name)
                shutil.move(file_path, new_path)
                print(f"  Renamed to: {new_name}")
                results.append(False)
            else:
                results.append(True)
        #except : 
            print("PRECOESSING WRONG " )

    # return a single bool if only one file was processed
    if len(results) == 1:
        return results[0]
    return results

def run_multi_blocks(
    overlap: float = 0.67,
    k: int = 20000,
    tradeoff: tuple = (1000, 0.5),
    timelimit: int = 2 * 3600,  # seconds
    size_cutoffs: list = None,
    k_values: list = None,
    frontloading_values: list = None,
    num_blocks_values: list = None,
):
    # set up sensible defaults if caller didn’t provide them
    if size_cutoffs is None:
        size_cutoffs = [200, 300]
    if k_values is None:
        k_values = [k]
    if frontloading_values is None:
        frontloading_values = [2, 4, 5]
    if num_blocks_values is None:
        num_blocks_values = [20, 22, 24]

    # echo parameters
    print_params(k, overlap, tradeoff, timelimit)

    # run the sweep
    results = run_parameter_sweep(

        size_cutoffs=size_cutoffs,
        k_values=k_values,
        frontloading_values=frontloading_values,
        num_blocks_values=num_blocks_values,
        timelimit = timelimit
        # you could also pass overlap, tradeoff, timelimit in here
    )

    # find & print best
    if not results.empty and 'conflicts' in results.columns:
        best_idx    = results['conflicts'].idxmin()
        best_params = results.loc[best_idx]
        print("\nBest parameter combination:")
        for col in results.columns:
            print(f"{col}: {best_params[col]}")
    process_block_assignments()
    # still process the blocks directory
#process_block_assignments()
#run_with_params(('param_id', 300, 3, 20000, 
#     22, LICENSES[0], 'sp25', 'p_co', '/home/asj53/BOScheduling/data/', '/home/asj53/BOScheduling/results/blocks'))