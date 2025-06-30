import pandas as pd
import numpy as np
import time
from config.settings import OPTIM_MODE , DATA_PATH , NUM_SLOTS 
def cleanup(x):
    original = x  # Save the original value

    # 1) Remove 'R' characters from the first 5 characters
    temp = x[:5].replace('R', '') + x[5:]

    # 2) Remove a trailing 'c', if present
    if temp.endswith('C'):
        modified = temp[:-1]
    else:
        modified = temp

    # If you want to see what changed, you can uncomment:
    # if original != modified:
    #     print(f"Changed: {original} -> {modified}")

    return modified
co = pd.read_csv(DATA_PATH + '/p_co.csv')
print('co', co )
co = co.set_index('Unnamed: 0')
co.columns =[cleanup(col) for col in co.columns]
co.index = [cleanup(col) for col in co.index] # PROBABLY REMOVE THIS AT SOME POINT


def num_conflicts(df, exam, slot):
    """
    Returns the count of conflicts caused by the specified exam if placed in the specified slot
    
    Parameters:
        exam: string of an exam
        slot: int for the proposed slot for that exam
    
    Returns:
        int: Number of conflicts
    """
    exams = df[df['slot'] == slot]['exam']  # list of exams in slot
    conflicts = co[exams].loc[exam].sum() if not exams.empty else 0  # total conflicts between exam and all exams in slot
    return conflicts


def num_b2bs(df, exam, slot):
    """
    Returns the count of back-to-backs caused by the specified exam if placed in the specified slot
    
    Parameters:
        exam: string of an exam to check
        slot: int for the slot to check exam in
        num_slots: total number of slots
    
    Returns:
        int: Number of back-to-back conflicts
    """
    count = 0
    if slot == 1:
        count = num_conflicts(df, exam, 2)
    elif slot == NUM_SLOTS:
        count = num_conflicts(df, exam, NUM_SLOTS - 1)
    else:
        count = num_conflicts(df, exam, slot - 1) + num_conflicts(df, exam, slot + 1)
    return count


def num_2i3s(df, exam, slot):
    """
    Returns the count of two-in-threes caused by the specified exam if placed in the specified slot
    
    Parameters:
        exam: string of an exam to check
        slot: int for the slot to check exam in
        num_slots: total number of slots
    
    Returns:
        int: Number of two-in-three conflicts
    """
    count = 0
    if slot <= 2:
        count = num_conflicts(df, exam, slot + 2)
    elif slot >= NUM_SLOTS-1:
        count = num_conflicts(df, exam, slot - 2)
    else:
        count = num_conflicts(df, exam, slot - 2) + num_conflicts(df, exam, slot + 2)
    return count


def create_dicts(df, exams, slots, num_slots):
    """
    Creates and returns three dictionaries: other_confs, other_b2bs, other_2i3s.

    For all three dictionaries, the keys are a tuple (i,s) where i is an exam in exams and s is a slot.
    The values are as follows:
       - other_confs: The number of conflicts between exam i and any exam in slot s not including the exams in exams.
       - other_b2bs: The number of b2bs between exam i and any exam in slot s not including the exams in exams.
       - other 2i3s: The number of two in threes between exam i and any exam in slot s not including the exams in exams.
    """
    dict_time = time.time()

    other_confs = {}
    other_b2bs = {}
    other_2i3s = {}

    dfOther = df[~df['exam'].isin(exams)]

    for slot in slots:
        for exam in exams:
            other_confs[(exam, slot)] = num_conflicts(dfOther, exam, slot)
            other_b2bs[(exam, slot)] = num_b2bs(dfOther, exam, slot)
            other_2i3s[(exam, slot)] = num_2i3s(dfOther, exam, slot)

    print('Dict create time:', round(time.time()-dict_time), 'seconds')

    return other_confs, other_b2bs, other_2i3s


def create_hot_start(sched, exams, tradeoff, assignment, slots):
    """
    Creates a hot start for the IP model
    - For the exams that are carrying over from the last layer, use the slots they were assigned from the last layer.
    - For the new exams that weren't in the last layer, order them by size and place them greedily given the exams already placed.
    """
    print('Creating hot start')
    hotStart_time = time.time()

    # Create empty dataframe
    hot = pd.DataFrame(columns=['exam', 'slot'])
    new = []  # List to hold exams not in 'assignment'
    rows_to_add = []  # List to store rows before concatenation

    for i in exams:
        if i in assignment[:, 0]:
            # Create a dictionary for the current row and append it to the list
            rows_to_add.append({'exam': i, 'slot': int(assignment[assignment[:, 0] == i][0, 1])})
        else:
            new.append(i)

    # Convert the list of dictionaries to a DataFrame and concatenate it to 'hot' all at once
    new_rows_df = pd.DataFrame(rows_to_add)
    hot = pd.concat([hot, new_rows_df], ignore_index=True)

    hot = pd.concat([sched, hot], ignore_index=True)

    rows_to_add = []  # List to store new rows before concatenation

    for i in new:
        best_slot = 0
        best_score = np.inf
        for s in slots:
            confs = num_conflicts(hot, i, s)
            b2bs = num_b2bs(hot, i, s)
            two_in_threes = num_2i3s(hot, i, s)
            score = tradeoff[0]*confs + b2bs + tradeoff[1]*two_in_threes
            if score < best_score:
                best_score = score
                best_slot = s

        # Add the best slot found for exam i to the list
        rows_to_add.append({'exam': i, 'slot': best_slot})

    # Convert the list of dictionaries to a DataFrame and concatenate it to 'hot' all at once
    new_rows_df = pd.DataFrame(rows_to_add)
    hot = pd.concat([hot, new_rows_df], ignore_index=True)

    print('Hot solution time:', round(time.time()-hotStart_time), 'seconds')

    return hot


def calculate_metrics(schedule, co):
    """
    Calculate schedule metrics: conflicts, back-to-backs, and two-in-threes
    
    Parameters:
        schedule: DataFrame with exam schedule
        co: DataFrame with exam conflicts
        
    Returns:
        tuple: (conflicts, b2bs, two_i3s)
    """
    conflicts = 0
    b2bs = 0
    two_i3s = 0
    
    # Group exams by slot
    slot_groups = schedule.groupby('slot')
    
    # Calculate conflicts within each slot
    for slot, group in slot_groups:
        exams = group['exam'].tolist()
        for i, exam1 in enumerate(exams):
            for exam2 in exams[i+1:]:
                conflicts += co[exam1][exam2]
    
    # Calculate back-to-backs and two-in-threes
    slots = sorted(schedule['slot'].unique())
    for i, slot1 in enumerate(slots[:-1]):
        exams1 = schedule[schedule['slot'] == slot1]['exam'].tolist()
        
        # Back-to-backs (consecutive slots)
        if i < len(slots) - 1 and slots[i+1] == slot1 + 1:
            exams2 = schedule[schedule['slot'] == slots[i+1]]['exam'].tolist()
            for exam1 in exams1:
                for exam2 in exams2:
                    b2bs += co[exam1][exam2]
        
        # Two-in-threes (slots 2 apart)
        if i < len(slots) - 2 and slots[i+2] == slot1 + 2:
            exams3 = schedule[schedule['slot'] == slots[i+2]]['exam'].tolist()
            for exam1 in exams1:
                for exam3 in exams3:
                    two_i3s += co[exam1][exam3]
    
    return conflicts, b2bs, two_i3s


def save_block_assignment(sched, output_file, output_dir):
    """
    Create and save a block assignment based on a layercake schedule
    
    Parameters:
        sched: DataFrame with exam schedule
        output_file: Name of output file
        output_dir: Directory to save output
    
    Returns:
        DataFrame: Processed schedule
    """
    # Ensure slot is integer type
    sched['slot'] = sched['slot'].astype(int)
    
    # Reorder blocks from 1 to num_block
    slots = list(np.sort(np.unique(sched['slot'])))
    sched['slot'] = sched['slot'].apply(lambda x: slots.index(x) + 1)
    
    # Rename columns for output
    sched = sched.rename({'exam': 'Exam Group', 'slot': 'Exam Block'}, axis='columns')
    
    # Save to CSV
    sched.to_csv(f'{output_dir}/{output_file}.csv', header=True, index=False)
    
    return sched

def  name_block_assignment(size_cutoff, frontloading, k,num_blocks):
    return OPTIM_MODE + 'size_cutoff'+str(size_cutoff)+ 'frontloading' +  str(frontloading) + 'num_blocks'+ str(num_blocks )
