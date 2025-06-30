import pandas as pd
import numpy as np
import time
import datetime
from itertools import product
from gurobipy import Model, GRB, quicksum, Env
import os
from config.settings import SAVE_PATH, DATA_PATH, EMPTY_BLOCKS, BA_TIME
from .helpers import create_dicts, create_hot_start, save_block_assignment,  calculate_metrics

#from helpers import create_dicts, create_hot_start, save_block_assignment,  calculate_metrics
co = pd.read_csv(DATA_PATH + '/p_co.csv').set_index('Unnamed: 0')
def IP(exams, large_exams, late_slots, last_slot_lim, tradeoff, other_confs, other_b2bs, other_2i3s, fl_pens, hot_starts=None, env=None, slots_to_use=None, param_id = 'cake'):
    """
    Integer Programming model for exam scheduling
    
    Parameters:
        exams: List of exams to schedule
        large_exams: List of large exams to prioritize in early slots
        late_slots: List of slots that are designated as late
        last_slot_lim: Limit for the last slot
        tradeoff: Tuple of (conf_weight, two_in_three_weight)
        other_confs: Dictionary of conflicts with exams outside the current layer
        other_b2bs: Dictionary of back-to-back conflicts with exams outside the current layer
        other_2i3s: Dictionary of two-in-three conflicts with exams outside the current layer
        timelimit: Time limit for the IP solver in seconds
        fl_pens: Dictionary of front-loading penalties by slot
        hot_starts: DataFrame with hot start solution
        env: Gurobi environment
        slots_to_use: List of slots to use
        
    Returns:
        tuple: (assignment, last_slot_lim)
    """
    size = pd.read_csv(DATA_PATH + '/exam_sizes.csv').set_index('exam')['size'].to_dict()
    

    buildTime_start = time.time()

    # Create model
    m = Model(env=env)

    # Define variables
    x = m.addVars(product(exams, slots_to_use), vtype=GRB.BINARY)   # Xis indicates that exam i is in slot s
    y = m.addVars(product(exams, exams))   # Yij indicates that exam i and exam j are in the same slot
    z = m.addVars(product(exams, exams))   # Zij indicates that exam i and exam j are in consecutive slots
    w = m.addVars(product(exams, exams))   # Wij indicates that exam i and exam j are two slots apart
    v = m.addVars(slots_to_use)   # Vij indicates the amount of students taking an exam in slot s.
    
    #print('exams:', exams)
    #print('slots_to_use:', slots_to_use)
    
    # Define constraints
    m.addConstrs(y[i, j] >= x[i, s] + x[j, s] - 1 for s in slots_to_use for i in exams for j in exams)   # Defining the y variables
    m.addConstrs(y[i, j] >= 0 for i in exams for j in exams)
    
    # Calculate appropriate back-to-back and 2-in-3 constraints
    # based on the actual slots being used
    valid_start_slots_b2b = [s for s in slots_to_use if s+1 in slots_to_use]
    valid_start_slots_2i3 = [s for s in slots_to_use if s+2 in slots_to_use]
    
    #print('valid_start_slots_b2b:', valid_start_slots_b2b)
    m.addConstrs(z[i, j] >= x[i, s] + x[j, s+1] - 1 
                for s in valid_start_slots_b2b for i in exams for j in exams)  # Defining the z variables
    m.addConstrs(z[i, j] >= 0 for i in exams for j in exams)
    
    #print('valid_start_slots_2i3:', valid_start_slots_2i3)
    m.addConstrs(w[i, j] >= x[i, s] + x[j, s+2] - 1 
                for s in valid_start_slots_2i3 for i in exams for j in exams)  # Defining the w variables

    m.addConstrs(w[i, j] >= 0 for i in exams for j in exams)
    #print('slots_to_use ' , slots_to_use )
    #print('size: ', size)
    #print('exams: ' , exams)
    #print('1R-6098C : ',  '1R-6098C' in exams )
    #print('co ', co)
    #print("co['1R-6098C']['1R-6098C']" , co['1R-6098C']['1R-6098C']) 
    m.addConstrs(quicksum(x[i, s]*size[i] for i in exams) == v[s] for s in slots_to_use)   # Defining the v variables
    #print('size[i] : ' , size)

    m.addConstrs(quicksum(x[i, s] for s in slots_to_use) == 1 for i in exams)   # One slot per exam
    #print('other_confs ; ' , other_confs)
    # Objective function components
    number_confs = quicksum(0.5*y[i, j]*co[i][j] for i in exams for j in exams) + \
                  quicksum(x[i, s]*other_confs[(i, s)] for i in exams for s in slots_to_use)
    #print('other_b2bs : ', other_b2bs )
    number_b2bs = quicksum(z[i, j]*co[i][j] for i in exams for j in exams) + \
                 quicksum(x[i, s]*other_b2bs[(i, s)] for i in exams for s in slots_to_use)
    #print('other_2i3s ' , other_2i3s )
    number_2i3s = quicksum(w[i, j]*co[i][j] for i in exams for j in exams) + \
                 quicksum(x[i, s]*other_2i3s[(i, s)] for i in exams for s in slots_to_use)
    #print('fl_pens ; ' , fl_pens )
    front_load = quicksum(x[i, s] * fl_pens[s] for i in large_exams for s in late_slots)

    # Set objective function
    m.setObjective(tradeoff[0]*number_confs + number_b2bs + tradeoff[1]*number_2i3s + front_load, GRB.MINIMIZE)
    
    # Set hot start if provided
    if isinstance(hot_starts, pd.DataFrame):
        hots = hot_starts
        for ind in product(exams, slots_to_use):
            i = ind[0]
            s = ind[1]
            if hots[hots['exam']==i]['slot'].tolist()[0] == s:
                x[ind].setAttr('Start', 1)
            else:
                x[ind].setAttr('Start', 0)
        for ind in product(exams, exams):
            i = ind[0]
            j = ind[1]
            slot_i = hots[hots['exam']==i]['slot'].tolist()[0]
            slot_j = hots[hots['exam']==j]['slot'].tolist()[0]
            if slot_i == slot_j:
                y[ind].Start = 1
            else:
                y[ind].Start = 0
            if abs(slot_i - slot_j) == 1:
                z[ind].Start = 1
            else:
                z[ind].Start = 0
            if abs(slot_i - slot_j) == 2:
                w[ind].Start = 1
            else:
                w[ind].Start = 0

    # Set model parameters
    m.update()
    print('Model build time:', round(time.time()-buildTime_start), 'seconds')
    m.Params.Timelimit = BA_TIME
    m.Params.SolFiles = f'/home/asj53/BOScheduling/block_assignment/Incumbents/inc_{os.getpid()}'   # record incumbent files

    # Optimize model
    m.write(os.path.join(SAVE_PATH, f'{param_id}.lp'))
    m.write(os.path.join(SAVE_PATH, f'{param_id}.mps'))
    m.update()
    m.optimize()
    

    # Extract solution
    assignment = []  # The new assignment, including the exams that didn't change slots
    for key in x.keys():
        if x[key].x == 1:
            assignment.append(list(key))
    assignment = np.array(assignment)

    return assignment, last_slot_lim


def run_layer_cake(param_id, size_cutoff, num_reserved_for_frontloading, k_val, num_blocks, license_info, 
                    overlap=0.67, tradeoff=(1000, 0.5),  MODE = 'SLOW' ):
    """
    Run layer cake algorithm with specific parameters. In FAST mode, skip IP and
    return a random-but-valid schedule and random integer metrics.
    """
    # first, load the minimal data needed to know your exams and slots:
    exam_sizes =  pd.read_csv(DATA_PATH + '/exam_sizes.csv')
    exam_list = exam_sizes['exam'].to_list()

    slots_to_use = [
        i
        for i in range(1, num_blocks + len(EMPTY_BLOCKS) + 1)
        if i not in EMPTY_BLOCKS
    ]
        
    if MODE == 'FAST':
        # --- FAST MODE: random assignment + random metrics ---
        # randomly assign each exam to a slot
        sched = pd.DataFrame({
            'exam': exam_list,
            'slot': np.random.choice(slots_to_use, size=len(exam_list))
        })
        # optionally save this dummy assignment
        save_block_assignment(sched, f'{param_id}', SAVE_PATH)
        
        # generate random integer metrics
        results = {
            'param_id': param_id,
            'size_cutoff': size_cutoff,
            'k': k_val,
            'num_blocks': num_blocks,
            'num_reserved_for_frontloading': num_reserved_for_frontloading,
            'conflicts': int(np.random.randint(0, 10)),
            'b2bs':     int(np.random.randint(0, 10)),
            'two_i3s':  int(np.random.randint(0, 10)),
        }
        print(f"[FAST MODE] Returning dummy results: {results}")
        return results
    # ----------------------------------------------------------
    
    # FULL MODE: actually run the IP
    env = Env(params=license_info)
    overlap_count = round(overlap * k_val)
    
    print(f'  Starting run {param_id}:')
    print(f'  Size cutoff: {size_cutoff}, Layer size (k): {k_val}, Blocks: {num_blocks}')
    print(f'  Reserved for frontloading: {num_reserved_for_frontloading}, Overlap count: {overlap_count}')
    print('Slots to use:', slots_to_use)
    
    late_slots = list(range(num_blocks - num_reserved_for_frontloading + 1, num_blocks + 1))
    fl_pens = dict(zip(slots_to_use, np.array(slots_to_use)**3))
    
    exam_sizes = exam_sizes.sort_values('size', ascending=False).reset_index(drop=True)
    exam_sizes['cum_sum'] = np.cumsum(exam_sizes['size'])
    
    sched = pd.DataFrame(columns=['exam', 'slot'])
    last_slot_lim = np.inf
    i = 0
    num_exams = len(exam_sizes)
    
    while i < num_exams:
        cur = 0 if i == 0 else exam_sizes.at[i-1, 'cum_sum']
        layer_df = exam_sizes[(exam_sizes['cum_sum'] > cur) & (exam_sizes['cum_sum'] <= cur + k_val)]
        exams = layer_df['exam'].tolist()
        large_exams = layer_df[layer_df['size'] >= size_cutoff]['exam'].tolist()
        
        hot_starts = None
        if i != 0:
            hot_starts = create_hot_start(sched, exams, tradeoff, assignment, slots_to_use)
        
        other_confs, other_b2bs, other_2i3s = create_dicts(sched, exams, slots_to_use, num_blocks)
        assignment, last_slot_lim = IP(
            exams, large_exams, late_slots, last_slot_lim, tradeoff,
            other_confs, other_b2bs, other_2i3s, fl_pens, hot_starts,
            env=env, slots_to_use=slots_to_use, param_id = param_id
        )
        
        stays_df = layer_df[layer_df['cum_sum'] < cur + k_val - overlap_count]
        for j, row in stays_df.iterrows():
            exam_name = row['exam']
            # Find this exam in the assignment
            assignment_row = assignment[assignment[:, 0] == exam_name][0]
            exam, slot = assignment_row
            sched.loc[len(sched)] = {'exam': exam, 'slot': int(slot)}

        
        i = stays_df.index.max() + 1
        progress = round(exam_sizes.at[i-1, 'cum_sum'] / exam_sizes.at[num_exams-1, 'cum_sum'], 3)
        print(f'Run {param_id} – Progress: {progress} / 1.000')
    
    #save_block_assignment(sched, f'{param_id}', SAVE_PATH + '/schedules/')
    sched.to_csv(SAVE_PATH + '/blocks/' + param_id + '.csv')
    conflicts, b2bs, two_i3s = calculate_metrics(sched, co)
    env.dispose() 
    return {
        'block_assignment_df' : sched,
        'param_id': param_id,
        'size_cutoff': size_cutoff,
        'k': k_val,
        'num_blocks': num_blocks,
        'num_reserved_for_frontloading': num_reserved_for_frontloading,
        'conflicts': conflicts,
        'b2bs': b2bs,
        'two_i3s': two_i3s
    }, sched 