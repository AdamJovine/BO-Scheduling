import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from IPython import display

import seaborn as sns
import copy
import os
import math
import random
from gurobipy import *
import sys, os
import datetime
from metrics.evaluate import evaluate_schedule 
from myproject.data.db_accessor import add_schedule


# 1) Save the original working directory
orig_cwd = os.getcwd()  

project_root = os.path.abspath(os.path.join(orig_cwd, ".."))

# 3) Insert project_root into sys.path so that 'import config.settings' works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4) Now import from config/settings.py
from config.settings import SAVE_PATH, DATA_PATH, NUM_SLOTS, SEMESTER, LICENSES, PP_TIME
from globals.build_global_sets import compute_slot_structures
from block_assignment.helpers import num_conflicts, num_b2bs, num_2i3s, create_dicts, create_hot_start, calculate_metrics, cleanup

from gurobipy import *

co = pd.read_csv(DATA_PATH + '/p_co.csv', index_col='Unnamed: 0')

def setup(ba, schedule_dict):
    """
    Load exam sizes, map exam blocks to slots, replace missing slots with max_slot + 1,
    merge size information, clean up exam group names, and return updated DataFrame and slot info.
    """
    # Read and prepare exam size data
    exam_sizes = pd.read_csv(DATA_PATH + '/exam_sizes.csv')
    exam_sizes = exam_sizes.sort_values('size', ascending=False).reset_index(drop=True)
    size = dict(zip(exam_sizes['exam'], exam_sizes['size']))

    # Copy input DataFrame and map slots
    ba = ba.copy()
    ba['slot'] = ba['Exam Block'].map(schedule_dict)

    # Replace NaN slots with max_slot + 1
    max_slot = int(np.nanmax(ba['slot'].values))
    ba['slot'] = ba['slot'].fillna(max_slot + 1).astype(int)

    # Ensure 'exam' column exists for merging
    if 'exam' not in ba.columns:
        ba['exam'] = ba['Exam Group']

    # Merge size information if not already present
    if 'size' not in ba.columns:
        ba = ba.merge(exam_sizes[['exam', 'size']], how='left', on='exam')

    # Clean up exam group names
    ba['Exam Group'] = ba['Exam Group'].apply(cleanup)

    # Determine unique slots and the number of slots
    slots = sorted(ba['slot'].unique().tolist())
    num_slots = slots[-1]

    return ba, slots, num_slots, size


def create_by_student(ba):
    enrl_df = pd.read_csv(DATA_PATH + '/enrl.csv')
    enrl_df['Exam Key'] = enrl_df['Exam Key'].apply(cleanup) 
    enrl_df = enrl_df.merge(ba, how='left', left_on='Exam Key', right_on='Exam Group')

    # 2) Build two grouped DataFrames:
    #    – One that collects all slots per student
    slots_df = (
        enrl_df
        .groupby('anon-netid')['slot']
        .apply(list)
        .reset_index(name='slots')
    )

    #    – One that collects all exam keys per student
    exam_df2 = (
        enrl_df
        .groupby('anon-netid')['Exam Key']
        .apply(list)
        .reset_index(name='exam')
    )

    # 3) Merge them on “anon-netid” (no suffixes!)
    by_student_block = pd.merge(slots_df, exam_df2, on='anon-netid').set_index('anon-netid')
    return by_student_block




def expand_schedule(schedule, tradeoff, num_slots ):
    """
    Returns a fuller dataframe of schedule with the following new columns:
    - b2bs: The number of back-to-backs that an exam causes
    - confs: The number of conflicts that an exam causes
    - score: A weighted sum of b2bs and confs, where a conflict is 'tradeoff' times worse than a b2b
    
    Before returning, orders the rows in the dataframe in descending order of back-to-backs
    Param schedule: a dataframe with a schedule, consisting of columns for exam and slot
    """
    df = schedule.copy()
    
    b2bs_list = []
    confs_list = []
    twoInThree_list = []
    for i in range(len(df)):
        exam = df['exam'][i]
        slot = df['slot'][i]
        #print('exam', exam)
        #print('slot' , slot)
        #print('schedule ' , schedule )
        b2bs = num_b2bs(schedule, exam, slot)
        confs = num_conflicts(schedule, exam, slot)
        twoInThrees = num_2i3s(schedule, exam, slot)
        b2bs_list.append(b2bs)
        confs_list.append(confs)
        twoInThree_list.append(twoInThrees)
        
    df['b2bs'] = b2bs_list
    df['confs'] = confs_list
    df['2i3s'] = twoInThree_list
    df['score'] = tradeoff[0]*df['confs'].values + df['b2bs'].values + tradeoff[1]*df['2i3s'].values
    df['score_norm'] = df['score'].values / df['size'].values
    return df.sort_values('score_norm', ascending=False, ignore_index = True)


def update_slot_b2bs(df, slot, num_slots):
    """
    Updates the back-to-back numbers in the dataframe for all exams in the given slot
    Param df: dataframe with exam, slot, and b2b columns
    Paral slot: the slot to update exams' b2bs in
    """
    exams = df[df['slot'] == slot]['exam'].tolist() # All exams in the given slot
    for e in exams:
        new_b2b = num_b2bs(df, e, slot)
        row = df.index[df['exam'] == e].tolist()[0]        
        df.loc[row, 'b2bs'] = new_b2b


def update_slot_confs(df, slot):
    """
    Updates the conflict numbers in the dataframe for all exams in the given slot
    Param df: dataframe with exam, slot, and conflict columns
    Paral slot: the slot to update exams' conflicts in
    """
    exams = df[df['slot'] == slot]['exam'].tolist() # All exams in the given slot
    for e in exams:
        new_conf = num_conflicts(df, e, slot)
        row = df.index[df['exam'] == e].tolist()[0]
        df.loc[row, 'confs'] = new_conf
        

def update_slot_2i3s(df, slot, num_slots):
    """
    Updates the 2i3 numbers in the dataframe for all exams in the given slot
    Param df: dataframe with exam, slot, and conflict columns
    Paral slot: the slot to update exams' conflicts in
    """
    exams = df[df['slot'] == slot]['exam'].tolist() # All exams in the given slot
    for e in exams:
        new_2i3 = num_2i3s(df, e, slot)
        row = df.index[df['exam'] == e].tolist()[0]
        df.loc[row, '2i3s'] = new_2i3
    
    
def reassign(df, assignment, num_slots, tradeoff):
    """
    reassigns exams in df according to the given assignment and updates b2bs, confs, and score accordingly.
    """
    slots_toUpdate_confs = set()
    slots_toUpdate_b2bs = set()
    slots_toUpdate_2i3s = set()
    #print('df , ' , df )
    for swap in assignment:
        exam = swap[0]
        slot = int(float(swap[1]))
        row = df.index[df['exam'] == exam].tolist()[0]
        old_slot = df.loc[row,'slot']
        df.loc[row, 'slot'] = slot
        
        slots_toUpdate_confs.add(old_slot)
        slots_toUpdate_confs.add(slot)
        
        for s in [old_slot, slot]:
            if s == 1:
                slots_toUpdate_b2bs.add(2)
                slots_toUpdate_2i3s.add(3)
            elif s == num_slots:
                slots_toUpdate_b2bs.add(num_slots-1)
                slots_toUpdate_2i3s.add(num_slots-2)
            else:
                slots_toUpdate_b2bs.add(s-1)
                slots_toUpdate_b2bs.add(s+1)
                slots_toUpdate_2i3s.add(s-2)
                slots_toUpdate_2i3s.add(s+2)        
            
    for swap in assignment:
        exam = swap[0]
        slot = int(float(swap[1]))
        row = df.index[df['exam'] == exam].tolist()[0]
        df.loc[row, 'confs'] = num_conflicts(df, exam, slot)
        df.loc[row, 'b2bs'] = num_b2bs(df, exam, slot)
        df.loc[row, '2i3s'] = num_2i3s(df, exam, slot)
    
    for slot in slots_toUpdate_confs:
        update_slot_confs(df, slot)
    for slot in slots_toUpdate_b2bs:
        update_slot_b2bs(df, slot, num_slots)
    for slot in slots_toUpdate_2i3s:
        update_slot_2i3s(df, slot, num_slots)
        
    df['score'] = tradeoff[0]*df['confs'].values + df['b2bs'].values + tradeoff[1]*df['2i3s'].values
    df['score_norm'] = df['score'].values / df['size'].values
    df.sort_values('score_norm', ascending=False, ignore_index = True, inplace = True)
    #print('df after, ' , df )
    
def get_b2bs(df):
    """
    Returns total number of back-to-backs in a schedule
    Param df: dataframe of to check back-to-backs in
    """
    return df['b2bs'].sum()/2


def get_confs(df):
    """
    Returns total number of conflicts in a schedule
    Param df: dataframe of to check conflicts in
    """
    return df['confs'].sum()/2

def get_2i3s(df):
    """
    Returns total number of conflicts in a schedule
    Param df: dataframe of to check conflicts in
    """
    return df['2i3s'].sum()/2


def get_score(df, tradeoff):
    """
    Returns total badness score in a schedule
    Param df: dataframe of to check badness score in
    """
    return get_b2bs(df) + tradeoff[0]*get_confs(df) + tradeoff[1]*get_2i3s(df)


def IP(df, large_exams, late_slots, last_slot_lim, exams, tradeoff, other_confs, other_b2bs, other_2i3s, env , slots, size):
    
    # The model and its parameters
    #m = Model()
    #if hosted: 
    slot_starts = compute_slot_structures(slots)
    start_slots_2i3 = slot_starts['two_in_three_start']
    start_slots_b2b =  slot_starts['eve_morn_start'] + slot_starts['other_b2b_start']
    #print(" slot_starts['eve_morn_start']" ,  slot_starts['eve_morn_start'] )
    #print("slot_starts['other_b2b_start'] " , slot_starts['other_b2b_start'] )
    m = Model(env = env)
    m.Params.Timelimit = 60
    
    original_assignment = df[df['exam'].isin(exams)][['exam','slot']].values   # The assignment (exams and slots) before the IP is run
    #print('exams ; ', exams )
    #print('slots ; ', slots )
    exams = set(exams)
    
    x = m.addVars(product(exams,slots), vtype=GRB.BINARY)   # Xis indicates that exam i is in slot s
    y = m.addVars(product(exams,exams))   # Yij indicates that exam i and exam j are in the same slot
    z = m.addVars(product(exams,exams))   # Zij indicates that exam i and exam j are in consecutive slots
    w = m.addVars(product(exams,exams))   # Wij indicates that exam i and exam j are two slots apart                                    
    v = m.addVars(slots)   # Vij indicates the amount of students taking an exam in slot s.

    
    m.addConstrs(y[i,j] >= x[i,s] + x[j,s] - 1 for s in slots for i in exams for j in exams)   # Defining the y variables
    m.addConstrs(y[i,j] >= 0 for i in exams for j in exams)
    
    m.addConstrs(z[i,j] >= x[i,s] + x[j,s+1] - 1 for s in start_slots_b2b for i in exams for j in exams)   # Defining the z variables
    m.addConstrs(z[i,j] >= 0 for i in exams for j in exams)
    
    m.addConstrs(w[i,j] >= x[i,s] + x[j,s+2] - 1 for s in start_slots_2i3 for i in exams for j in exams)   # Defining the w variables
    m.addConstrs(w[i,j] >= 0 for i in exams for j in exams)
    
    m.addConstrs(quicksum(x[i,s]*size[i] for i in exams) == v[s] for s in slots)   # Defining the v variables
    
    m.addConstrs(quicksum(x[i,s] for s in slots) == 1 for i in exams)   # One slot per exam
    
    # Restricting total students in the last time slot
    m.addConstrs(v[i]<=last_slot_lim for i in late_slots)

    
    #large_exams = set(large_exams).intersection(set(exams))
    #print(large_exams)
    #Keep large exams away from late time slots
    #m.addConstrs(quicksum(x[i, s] for i in large_exams) == 0 for s in late_slots)
    
    # Fix large exams:
    for ind in product(large_exams,slots):
        i = ind[0]
        s = ind[1]
        if df[df['exam']==i]['slot'].tolist()[0] == s:
            m.addConstr(x[ind] == 1)
    
    number_confs = quicksum(0.5*y[i,j]*co[i][j] for i in exams for j in exams) + quicksum(x[i,s]*other_confs[(i,s)] for i in exams for s in slots)
    number_b2bs = quicksum(z[i,j]*co[i][j] for i in exams for j in exams) + quicksum(x[i,s]*other_b2bs[(i,s)] for i in exams for s in slots)
    number_2i3s = quicksum(w[i,j]*co[i][j] for i in exams for j in exams) + quicksum(x[i,s]*other_2i3s[(i,s)] for i in exams for s in slots)
    
    # Objective
    m.setObjective(tradeoff[0]*number_confs + number_b2bs + tradeoff[1]*number_2i3s, GRB.MINIMIZE)
     
    # Hot start
    for ind in product(exams,slots):
        i = ind[0]
        s = ind[1]
        if df[df['exam']==i]['slot'].tolist()[0] == s:
            x[ind].Start = 1
        else:
            x[ind].Start = 0
    for ind in product(exams,exams):
        i = ind[0]
        j = ind[1]
        slot_i = df[df['exam']==i]['slot'].tolist()[0]
        slot_j = df[df['exam']==j]['slot'].tolist()[0]
        
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
    
    
    m.update()
    m.optimize()
    
    if m.status == GRB.OPTIMAL:   
        new_assignment = []  # The new assignment, including the eams that didn't change slots
        for key in x.keys():
            if x[key].x == 1:
                new_assignment.append(list(key))
        new_assignment = np.array(new_assignment)

        #assignment = new_assignment[(original_assignment[:,1].astype(float).astype(int) - new_assignment[:,1].astype(float).astype(int)) != 0]   # The assignment with only the exams whose slot actually changed
        orig_df = pd.DataFrame(
            original_assignment, 
            columns=['exam','slot_old']
        ).astype({'slot_old': int})

        new_df = pd.DataFrame(
            new_assignment, 
            columns=['exam','slot_new']
        ).astype({'slot_new': int})

        # merge on exam so that each row has both old and new slot
        merged = orig_df.merge(new_df, on='exam', how='inner')

        # pick only the exams whose slot actually changed
        changed = merged[merged['slot_old'] != merged['slot_new']]

        # now assignment is a 2-column array [exam, slot_new] for only the changed exams
        assignment = changed[['exam','slot_new']].values
        return assignment, last_slot_lim, m.status
    else: 
        print('EMPTY ASSIGN')
        
        return [] , last_slot_lim, m.status
   




def run_pp(license, ba, schedule_dict, chunk, size_cutoff, big_cutoff, pp_params, sched_name):
    """
    ba is the block assignment df 
    schedule is the block to slot map df 
    chunk is how many exams are rescheduled at once
    size_cutoff is the exam-size threshold for "large" exams which we don't move 
    big_cutoff is the slot after which no large blocks (blocks with more than 3000 exams) can be scheduled
    """
    print('RUNNING PP ')
    #print('ARGUEMNTS : ' , license, ba, schedule_dict, chunk, size_cutoff, big_cutoff, pp_params,sched_name)
    env = Env(params=license)
    ba, slots, num_slots, size = setup(ba, schedule_dict)
    #ba['exam'] = ba['Exam Group'].apply(cleanup)
    #exam_sizes = pd.read_csv(DATA_PATH + '/exam_sizes.csv')
    #ba = ba.merge(exam_sizes, how='inner', left_on='exam', right_on='exam')
    last_slot_lim = 3000
    twointhree_weight , fl_pens= pp_params
    tradeoff = (1000, twointhree_weight)  # (<how much worse is a conflict compared to a b2b>, <how much worse is a 2i3 compared to a b2b>)
    k = chunk  # The number of exams to reorder at a time.
    forward_step = 0.4 * k
    backward_step = 0.2 * k
    late_slots = list(np.array(slots)[np.array(slots) >= big_cutoff])

    sched = expand_schedule(ba, tradeoff, num_slots)
    # initialize plot data
    times = [0]
    scores = [get_score(sched, tradeoff)]

    #sched.to_csv(SAVE_PATH + '/schedules/INITIAL' +sched_name + '.csv', header=True, index=False)
    
    start_time = datetime.datetime.now()
    #slot_structures = compute_slot_structures(list(range(1, NUM_SLOTS + 1)))
    #global_sets = {**slot_structures}
    #evaluate_schedule(sched, exam_sizes, [], global_sets, slots_per_day=3)
    
    now = datetime.datetime.now()

    while True:
        last_score = get_score(sched, tradeoff)
        i = 0
        while True:
            current_time = datetime.datetime.now()
            elapsed_time = (current_time - start_time).total_seconds()
            if elapsed_time > PP_TIME:
                break

            i = int(i)
            print('i : ' , i )
            best_score = get_score(sched, tradeoff)
            cur_sched = copy.deepcopy(sched)
            exams_to_swap = sched.iloc[i : min(i + k, len(sched))]
            exams = list(exams_to_swap['exam'])
            large_exams = exams_to_swap[exams_to_swap['size'] >= size_cutoff]['exam'].tolist()
            #print('sched ' , sched )
            other_confs, other_b2bs, other_2i3s = create_dicts(
                sched, exams, slots, num_slots
            )
            #print('sched after ' , sched )
            assignment, last_slot_lim, status = IP(
                sched,
                large_exams,
                late_slots,
                last_slot_lim,
                exams,
                tradeoff,
                other_confs,
                other_b2bs,
                other_2i3s,
                env,
                slots,
                size,
            )
            reassign(sched, assignment, num_slots, tradeoff)
            new_score = get_score(sched, tradeoff)
            #evaluate_schedule(sched, exam_sizes, [], global_sets, slots_per_day=3)
            if status == GRB.INFEASIBLE:
                break

            if best_score == new_score:
                i += forward_step
                if i >= len(sched):
                    break
            elif best_score < new_score:
                sched = copy.deepcopy(cur_sched)
                i += forward_step * 2
                if i >= len(sched):
                    break
            else:
                i = max(0, i - backward_step)
                times.append((datetime.datetime.now() - now).seconds)
                scores.append(new_score)
                #update_plot()
                if len(times) > 4:
                    speed_ind = len(times) - 4
                    speed = 60 * (scores[-1] - scores[speed_ind]) / (
                        times[-1] - times[speed_ind]
                    )
                # continue looping
            print('best_score : ' , best_score )
        current_time = datetime.datetime.now()
        elapsed_time = (current_time - start_time).total_seconds()
        new_score = get_score(sched, tradeoff)

        if new_score >= last_score or status == GRB.INFEASIBLE or elapsed_time > PP_TIME:
            break
    
    sched.to_csv(SAVE_PATH + '/schedules/' +sched_name + '.csv', header=True, index=False)
    add_schedule(sched_name, sched)
    env.dispose() 
    return sched






