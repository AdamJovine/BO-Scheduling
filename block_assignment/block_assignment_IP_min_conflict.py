
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import os
import math
from copy import copy
import random
pd.set_option('display.max_columns', None)
from itertools import takewhile
import gurobipy as gp
from gurobipy import *
from itertools import product
import sys
from config.settings import BA_TIME
orig_cwd = os.getcwd()  

project_root = os.path.abspath(os.path.join(orig_cwd, ".."))

# 3) Insert project_root into sys.path so that 'import config.settings' works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4) Now import from config/settings.py
from config.settings import SAVE_PATH, DATA_PATH, NUM_SLOTS, SEMESTER, LICENSES
from globals.build_global_sets import compute_slot_structures
from block_assignment.helpers import num_conflicts, num_b2bs, num_2i3s, create_dicts, create_hot_start, calculate_metrics


def group_partition_ip(k,n,  fixed_dict, groups, large_groups, ccm, group_pairs, env,read=True, readfile=None, param_id = 'block'):
    '''
    Input Parameters:
        k                   := number of desired groups
        fixed_dict          := dictionary of the form {course:group} containing pre-fixed course/group pairings
        courses             := list of all courses to be considered
        ccm                 := course coenrollment matrix
        read                := read in a starting solution or not
        readfile            := name of file to be read in
    '''
    m = Model(env = env)
    m.Params.SolFiles = "/home/asj53/BOScheduling/block_assignment/Incumbents/incumbent"   # record
    blocks = list(np.arange(1,k+1))
    inbetween = blocks[-1:1]
    
    x = m.addVars(list(product(groups, blocks)), vtype=GRB.BINARY, name='x')
    y = m.addVars(group_pairs, vtype=GRB.BINARY, name='y')
    z = m.addVars(group_pairs, vtype=GRB.BINARY, name='z') # if 2 courses r next to eachother

    group_assignments = m.addVars(groups, vtype=GRB.INTEGER, name='group_assignments')

    m.addConstrs((gp.quicksum(x[(c,i)] for i in blocks) == 1 for c in groups),
                 name='color_every_exam_group')
    m.addConstrs((x[(d[0],i)] + x[(d[1],i)] <= 1+ y[d] for d in group_pairs for i in blocks), 
                 name='conflict_enforcement')
    m.addConstrs((group_assignments[c] == gp.quicksum(x[(c,i)]*i for i in blocks) for c in groups), 
                 name='write_output')
    m.addConstrs((x[(group, block)] == 1 for group, block in fixed_dict.items()),
                name='fixed_block_constraints')
    m.addConstrs((x[(c,i)] == 0 for c in large_groups for i in range(1,n + 1)), 
                 name = 'no_large_exam_blocks')
    # Break symmetry
    m.addConstrs(((gp.quicksum(x[(c,i)] for c in groups) >= gp.quicksum(x[c,i+1] for c in groups)) for i in blocks[:-1]),
                 name = 'symmetry_break')
    m.addConstrs((x[(d[0],i)] + x[(d[1],i +1)] <= 1 + z[d[0],d[1]] for d in group_pairs for i in inbetween)  ,name = 'if blocks b to b')

    m.update()
    if read and readfile is not None:
        warm_start_df = pd.read_csv(readfile)
        for idx, row in warm_start_df.iterrows():
            exam = row['Exam Group']
            assigned_block = int(row['Exam Block'])
            for i in blocks:
                if (exam, i) in x:
                    x[(exam, i)].start = 1 if i == assigned_block else 0
            if exam in group_assignments:
                group_assignments[exam].start = assigned_block


    m.setObjective(gp.quicksum(1000 *y[d]*ccm.at[d[0], d[1]] + z[d]*ccm.at[d[0], d[1]] for d in group_pairs) , GRB.MINIMIZE)
    print('timelimit: ', BA_TIME)
    m.setParam('Timelimit',BA_TIME)  # SET TIME LIMIT HERE (seconds)
    m.write(os.path.join(SAVE_PATH, f'{param_id}.lp'))
    m.write(os.path.join(SAVE_PATH, f'{param_id}.mps'))
    m.optimize()
    
    obj = m.getObjective().getValue()
    df_ga = pd.DataFrame(columns = ['Exam Group', 'Exam Block'])

    rows_list = []
    for k in group_assignments.keys():
        rows_list.append({'Exam Group': k, 'Exam Block': group_assignments[k].X})

    df_ga = pd.concat([df_ga, pd.DataFrame(rows_list)], ignore_index=True)

        
    return obj, df_ga, m.ObjBound

def run_min_IP(param_id , size_cutoff, num_reserved,k_val, num_blocks, license):
  group_coenroll_matrix = pd.read_csv(DATA_PATH + '/p_co.csv', index_col = 0)
  exam_groups = list(group_coenroll_matrix.columns)
  exam_sizes =  pd.read_csv(DATA_PATH + '/exam_sizes.csv', index_col = 0)
  group_pairs = []
  for i in exam_groups:
      for j in exam_groups:
          if group_coenroll_matrix.at[i,j] > 0:
              group_pairs.append((i,j))
              
  env = Env(params=license)

  large = exam_sizes[exam_sizes['size']>size_cutoff].index

  obj, assign_df, bound = group_partition_ip(num_blocks,num_reserved,  {}, groups=exam_groups,large_groups = large,  ccm=group_coenroll_matrix, group_pairs = group_pairs, env = env,  read=False , readfile =  '22cakesize300res6.csv', param_id = param_id)

  assign_df['Exam Block'] = assign_df['Exam Block'].astype(int)
  print('BLOCK ASSIGN : ' , "obj : " , obj, ' assign_df : ' , assign_df)
  assign_df.to_csv(SAVE_PATH + '/blocks/' + param_id + '.csv', header=True, index=False)
  return obj, assign_df 