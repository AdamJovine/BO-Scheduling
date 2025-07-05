# -*- coding: utf-8 -*-
# Adam's personal Access token : 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

# GLOBAL VARIABLES

#WLSACCESSID = '7820242a-1059-4e41-be5e-249bf3b03c9f' # JK
#WLSSECRET = '1b840509-d813-4b67-a764-fbb5b30fa693'
#LICENSEID = 2471364

#WLSACCESSID='242e9db4-8e62-4689-9cf8-86bbed11c64b' # Claire
#WLSSECRET='170fbb2e-5b78-4820-a401-0016773c518f'
#LICENSEID=2409395

#WLSACCESSID = '0fb92c9f-9175-4f1c-a107-ab835fc599b7' # Josh
#WLSSECRET = '680f2cd4-fbe5-432b-86b5-e8b14e8c73ef'
#LICENSEID = 2554057

#WLSACCESSID= '6c4ca0fe-20d3-44ff-853c-28164e486aeb' # Wendy
#WLSSECRET= 'cf7eea0d-abbb-4f83-b09c-5f8d7922d6c3'
#LICENSEID= 933109

#WLSACCESSID='5f815caa-7be8-4350-9b6e-0f60f81b3f34' # Jacob
#WLSSECRET='ff974abc-ac9e-4b0e-ba7f-359a2eb8c9fa'
#LICENSEID=2472650


#WLSACCESSID = 'ba87edc6-09f8-498b-b499-3e217b7e7885' # Sharafa
#WLSSECRET = '11099141-cd9f-4d77-9d50-d5725a3b5c4a'
#LICENSEID = 2468274

#WLSACCESSID='94f291a0-38b1-475c-b39f-c50e2ef84f39' # Selina
#WLSSECRET='3365e397-bda9-4721-ba58-acd6ea9dfc23'
#LICENSEID=2409153

#WLSACCESSID =  'bd120bc9-6607-4340-8ee1-db936137c742' #Hana
#WLSSECRET = '2da68506-6922-4c6a-bf78-c84246f711ff'
#LICENSEID=  2468271

#WLSACCESSID = '8c2c78f4-617e-4762-a9b8-9a9ad8a076f2' #Hedy
#WLSSECRET= 'cf7a4d3d-44b3-425e-b1bb-e6e7383ef665'
#LICENSEID= 2554067

#WLSACCESSID='89f148ce-5380-4e45-9743-db24dc5b24a1' # Reevu
#WLSSECRET='bdd33d9f-60f1-43b0-8c66-b64b9130bf8b'
#LICENSEID=2557789

WLSACCESSID='2954ef62-8fa8-4f22-a875-c2ebc1136e22'
WLSSECRET='08ab134c-9a73-4212-a3c0-d625c9f8a662'
LICENSEID=931025

sem = 'sp25'
co_name = 'p_co'
num_blocks = 22 # How many blocks in blocks assignment
size_cutoff = 300
reserved = 6# how many blocks don't have any large courses
assignment_type = 'cake' # model to create block assignment
key_name = 'Exam Key'
slots = {22 : '[1,2,4,5,8,9,10,11,12,13,14,15,16,17,18,18,20,21,22,23,24,25]', 23: '[1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]' , 24: '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]', 25 : '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]' , 26 : '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'}

import pandas as pd
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict

num_slots = 24
first_list = [0 for i in range(24)]
for j in range(num_slots-3, num_slots):
  first_list[j] = 3
for k in range(num_slots - 6, num_slots - 3):
  first_list[k] = 2
for k in range(num_slots - 9, num_slots - 6):
  first_list[k] = 1
license = {
    "WLSACCESSID": '0fb92c9f-9175-4f1c-a107-ab835fc599b7' ,
    "WLSSECRET":  '680f2cd4-fbe5-432b-86b5-e8b14e8c73ef',
    "LICENSEID": 2554057 }

env =gp.Env(params=license)
# Using a lambda (with defaults if key not found)

# Indexing for decision variables
os.environ["TORCH_COMPILE_DISABLE"] = "1"

# ==========================
# Preprocessing and Data Load
# ==========================
total_slots = 24
semester = 'sp25'

# Load and clean block assignment /home/asj53/final-scheduling/results/sp25/blocks/22cakesize300res6.csv 
ba = pd.read_csv('/home/asj53/final-scheduling/results/sp25/blocks/22cakesize300res6.csv')
ba['Exam Group'] = ba['Exam Group'].astype(str)

block_names = np.sort(ba['Exam Block'].unique())
adjusted_block_dict = dict(zip(block_names, np.arange(1, len(block_names) + 1)))
ba['adj_block'] = ba['Exam Block'].map(adjusted_block_dict)

# Normalize Exam Group IDs
ba['Exam Group'] = ba['Exam Group'].apply(
    lambda exam_group: str(int(float(exam_group))) if str(exam_group).replace('.', '', 1).isdigit() else str(exam_group)
)
print(ba )

# Load and clean exam_sizes
exam_sizes = pd.read_csv(f"/home/asj53/final-scheduling/data/sp25/exam_sizes.csv")
exam_sizes = exam_sizes.sort_values('size', ascending=False).reset_index(drop=True)
exam_sizes['exam'] = exam_sizes['exam'].apply(
    lambda exam_id: str(int(float(exam_id))) if str(exam_id).replace('.', '', 1).isdigit() else str(exam_id)
)
size = dict(zip(exam_sizes['exam'], exam_sizes['size']))

# Load exam_df and join with block assignments
exam_df = pd.read_csv('/home/asj53/final-scheduling/data/sp25/exam_df.csv', low_memory=False)
exam_df_with_blocks = exam_df.merge(ba, how='left', left_on='Exam Key', right_on='Exam Group')

# ==========================
# Build Co-enrollment Dictionaries
# ==========================


by_student_block = exam_df_with_blocks.groupby('anon-netid')['adj_block'].apply(list).reset_index(name='blocks')


pairwise_coenrollment_counts = {}  # pairwise co-enrollment: number of students in both blocks
triple_coenrollment_counts = {}   # triple co-enrollment: number of students in three blocks
quadruple_coenrollment_counts = {}# quadruple co-enrollment: number of students in four blocks
slots = np.arange(1, num_slots+1)
slots = [int(s) for s in slots]

for i in slots:
    for j in slots:
        for k in slots:
            for l in slots:
                quadruple_coenrollment_counts[(i, j, k, l)] = 0
            triple_coenrollment_counts[(i, j, k)] = 0
        pairwise_coenrollment_counts[(i, j)] = 0

# Fill coenrollment dictionaries
for block_list in by_student_block['blocks']:
    for i in block_list:
        for j in block_list:
            for k in block_list:
                for l in block_list:
                    if ((i != j) and (j != k) and (i != k) and (i != l) and (j != l) and (k != l)):
                        quadruple_coenrollment_counts[(i, j, k, l)] += 1
                if ((i != j) and (j != k) and (i != k)):
                    triple_coenrollment_counts[(i, j, k)] += 1
            if (i != j):
                pairwise_coenrollment_counts[(i, j)] += 1


slots_per_day = 3
# Translation to different slot notation
slots_n = range(1,len(slots)+1)
d = dict(zip(slots, slots_n))
print(d)
slots_e = list(slots) + [np.inf]*10

triple_24_start = []
triple_day_start = []
eve_morn_start = []
other_b2b_start = []
print('slots' , slots)
for j in range(len(slots) ):
    s = slots[j]
    if s+1 == slots_e[j+1]: # 11
        if s%slots_per_day == 0:
            eve_morn_start.append(d[s])
        else:
            other_b2b_start.append(d[s])
        if s+2 == slots_e[j+2]: # 111
            if slots_per_day - s%slots_per_day >= 2 and slots_per_day - s%slots_per_day != slots_per_day:
                triple_day_start.append(d[s])
            else:
                triple_24_start.append(d[s])

block_pair = [(i, j) for i in slots for j in slots]
print('triple_24_start' , triple_24_start )
print('triple_day_start' , triple_day_start )

print('eve_morn_start' , eve_morn_start )
print('other_b2b_start' , other_b2b_start )
shifted_slots = [int(i) for i in np.roll(slots, -1)]
next_slot = dict(zip(slots, shifted_slots))
print('next_slot' , next_slot)

triple_in_day = triple_day_start
triple_in_24hr = triple_24_start
tripi_tropi = np.sort(triple_in_day+triple_in_24hr)

block_slot = [(i, s) for i in slots for s in slots]
triple_slots = [ i for i in range(1, num_slots -1)]
block_sequence_trip = [(i, j, k) for i in slots for j in slots for k in slots]
block_sequence_quad = [(i, j, k, l) for i in slots for j in slots for k in slots for l in slots]
block_sequence_slot = [(i, j, k, s) for i in slots for j in slots for k in slots for s in slots]
print('triple_slots' , )
student_unique_block = {}
student_unique_block_pairs = {}
for index, row in by_student_block.iterrows():
    if len(row["blocks"]) == 1:
        if row["blocks"][0] in student_unique_block:
            student_unique_block[row["blocks"][0]] += 1
        else:
            student_unique_block[row["blocks"][0]] = 1
    if len(row["blocks"]) == 2:
        a, b = row["blocks"][0], row["blocks"][1]
        if (a, b) in student_unique_block_pairs:
            student_unique_block_pairs[(a, b)] += 1
        else:
            student_unique_block_pairs[(a, b)] = 1
        if (b, a) in student_unique_block_pairs:
            student_unique_block_pairs[(b, a)] += 1
        else:
            student_unique_block_pairs[(b, a)] = 1
for item in range(1, total_slots + 1):
    if item not in student_unique_block:
        student_unique_block[item] = 0
    for item2 in range(1, total_slots + 1):
        if (item, item2) not in student_unique_block_pairs:
            student_unique_block_pairs[(item, item2)] = 0
print (student_unique_block)

exam_sizes = exam_sizes.sort_values('size', ascending=False).reset_index().drop('index', axis=1)
# Import coenrollment matrix
co = pd.read_csv('/home/asj53/final-scheduling/data/'+semester+'/p_co.csv', index_col='Unnamed: 0')

################ Calculate starting slots ################

quint_start = []
quad_start = []
four_in_five_start = []  # not including quints or quads  
triple_24_start = []
triple_day_start = []
three_in_four_start = []  # not including triples
eve_morn_start = []
other_b2b_start = []
two_in_three_start = []

slots_e = slots + [np.inf]*10

for j in range(len(slots)):
    s = slots[j]
    if s+1 == slots_e[j+1]: # 11  
        if s%slots_per_day == 0:
            eve_morn_start.append(d[s])
        else:
            other_b2b_start.append(d[s])       
        if s+2 == slots_e[j+2]: # 111
            if slots_per_day - s%slots_per_day >= 2 and slots_per_day - s%slots_per_day != slots_per_day:
                triple_day_start.append(d[s])
            else:
                triple_24_start.append(d[s])             
            if s+3 == slots_e[j+3]: # 1111
                quad_start.append(d[s])     
                if s+4 == slots_e[j+4]: # 11111
                    quint_start.append(d[s])         
            else: #1110
                three_in_four_start.append(d[s])
                if s+4 == slots_e[j+3]: # 11101
                    four_in_five_start.append(d[s]) 
        else: # 110
            if s+3 == slots_e[j+2]: # 1101
                three_in_four_start.append(d[s])        
                if s+4 == slots_e[j+3]: # 11011
                    four_in_five_start.append(d[s])                
    else: # 10 
        if s+2 == slots_e[j+1]: # 101
            two_in_three_start.append(d[s])     
            if s+3 == slots_e[j+2]: #1011
                three_in_four_start.append(d[s])     
                if s+4 == slots_e[j+3]: #10111
                    four_in_five_start.append(d[s])
                    
# Adjust two_in_three_start
n = []
for j in two_in_three_start:
    n.append([j,j+1])
for j in triple_24_start + triple_day_start:
    n.append([j,j+2])
two_in_three_start = np.sort(np.array(n), axis=0)


################ Set up schedule ################
def get_metrics(sched, out ):
  pd.DataFrame(data={
    'sem': 'sp25',
    'image_name': out + 'img',
    'file_name': out + 'metrics',
    'schedule': out,
    'num_slots': 24,
    'key': 'Exam Key'
  }, index=[1]).to_csv('metrics.csv')

  lic_param = pd.read_csv('metrics.csv')
  semester = lic_param['sem'].values[0]
  num_slots = lic_param['num_slots'].values[0]
  block_assignment = sched.copy()
  block_assignment['Exam Group'] = block_assignment['exam']
  block_assignment['Exam Slot'] = block_assignment['slot']
  slots1 = list(np.sort(np.unique(sched['slot'].values)))
  slots2 = range(1,len(slots1)+1)
  d = dict(zip(slots1, slots2))
  block_assignment['Exam Slot'] = block_assignment['Exam Slot'].map(d)


  ################ Create by_student df ################             A df which shows for each student a list of slots they have exams in

  # Join student enrollment with block assignments
  exam_df_with_blocks = exam_df.merge(block_assignment, how='left', left_on='Exam Key', right_on='Exam Group')

  blocks = np.sort(block_assignment['Exam Slot'].unique())
  by_student_block = exam_df_with_blocks.groupby('anon-netid')['Exam Slot'].apply(list).reset_index(name='blocks')

  max_values = by_student_block['blocks'].apply(lambda x: max(x) if x else np.nan)

  # Compute the average of these maximum values
  average_max = max_values.mean()

  print("The average maximum is:", average_max)
  sched['lateness'] = sched['slot'] - 15
  sched['weighted'] = sched['lateness'] * sched['size'].apply(lambda x : 0 if x<100 else x)
  sched['weighted'] = sched['weighted'].apply(lambda x : 0 if x < 0 else x ) 
  lateness = sched['weighted'].sum()
  print('lateness' , lateness ) 
  conflicts = 0


  for s in range(len(by_student_block)):
      #print(by_student_block['blocks'][s])
      for b in range(len(by_student_block['blocks'][s])):
          if(by_student_block['blocks'][s].count(by_student_block['blocks'][s][b]) > 1):
              conflicts += by_student_block['blocks'][s].count(by_student_block['blocks'][s][b]) - 1
              by_student_block['blocks'][s][b] = -1
  print("conflicts:", conflicts)

  quint = [[i, i+1, i+2, i+3, i+4] for i in quint_start]
  quint_count = 0
  for s in range(len(by_student_block)):
      for q in quint:
          if(all(b in by_student_block['blocks'][s] for b in q)):
              for b in q:
                  by_student_block['blocks'][s].remove(b)
              quint_count += 1
  print("quints:", quint_count)

  quad = [[i, i+1, i+2, i+3] for i in quad_start]
  quad_count = 0
  for s in range(len(by_student_block)):
      for q in quad:
          if(all(b in by_student_block['blocks'][s] for b in q)):
              for b in q:
                  by_student_block['blocks'][s].remove(b)
              quad_count += 1
  print("quads:", quad_count)



  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  triple_24 = [[i, i+1, i+2] for i in triple_24_start]
  triple_24_count = 0
  has_triple = set()
  for s in range(len(by_student_block)):
      for t in triple_24:
          if(all(b in by_student_block['blocks'][s] for b in t)):
              for b in t:
                  by_student_block['blocks'][s].remove(b)
              triple_24_count += 1
              #print(b)
              has_triple.add(b)
  print("triple in 24h (no gaps):", triple_24_count)

  triple_day = [[i, i+1, i+2] for i in triple_day_start]
  triple_day_count = 0
  for s in range(len(by_student_block)):
      for t in triple_day:
          if(all(b in by_student_block['blocks'][s] for b in t)):
              for b in t:
                  by_student_block['blocks'][s].remove(b)
              triple_day_count += 1
              #print(b)
              has_triple.add(b)
  print("triple in same day (no gaps):", triple_day_count)
  #print('has Trip: ', has_triple)

  four_in_five_count = 0
  for s in range(len(by_student_block)):
      for t in quint:
          if sum([b in by_student_block['blocks'][s] for b in t]) >= 4:
              for b in t:
                  if b in by_student_block['blocks'][s]:
                      by_student_block['blocks'][s].remove(b)
              four_in_five_count += 1
  four_in_five = [[i, i+1, i+2, i+3] for i in four_in_five_start]
  for s in range(len(by_student_block)):
      for t in four_in_five:
          if(all(b in by_student_block['blocks'][s] for b in t)):
              for b in t:
                  by_student_block['blocks'][s].remove(b)
              four_in_five_count += 1
  print("four in five slots:", four_in_five_count)



  three_in_four_count = 0
  for s in range(len(by_student_block)):
      for t in quad:
          if sum([b in by_student_block['blocks'][s] for b in t]) >= 3:
              for b in t:
                  if b in by_student_block['blocks'][s]:
                      by_student_block['blocks'][s].remove(b)
              three_in_four_count += 1
  three_in_four = [[i, i+1, i+2] for i in three_in_four_start]
  for s in range(len(by_student_block)):
      for t in three_in_four:
          if(all(b in by_student_block['blocks'][s] for b in t)):
              for b in t:
                  by_student_block['blocks'][s].remove(b)
              three_in_four_count += 1
  print("three in four slots:", three_in_four_count)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  eve_morn = [[i, i+1] for i in eve_morn_start]
  eve_morn_count = 0
  for s in range(len(by_student_block)):
      for p in eve_morn:
          if(all(b in by_student_block['blocks'][s] for b in p)):
              for b in p:
                  by_student_block['blocks'][s].remove(b)
              eve_morn_count += 1
  print("evening/morning b2b:", eve_morn_count)

  other_b2b = [[i, i+1] for i in other_b2b_start]
  other_b2b_count = 0
  for s in range(len(by_student_block)):
      for p in other_b2b:
          if(all(b in by_student_block['blocks'][s] for b in p)):
              for b in p:
                  by_student_block['blocks'][s].remove(b)
              other_b2b_count += 1
  print("other b2b:", other_b2b_count)

  two_in_three_count = 0
  for s in range(len(by_student_block)):
      for t in two_in_three_start:
          if(all(b in by_student_block['blocks'][s] for b in t)):
              for b in t[:-1]:
                  by_student_block['blocks'][s].remove(b)
              two_in_three_count += 1
  print("two in three slots:", two_in_three_count)

  # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  late_exam_count = 0
  for s in range(len(by_student_block)):
      if len(by_student_block['blocks'][s]) == 1:
          if by_student_block['blocks'][s][0] > 18:
              late_exam_count += 1
  print ("singular late exam count: ", late_exam_count)

  two_exams_large_gap = 0
  for s in range(len(by_student_block)):
      if len(by_student_block['blocks'][s]) == 2:
          mini = min(by_student_block['blocks'][s])
          maxi = max(by_student_block['blocks'][s])
          if maxi - mini > 15:
              two_exams_large_gap += 1
  print ("two exams, large gap: ", two_exams_large_gap)
  metrics = pd.DataFrame(data = {'conflicts': conflicts, 
    'quints': quint_count, 
    'quads': quad_count, 
    'four in five slots': four_in_five_count,
    'triple in 24h (no gaps)': triple_24_count,
    'triple in same day (no gaps)': triple_day_count,
    'three in four slots': three_in_four_count,
    'evening/morning b2b': eve_morn_count,
    'other b2b': other_b2b_count,
    'two in three slots': two_in_three_count,
    'singular late exam': late_exam_count,
    'two exams, large gap': two_exams_large_gap,
    'avg_max' : average_max , 
    'lateness' : lateness }, index = [lic_param['image_name'].values[0]],
    
    
    ) 

  metrics.to_csv('/home/asj53/final-scheduling/results/'+semester+'/metrics/' + lic_param['file_name'].values[0] + '.csv', header=True, index=False)






def schedule_ip(alpha=16, beta=16, gamma1=1, gamma2=1, delta=0, vega=3, theta=0,
                  triple_coenrollment_counts=None, pairwise_coenrollment_counts=None,
                  lambda_large1=200, lambda_large2=100, lambda_big=200,
                  large_blocks_1=None, large_blocks_2=None, big_blocks=None,
                  early_slots_1=None, early_slots_2=None, slots= None, read=False,  readfile='warm_start.sol', writefile="warm_start.sol") :

    slots = np.arange(1, num_slots+1)
    slots = [int(s) for s in slots]
    print('triple_24_start' , triple_24_start )
    print('triple_day_start' , triple_day_start )

    print('eve_morn_start' , eve_morn_start )
    print('other_b2b_start' , other_b2b_start )
    #shifted_slots = [int(i) for i in np.roll(slots, -1)]
    next_slot = dict(zip(slots, shifted_slots))
    print('next_slot' , next_slot)


    print('tripi_tropi' , tripi_tropi)

    m = gp.Model('Scheduler', env=env)

    x = m.addVars(block_sequence_slot, vtype=GRB.BINARY, name='group_seq_indicator')
    y = m.addVars(block_sequence_trip, vtype=GRB.BINARY, name='y')
    schedule = m.addVars(slots, vtype=GRB.INTEGER, name='slot_assignment')
    b = m.addVars(block_slot, vtype=GRB.BINARY, name='b')
    block_assigned = m.addVars(slots, vtype=GRB.INTEGER, name='block_assigned')
    block_diff = m.addVars(block_pair, vtype=GRB.INTEGER, name='block_diff')
    block_diff_large = m.addVars(block_pair, vtype=GRB.BINARY, name='block_diff_large')

    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in slots) == 1 for i in slots))
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for k in slots for s in slots) == 1 for j in slots))
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for j in slots for s in slots) == 1 for k in slots))
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for j in slots for k in slots) == 1 for s in slots))
    m.addConstrs((gp.quicksum(b[i, s] for i in slots) == 1 for s in slots), name="slot_unique_assignment" )
    m.addConstrs((x[(i, i, k, s)] == 0 for i in slots for k in slots for s in slots))
    m.addConstrs((x[(i, j, i, s)] == 0 for i in slots for j in slots for s in slots))
    m.addConstrs((x[(i, j, j, s)] == 0 for i in slots for j in slots for s in slots))

    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots) == gp.quicksum(x[(j, k, l, next_slot[s])] for l in slots)
                  for j in slots for k in slots for s in slots))
    print('triple_slots ' , triple_slots)
    m.addConstrs(y[(i, j, k)] == gp.quicksum(x[(i, j, k, s)] for s in triple_slots) for i in triple_slots for j in triple_slots for k in triple_slots)

    m.addConstrs((schedule[s] == gp.quicksum(i * x[(i, j, k, s)] for i in slots for j in slots for k in slots) for s in slots))
    m.addConstrs((b[i, s] == gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots) for i in slots for s in slots))
    m.addConstrs((block_assigned[i] == gp.quicksum(s * b[i, s] for s in slots) for i in slots))

    m.addConstrs((block_diff[(i, j)] >= block_assigned[i] - block_assigned[j] for i in slots for j in slots))
    m.addConstrs((block_diff[(i, j)] >= block_assigned[j] - block_assigned[i] for i in slots for j in slots))

    big_m = 20
    c = 16
    m.addConstrs((block_diff[(i, j)] >= c * block_diff_large[(i, j)] for i in slots for j in slots))
    m.addConstrs((block_diff[(i, j)] <= c - 1 + big_m * block_diff_large[(i, j)] for i in slots for j in slots))

    m.addConstrs(gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in early_slots_1) == 1 for i in large_blocks_1)
    m.addConstrs(gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in early_slots_2) == 1 for i in large_blocks_1)

    triple_in_day_var = m.addVar(vtype=GRB.INTEGER, name='triple_in_day')
    triple_in_24hr_var = m.addVar(vtype=GRB.INTEGER, name='triple_in_24hr')
    b2b_eveMorn_var = m.addVar(vtype=GRB.INTEGER, name='b2b_eveMorn')
    b2b_other_var = m.addVar(vtype=GRB.INTEGER, name='b2b_other')
    three_exams_four_slots_var = m.addVar(vtype=GRB.INTEGER, name='three_exams_four_slots')
    first_slot_penalty = m.addVar(vtype=GRB.INTEGER, name='first_slot_penalty')
    two_slot_diff_penalty = m.addVar(vtype=GRB.INTEGER, name='two_slot_diff_penalty')
    two_exams_largegap = m.addVar(vtype=GRB.INTEGER, name='two_exams_largegap')

    m.addConstr((gp.quicksum(triple_coenrollment_counts[(i, j, k)] * x[(i, j, k, s)] for i in slots for j in slots for k in slots for s in list(triple_in_day)) == triple_in_day_var))
    m.addConstr((gp.quicksum(triple_coenrollment_counts[(i, j, k)] * x[(i, j, k, s)] for i in slots for j in slots for k in slots for s in list(triple_in_24hr)) == triple_in_24hr_var))
    m.addConstr((gp.quicksum(pairwise_coenrollment_counts[(i, j)] * x[(i, j, k, s)] for i in slots for j in slots for k in slots for s in list(eve_morn_start)) == b2b_eveMorn_var))
    m.addConstr((gp.quicksum(pairwise_coenrollment_counts[(i, j)] * x[(i, j, k, s)] for i in slots for j in slots for k in slots for s in list(other_b2b_start)) == b2b_other_var))

    m.addConstr((gp.quicksum(b[i, s] * student_unique_block[i] * first_list[s - 1] for i in slots for s in slots) == first_slot_penalty))
    m.addConstr((gp.quicksum(block_diff[(i, j)] * student_unique_block_pairs[(i, j)] for i in slots for j in slots) == two_slot_diff_penalty))
    m.addConstr((gp.quicksum(block_diff_large[(i, j)] * student_unique_block_pairs[(i, j)] for i in slots for j in slots if j >= i) == two_exams_largegap))

    m.setObjective(
        alpha * triple_in_day_var +
        beta * triple_in_24hr_var +
        gamma1 * b2b_eveMorn_var +
        gamma2 * b2b_other_var +
        delta * three_exams_four_slots_var +
        vega * first_slot_penalty +
        theta * two_exams_largegap +
        lambda_large1 * gp.quicksum(first_list[s - 1] * b[i, s] for i in large_blocks_1 for s in slots) +
        lambda_large2 * gp.quicksum(first_list[s - 1] * b[i, s] for i in large_blocks_2 for s in slots) +
        lambda_big * gp.quicksum(first_list[s - 1] * b[i, s] for i in big_blocks for s in slots),
        GRB.MINIMIZE
    )

    m.setParam('Timelimit', 600 )
    m.update()
    if read and readfile is not None:
      if os.path.exists(readfile):
          print("Warm start file found. Reading from:", readfile)
          m.read(readfile)
      else:
          print("Warm start file not found at:", readfile)

    if read and readfile is not None and os.path.exists(readfile):
      print('IN READ FILE')
      m.read(readfile)
    m.optimize()
    if m.status in [GRB.OPTIMAL, GRB.INTERRUPTED, GRB.TIME_LIMIT]:
        print('IN WRITEFILE')
        m.write(  writefile)
    if m.status == GRB.INFEASIBLE:
      print("Model is infeasible. Please check the IIS and adjust your constraints.")
      m.computeIIS()
      m.write("model.ilp")
    output = pd.DataFrame(columns=['slot', 'block'])

    new_row = pd.DataFrame({'slot': [k], 'block': [schedule[k].x]})
    if not new_row['block'].isna().all():
        output = pd.concat([output, new_row], ignore_index=True)
    output['slot'] = output['slot'].astype(int)
    obj = m.getObjective().getValue()


    return obj, output, 0


get_seq_name = lambda p :   str(num_blocks) + assignment_type + 'size' + str(size_cutoff) + 'res' + str(reserved) +'large_b' +str(p['large_b']) +'big_exam' + str(p['big_exam']) + 'big_block' + str(p['big_block']) + 'large_1' + str(p['large_1']) +  'large_2' +str(p[ 'large_2'])

# Load some input data
si = pd.read_csv('/home/asj53/final-scheduling/data/sp25/exam_sizes.csv').set_index('exam')

def generate_param_string(**kwargs):
    """
    Generates a string by concatenating each parameter's name with its value.

    Example:
        generate_param_string(alpha=5, beta=6)
        returns: 'alpha5beta6'
    """
    return 'new'.join(f"{key}{value}" for key, value in kwargs.items())

# Define file locations for warm start and run counter.
warm_start_file = '/home/asj53/final-scheduling/warm_start.sol'
counter_file = '/home/asj53/final-scheduling/run_counter.txt'

# Check if the run counter file exists and read the counter.
if os.path.exists(counter_file):
    with open(counter_file, 'r') as f:
        run_count = int(f.read().strip())
else:
    run_count = 0

# Use warm start if this is not the first run.
use_warm_start = (run_count > 0)

# Increment the counter and write it back.
run_count += 1
with open(counter_file, 'w') as f:
    f.write(str(run_count))

# Example objective function for schedule quality (lower is better)
def exam_schedule_objective(alpha, gamma, delta, vega, theta, large_block_size, large_exam_weight, large_block_weight, large_size_1, large_size_2):
    size_cutoff_1 = large_size_1
    exam_block_size = ba[['Exam Group', 'Exam Block']].merge(
        exam_sizes, how='inner', left_on='Exam Group', right_on='exam'
    )[["Exam Group", "Exam Block", "size"]]
    size_cutoff_2 = large_size_2
    reserved_blocks = []

    alpha1 = beta1 = alpha
    gamma3 = gamma
    delta1 = delta
    size_cutoff_1_weight = large_exam_weight
    size_cutoff_2_weight = large_exam_weight

    lambda_big_exam_1 = 1 * size_cutoff_1_weight
    lambda_big_exam_2 = 2 * size_cutoff_2_weight
    large_block = large_block_size
    lambda_big_block  = large_block_weight
  
    first1 = first_list
    slot_cutoff_1 = 24
    slot_cutoff_2 = 24
    early_slots_1 = list(range(1, slot_cutoff_1))
    early_slots_2 = list(range(1, slot_cutoff_2))

    large_blocks_1 = exam_block_size[exam_block_size['size'] > size_cutoff_1]['Exam Block'].unique()
    large_blocks_2 = exam_block_size[
        (exam_block_size['size'] > size_cutoff_2) & (exam_block_size['size'] <= size_cutoff_1)
    ]['Exam Block'].unique()

    block_to_size = exam_block_size.groupby('Exam Block').sum('size')
    big_blocks = block_to_size[block_to_size['size'] > large_block].index

    # Print parameters for debugging.
    print("alpha1:", alpha1)
    print("beta1:", beta1)
    print("gamma3 (1st):", gamma3)
    print("gamma3 (2nd):", gamma3)
    print("delta1:", delta1)
    print("theta1:", theta)
    print("vega:", vega)

    print("lambda_big_exam_1:", lambda_big_exam_1)
    print("lambda_big_exam_2:", lambda_big_exam_2)
    print("lambda_big_block:", lambda_big_block)
    print("size_cutoff_1:", size_cutoff_1)
    print("size_cutoff_2:", size_cutoff_2)
    print("large_blocks_1:", large_blocks_1)
    print("large_block:", large_block)
    print("big_blocks:", big_blocks)
    print("large_blocks_2:", large_blocks_2)
    print("early_slots_1:", early_slots_1)
    print("early_slots_2:", early_slots_2)
    print("slots:", slots)
    print("use_warm_start:", use_warm_start)
    print("warm_start_file:", warm_start_file)
    # Pass the warm start parameters to schedule_ip.
    obj, schedule, penalty = schedule_ip(
        alpha=alpha1,
        beta=beta1,
        vega = vega, 
        theta = theta, 
        gamma1=gamma3,
        gamma2=gamma3,
        delta=delta1,
        triple_coenrollment_counts=triple_coenrollment_counts,
        pairwise_coenrollment_counts=pairwise_coenrollment_counts,
        lambda_large1=lambda_big_exam_1,
        lambda_large2=lambda_big_exam_1,
        lambda_big=lambda_big_block,
        large_blocks_1=large_blocks_1,
        large_blocks_2=large_blocks_2,
        big_blocks=big_blocks,
        early_slots_1=early_slots_1,
        early_slots_2=early_slots_2,
        slots=slots,
        read=use_warm_start,         # Use warm start if this is not the first run.
        readfile=warm_start_file     # File from which to read the warm start.
    )

    try: 
        columns = ['Exam Group', 'block', 'Exam Slot']
        out = generate_param_string(
            alpha=alpha1,
            beta=beta1,
            vega = vega , 
            theta = theta, 
            gamma1=gamma3,
            gamma2=gamma3,
            delta=delta1,
            lambda_large1=lambda_big_exam_1,
            lambda_large2=lambda_big_block,
        )
        print('ba' , ba )
        local_ba = ba.copy()
        exam_block_size = local_ba[['Exam Group', 'Exam Block']].merge(
            exam_sizes, how='inner', left_on='Exam Group', right_on='exam'
        )[["Exam Group", "Exam Block", "size"]]
        print('schedule:', schedule)

        block_to_slot = dict(zip(schedule['block'], schedule['slot']))
        pd.DataFrame(block_to_slot.items(), columns=['Exam Block', 'Exam Slot']).to_csv(out + 'sched.csv')
        local_ba['new_slot'] = local_ba['Exam Block'].apply(
            lambda x: block_to_slot.get(x, np.nan)
        )

        # Construct the exam_group_schedule DataFrame using the updated slot.
        exam_group_schedule = local_ba[['Exam Group', 'Exam Block', 'new_slot']].copy()
        print('exam_group_schedule:', exam_group_schedule)
        exam_group_schedule.columns = ['exam', 'Exam Block', 'slot']
        schi = exam_group_schedule
        m_df = schi.join(si).reset_index()
        m_df.to_csv('/home/asj53/final-scheduling/results/sp25/schedules/' + out + '.csv')
        get_metrics(m_df , out)
        
        # Return the score (lower score is better).
        mets = pd.read_csv('/home/asj53/final-scheduling/results/sp25/metrics/' + out + 'metrics.csv')

        return (
            int(mets['triple in 24h (no gaps)'].iloc[0] + mets['triple in same day (no gaps)'].iloc[0]),
            int(mets['three in four slots'].iloc[0]),
            int(mets['evening/morning b2b'].iloc[0] + mets['other b2b'].iloc[0]),
            int(mets['two in three slots'].iloc[0]),
            int(mets['singular late exam'].iloc[0]),
            int(mets['two exams, large gap'].iloc[0]),
            int(mets['avg_max'].iloc[0]),
            int(mets['lateness'].iloc[0])
        )

        
    except:
        return (200, 40000, 3000, 200, 40000, 3000, 200, 40000)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Function to compute the Pareto frontier from a set of objective values.
def compute_pareto_frontier(Y):
    """
    Y: an (n_points, 2) array where each row is [obj1, obj2] to be minimized.
    Returns a boolean mask indicating which points are Pareto optimal.
    """
    n_points = Y.shape[0]
    is_pareto = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # For minimization: if every objective of j is less than or equal to i and at least one is strictly less, i is dominated.
                if np.all(Y[j] <= Y[i]) and np.any(Y[j] < Y[i]):
                    is_pareto[i] = False
                    break
    return is_pareto

# Define parameter bounds for each of the 5 parameters:
# (Adjust these as needed; currently using 8 parameters as per your bounds.)
bounds = np.array([
    [10, 100],
    [5, 50],
    [0,10] , 
    [0,10] , 
    [0, 10],
    [1000, 2000],  # large_block_size
    [0, 100],      # large_exam_weight
    [0, 100],      # large_block_weight
    [200, 400],    # large_size_1
    [100, 200]     # large_size_2
])

# Number of initial random evaluations
n_initial = 10
dim = bounds.shape[0]

# Initialize with random parameters
X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial, dim))


import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Assume exam_schedule_objective returns at least 9 values.
n_objectives = 8
def evaluate_candidate(candidate):
    return exam_schedule_objective(*candidate)

with ProcessPoolExecutor(max_workers=5) as executor:
    initial_results = list(executor.map(evaluate_candidate, X))

# Extract the first n_objectives for each evaluation.
Y_list = [result[:n_objectives] for result in initial_results]
print('INITIAL RES ' , initial_results)
print('YLIST ' , Y_list)
Y = np.array(Y_list)  # Now Y has shape (n_initial, 9)

# -------------------------------
# Thompson Sampling parameters
# -------------------------------
n_iterations = 100
n_candidates = 1000  # Number of candidate points to sample for optimizing the scalarized function

# -------------------------------
# Thompson Sampling loop (with 5 candidates evaluated concurrently)
# -------------------------------
for it in range(n_iterations):
    # Fit a GP model for each objective.
    kernel = Matern(nu=2.5)
    gps = []
    for obj in range(n_objectives):
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True).fit(X, Y[:, obj])
        gps.append(gpr)
    
    # Random scalarization: sample weights from a Dirichlet distribution over 9 objectives.
    weights = np.random.dirichlet(np.ones(n_objectives))
    
    # Generate candidate points uniformly over the parameter space.
    candidate_X = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, dim))
    
    # For each candidate, sample one function realization from each GP and compute the weighted sum.
    scalarized = np.zeros(n_candidates)
    for i, gpr in enumerate(gps):
        sample = gpr.sample_y(candidate_X, n_samples=1).flatten()
        scalarized += weights[i] * sample
    
    # Select the top 5 candidates that minimize the scalarized function.
    top_indices = np.argsort(scalarized)[:5]
    best_candidates = candidate_X[top_indices]
    
    with ProcessPoolExecutor(max_workers=20) as executor:
        batch_results = list(executor.map(evaluate_candidate, best_candidates))
    
    # Update our dataset with the results from these 5 evaluations.
    for candidate, result in zip(best_candidates, batch_results):
        objectives = result[:n_objectives]
        X = np.vstack([X, candidate])
        Y = np.vstack([Y, objectives])
        print(f"Iteration {it+1}: Candidate = {candidate}, Objectives = {objectives}")


param_names = ['alpha', 'gamma', 'delta', 'vega', 'theta', 
               'large_block_size', 'large_exam_weight', 'large_block_weight', 
               'large_size_1', 'large_size_2']
obj_names = ['total_triple',    # triple in 24h + triple in same day
             'three_in_four',   # three in four slots
             'total_b2b',       # evening/morning + other b2b
             'two_in_three',    # two in three slots
             'singular_late',   # singular late exam count
             'two_exams_large_gap',  # two exams, large gap
             'avg_max',         # average maximum exam slot across students
             'lateness']        # total weighted lateness

# Horizontally stack the candidate parameters (X) and objective values (Y).
data = np.hstack((X, Y))
columns = param_names + obj_names
df_results = pd.DataFrame(data, columns=columns)

# Compute Pareto optimality across all 8 objectives.
# (The provided function works generically; it marks a point as Pareto optimal if no other candidate dominates it.)
df_results['pareto'] = compute_pareto_frontier(Y)

# Display the first few rows of the DataFrame.
print(df_results.head())

# Save the DataFrame to a CSV file in /home/asj53/
df_results.to_csv("/home/asj53/GaussinSamplingbo_results20.csv", index=False)