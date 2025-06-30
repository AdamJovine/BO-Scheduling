import numpy as np
import pandas as pd
import math
import os
from matplotlib import pyplot as plt
import seaborn as sns
from itertools import product
from config.settings import SAVE_PATH, DATA_PATH, UI_PATH , SEMESTER
from globals.build_global_sets import normalize_and_merge
times = ['']
def slots_to_time(slots):
    d = {}
    if 'fa' in SEMESTER : 
        d=  {
            1: 'Dec 13, 9am', 
            2: 'Dec 13, 2pm',
            3: 'Dec 13, 7pm',
            4: 'Dec 14, 9am',
            5: 'Dec 14, 2pm',
            6: 'Dec 14, 7pm',
            7: 'Dec 15, 9am',
            8: 'Dec 15, 2pm',
            9: 'Dec 15, 7pm',
            10: 'Dec 16, 9am',
            11: 'Dec 16, 2pm',
            12: 'Dec 16, 7pm',
            13: 'Dec 17, 9am',
            14: 'Dec 17, 2pm',
            15: 'Dec 17, 7pm',
            16: 'Dec 18, 9am',
            17: 'Dec 18, 2pm',
            18: 'Dec 18, 7pm',
            19: 'Dec 19, 9am',
            20: 'Dec 19, 2pm',
            21: 'Dec 19, 7pm',
            22: 'Dec 20, 9am',
            23: 'Dec 20, 2pm',
            24: 'Dec 20, 7pm',
            25: 'Dec 21, 9am',
            26: 'Dec 21, 2pm',
            27: 'Dec 21, 7pm'}
    else:  
        d=   {
        1:  'May 11, 9am',
        2:  'May 11, 2pm',
        3:  'May 11, 7pm',
        4:  'May 12, 9am',
        5:  'May 12, 2pm',
        6:  'May 12, 7pm',
        7:  'May 13, 9am',
        8:  'May 13, 2pm',
        9:  'May 13, 7pm',
        10: 'May 14, 9am',
        11: 'May 14, 2pm',
        12: 'May 14, 7mm',
        13: 'May 15, 9am',
        14: 'May 15, 2pm',
        15: 'May 15, 7pm',
        16: 'May 16, 9am',
        17: 'May 16, 2pm',
        18: 'May 16, 7pm',
        19: 'May 17, 9am',
        20: 'May 17, 2pm',
        21: 'May 17, 7pm',
        22: 'May 18, 9am',
        23: 'May 18, 2pm',
        24: 'May 18, 7pm', 
        25: 'May 19, 9am', 
        26: 'May 19, 2pm',
        27: 'May 19, 7pm' }
    
    return [d[s] for s in slots]
# Create the chart
def get_plot(schedule_name, name):
  sched = pd.read_csv(SAVE_PATH +'/schedules/' + schedule_name )
  exam_sizes = pd.read_csv(DATA_PATH + '/exam_sizes.csv')
  slots = np.unique(sched['slot'].values)

  num_slots1 = len(slots)
  num_slots2 = int(max(slots))
  h = np.zeros(num_slots2)
  h1 = np.zeros(num_slots2)
  h2 = np.zeros(num_slots2)
  h3 = np.zeros(num_slots2)
  h4 = np.zeros(num_slots2)
  for s in slots:
      s = int(s)
      exams = sched[sched['slot']==s]['exam'].tolist()
      exams_over_400 = sched[(sched['slot']==s) & (sched['size']>= 400)]['exam'].tolist()
      exams_in_300_400 = sched[(sched['slot']==s) & (sched['size']>= 300) & (sched['size']< 400)]['exam'].tolist()
      exams_in_200_300 = sched[(sched['slot']==s) & (sched['size']>= 200) & (sched['size']< 300)]['exam'].tolist()
      exams_in_100_200 = sched[(sched['slot']==s) & (sched['size']>= 100) & (sched['size']< 200)]['exam'].tolist()
      sizes_over_400 = exam_sizes[exam_sizes['exam'].isin(exams_over_400)]['size'].sum()
      sizes_in_300_400 = exam_sizes[exam_sizes['exam'].isin(exams_in_300_400)]['size'].sum()
      sizes_in_200_300 = exam_sizes[exam_sizes['exam'].isin(exams_in_200_300 )]['size'].sum()
      sizes_in_100_200 = exam_sizes[exam_sizes['exam'].isin(exams_in_100_200 )]['size'].sum()
      sizes = exam_sizes[exam_sizes['exam'].isin(exams)]['size'].sum()
      h[s-1] = sizes
      h1[s-1] = sizes_over_400
      h2[s-1] = sizes_in_300_400
      h3[s-1] = sizes_in_200_300
      h4[s-1] = sizes_in_100_200

  plt.style.use('classic')
  plt.figure(figsize=(18, 12))

  # plt.bar(x=slots, height=[max(h)]*num_slots1, color='red', alpha=0.4, width = 1, align = 'center')       
  plt.bar(x=range(1,num_slots2+1), height=h1, align='center', width=1, 
          color = 'tab:red', label = "Exams w/ over 400 students")
  plt.bar(x=range(1,num_slots2+1), height=h2, align='center', width=1, 
          bottom = h1, color = 'tab:orange', label = "Exams w/ over 300 but less than 400 students")
  plt.bar(x=range(1,num_slots2+1), height=h3, align='center', width=1, 
          bottom = h1+h2, color = 'gold', label = "Exams w/ over 200 but less than 300 students")
  plt.bar(x=range(1,num_slots2+1), height=h4, align='center', width=1, 
          bottom = h1+h2+h3, color = 'pink', label = "Exams w/ over 100 but less than 200 students")

  plt.bar(x=range(1,num_slots2+1), height=h-h1-h2-h3-h4, align='center',
          bottom = h1+h2+h3+h4, width=1, color = 'tab:purple', label = "Other Exams")

  plt.xlabel('Times', fontsize=20)
  plt.xticks(np.arange(1, num_slots2 + 1), slots_to_time(np.arange(1, num_slots2 + 1)), rotation = 90, fontsize=16)
  plt.yticks(fontsize = 16)
  plt.ylabel('Number of students',  fontsize=20)
  plt.title('Number of students taking an exam in each time slot',  fontsize=25)
  plt.legend(loc = 'best', fontsize=14)
  plt.savefig(UI_PATH + name + '.png')
  
  plt.show()

def last_day(sched_name, save_name):
    #goop['Exam Block'] = 
    #sched, by_student_block = normalize_and_merge(goop,)
    sched = pd.read_csv(SAVE_PATH + '/schedules/' + sched_name)
    print(sched)
    enrl_df = pd.read_csv(DATA_PATH + '/enrl.csv')
    enrl_df = enrl_df.merge(sched, left_on = 'Exam Key', right_on = 'Exam Group')
    by_student_block = enrl_df.groupby('anon-netid')['slot'].apply(list).reset_index(name='slots') #create_by_student_slot_df(exam_df, schedule_dict)
    by_student_block['last_block'] = by_student_block['slots'].apply(lambda x: max(x)).copy()
    last_block_counts = by_student_block['last_block'].value_counts().reset_index()
    last_block_counts.columns = ['last_block', 'occurrences']

    last_block_counts = last_block_counts.sort_values(by='last_block').reset_index(drop=True)
    print('last_block_counts' , last_block_counts )

    slots = np.unique(sched['slot'].values)
    # Ensure num_slots2 is an integer for range function
    num_slots2 = int(max(slots)) if len(slots) > 0 else 0

    print('slot , ' , slots)
    h = np.zeros(num_slots2)

    # Convert last_block_counts to a dictionary for efficient lookup
    counts_dict = last_block_counts.set_index('last_block')['occurrences'].to_dict()

    for s in range(1, num_slots2 + 1): # Iterate through all possible slot numbers
        # Get the occurrence count from the dictionary, defaulting to 0 if not found
        h[s-1] = counts_dict.get(float(s), 0)

    plt.style.use('classic')
    plt.figure(figsize=(18, 12))
    plt.bar(x=range(1,num_slots2+1), height=h, align='center', width=1, color = 'pink')

    plt.xlabel('Times', fontsize=20)
    # Ensure the ticks cover all possible slots up to num_slots2
    plt.xticks(np.arange(1, num_slots2 + 1), slots_to_time(np.arange(1, num_slots2 + 1)), rotation = 90, fontsize=16)
    plt.yticks(fontsize = 16)
    plt.ylabel('Number of students',  fontsize=20)
    plt.title('Number of students taking their last exam in each time slot',  fontsize=25)
    plt.savefig(UI_PATH +save_name+ '_dist.png' )
    plt.show()
    