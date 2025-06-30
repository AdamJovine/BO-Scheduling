import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os
from config.settings import SAVE_PATH , EMPTY_BLOCKS, NUM_SLOTS ,SEQ_TIME,  GLOBAL_TIMESTAMP, get_name
from globals.build_global_sets import normalize_and_merge 
def build_scheduler_model(params, coenrollment_data, env, global_sets):
    def _truncate(obj, length=1000):
        s = repr(obj)
        return s[:length] + ("…" if len(s) > length else "")
    
    #print("params:", _truncate(params))

    slots = global_sets["slots"]
    #print("slots:", _truncate(slots))

    next_slot = global_sets["next_slot"]
    #print("next_slot:", _truncate(next_slot))

    triple_slots = global_sets["triple_in_day"] + global_sets["triple_in_24hr"]
    #print("triple_slots:", _truncate(triple_slots))

    block_sequence_slot = global_sets["block_sequence_slot"]
    #print(
    #    f"block_sequence_slot (len={len(block_sequence_slot)}): "
    #    f"{_truncate(block_sequence_slot)}"
    #)

    block_sequence_trip = global_sets["block_sequence_trip"]
    #print(
    #    f"block_sequence_trip (len={len(block_sequence_trip)}): "
    #   f"{_truncate(block_sequence_trip)}"
    #)

    block_pair = global_sets["block_pair"]
    #print(
    #    f"block_pair (len={len(block_pair)}): "
    #    f"{_truncate(block_pair)}"
    #)

    block_slot = global_sets["block_slot"]
    #print(
    #   f"block_slot (len={len(block_slot)}): "
    #   f"{_truncate(block_slot)}"
    #)

    first_list = global_sets["first_list"]
    #print("first_list:", _truncate(first_list))
    m = gp.Model("Scheduler", env=env)
    # Variables
    x = m.addVars(block_sequence_slot, vtype=GRB.BINARY, name="group_seq_indicator")
    y = m.addVars(block_sequence_trip, vtype=GRB.BINARY, name="y")
    schedule = m.addVars(slots, vtype=GRB.INTEGER, name="slot_assignment")
    b = m.addVars(block_slot, vtype=GRB.BINARY, name="b")
    block_assigned = m.addVars(slots, vtype=GRB.INTEGER, name="block_assigned")
    block_diff = m.addVars(block_pair, vtype=GRB.INTEGER, name="block_diff")
    block_diff_large = m.addVars(block_pair, vtype=GRB.BINARY, name="block_diff_large")

    # Core constraints
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in slots) == 1 for i in slots), name="core1")
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for k in slots for s in slots) == 1 for j in slots), name="core2")
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for j in slots for s in slots) == 1 for k in slots), name="core3")
    m.addConstrs((gp.quicksum(x[(i, j, k, s)] for i in slots for j in slots for k in slots) == 1 for s in slots), name="core4")

    m.addConstrs((gp.quicksum(b[i, s] for i in slots) == 1 for s in slots), name="slot_unique_assignment")

    m.addConstrs((x[(i, i, k, s)] == 0 for i in slots for k in slots for s in slots),name="zero1")
    m.addConstrs((x[(i, j, i, s)] == 0 for i in slots for j in slots for s in slots),name="zero2")
    m.addConstrs((x[(i, j, j, s)] == 0 for i in slots for j in slots for s in slots),name="zero3")

    m.addConstrs((
        gp.quicksum(x[(i, j, k, s)] for i in slots) ==
        gp.quicksum(x[(j, k, l, next_slot[s])] for l in slots)
        for j in slots for k in slots for s in slots
    ),name="next1")

    for (i, j, k) in block_sequence_trip:
        m.addConstr(
            y[(i, j, k)] ==
            gp.quicksum(x[(i, j, k, s)] for s in triple_slots),name="trips"
        )

    m.addConstrs((schedule[s] == gp.quicksum(i * x[(i, j, k, s)] for i in slots for j in slots for k in slots) for s in slots),name="sched")
    m.addConstrs((b[i, s] == gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots) for i in slots for s in slots),name="bis")
    m.addConstrs((block_assigned[i] == gp.quicksum(s * b[i, s] for s in slots) for i in slots),name="block assign")

    m.addConstrs((block_diff[(i, j)] >= block_assigned[i] - block_assigned[j] for i in slots for j in slots),name="block_diff1")
    m.addConstrs((block_diff[(i, j)] >= block_assigned[j] - block_assigned[i] for i in slots for j in slots),name="block_diff2")

    big_m = 20
    c = 16
    m.addConstrs((block_diff[(i, j)] >= c * block_diff_large[(i, j)] for i in slots for j in slots),name="block_diff3")
    m.addConstrs((block_diff[(i, j)] <= c - 1 + big_m * block_diff_large[(i, j)] for i in slots for j in slots),name="block_diff4")

    # Early slot constraints
    #print('params["early_slots_1"] : ' , params["early_slots_1"] )
    #print("params['big_blocks'] ;" , params['big_blocks'])
    m.addConstrs(
        (gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in params["early_slots_1"]) == 1 for i in params['big_blocks'] ), 
        name="early1")

    
    #m.addConstrs(
    #    gp.quicksum(x[(i, j, k, s)] for j in slots for k in slots for s in params["big_blocks"]) == 1
    #    for i in params["big_blocks"]
    #)

    for i, reserved in enumerate(EMPTY_BLOCKS):
        print('aggg ',NUM_SLOTS - i , j, k, int(reserved))
        m.addConstr(gp.quicksum(x[(NUM_SLOTS - i ,j,k,int(reserved))] for j in slots for k in slots) == 1)

    # Penalty variables
    triple_in_day_var = m.addVar(vtype=GRB.INTEGER, name="triple_in_day")
    triple_in_24hr_var = m.addVar(vtype=GRB.INTEGER, name="triple_in_24hr")
    b2b_eveMorn_var = m.addVar(vtype=GRB.INTEGER, name="b2b_eveMorn")
    b2b_other_var = m.addVar(vtype=GRB.INTEGER, name="b2b_other")
    three_exams_four_slots_var = m.addVar(vtype=GRB.INTEGER, name="three_exams_four_slots")
    first_slot_penalty = m.addVar(vtype=GRB.INTEGER, name="first_slot_penalty")
    two_slot_diff_penalty = m.addVar(vtype=GRB.INTEGER, name="two_slot_diff_penalty")
    two_exams_largegap = m.addVar(vtype=GRB.INTEGER, name="two_exams_largegap")

    # Coenrollment penalties (use .get to prevent KeyErrors)
    m.addConstr(
        gp.quicksum(
            coenrollment_data["triple"].get((i, j, k), 0) * x[(i, j, k, s)]
            for i in slots for j in slots for k in slots for s in global_sets["triple_in_day"]
        ) == triple_in_day_var
    ,name="trip in day")

    m.addConstr(
        gp.quicksum(
            coenrollment_data["triple"].get((i, j, k), 0) * x[(i, j, k, s)]
            for i in slots for j in slots for k in slots for s in global_sets["triple_in_24hr"]
        ) == triple_in_24hr_var
    ,name="trip in 24")

    m.addConstr(
        gp.quicksum(
            coenrollment_data["pairwise"].get((i, j), 0) * x[(i, j, k, s)]
            for i in slots for j in slots for k in slots for s in global_sets["eve_morn_start"]
        ) == b2b_eveMorn_var
    ,name="even morn btb")

    m.addConstr(
        gp.quicksum(
            coenrollment_data["pairwise"].get((i, j), 0) * x[(i, j, k, s)]
            for i in slots for j in slots for k in slots for s in global_sets["other_b2b_start"]
        ) == b2b_other_var
    ,name="other b2b")

    m.addConstr(
        gp.quicksum(
            b[i, s] * coenrollment_data["student_unique_block"].get(i, 0) * first_list[s - 1]
            for i in slots for s in slots
        ) == first_slot_penalty
    ,name="unique")

    m.addConstr(
        gp.quicksum(
            block_diff[(i, j)] * coenrollment_data["student_unique_block_pairs"].get((i, j), 0)
            for i in slots for j in slots
        ) == two_slot_diff_penalty
    ,name="unique_pair1")

    m.addConstr(
        gp.quicksum(
            block_diff_large[(i, j)] * coenrollment_data["student_unique_block_pairs"].get((i, j), 0)
            for i in slots for j in slots if j >= i
        ) == two_exams_largegap
    ,name="unique_pair2")

    # Objective
    m.setObjective(
        params["alpha"] * triple_in_day_var +
        params["beta"] * triple_in_24hr_var +
        params["gamma1"] * b2b_eveMorn_var +
        params["gamma2"] * b2b_other_var +
        params["delta"] * three_exams_four_slots_var +
        params["vega"] * first_slot_penalty +
        params["theta"] * two_exams_largegap +
        params["lambda_large1"] * gp.quicksum(first_list[s - 1] * b[i, s] for i in params["large_blocks_1"] for s in slots) +
        params["lambda_large2"] * gp.quicksum(first_list[s - 1] * b[i, s] for i in params["large_blocks_2"] for s in slots) +
        params["lambda_big"] * gp.quicksum(first_list[s - 1] * b[i, s] for i in params["big_blocks"] for s in slots),
        GRB.MINIMIZE
    )

    return m


def solve_model(model, params=None, warm_start_path=None):


    if warm_start_path and os.path.exists(warm_start_path):
        print(f"Warm start file found at {warm_start_path}")
        model.read(warm_start_path)

    model.setParam('TimeLimit', SEQ_TIME)
    model.setParam('OutputFlag', 0)      # removes IP logs 
    model.optimize()
    
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible.")
        model.computeIIS()
        model.write("model_{}.ilp")

    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
        schedule = {}
        for v in model.getVars():
            if v.VarName.startswith('slot_assignment') and v.X != 0:
                slot = int(v.VarName.split('[')[1].rstrip(']'))
                block = int(v.X)
                schedule[block] = slot

        # If params provided, save schedule
        
        print('SCHEDULEEE' , schedule)
        return schedule, model.ObjVal

    return None, None


def sequencing(param_dict, global_sets, license_env, block_path):

    #try:
        # Build and solve the scheduling model
        env = gp.Env(params=license_env)
        model = build_scheduler_model(
            params=param_dict,
            coenrollment_data={
                "triple": global_sets["triple"],
                "pairwise": global_sets["pairwise"],
                "student_unique_block": global_sets["student_unique_block"],
                "student_unique_block_pairs": global_sets["student_unique_block_pairs"]
            },
            env=env,
            global_sets=global_sets
        )

        schedule_dict, obj_val = solve_model(model)
        if schedule_dict is None:
            print(
                'param_d' , param_dict , 
                'global_s' , global_sets , 
                'blockpath  , ' , block_path 
            )
            #raise ValueError("No valid schedule")
            return None, None 
        schedule_df = pd.DataFrame([
            {'block': exam, 'slot': slot}
            for slot, exam in schedule_dict.items()
        ])
        name = get_name(param_dict, global_ts=GLOBAL_TIMESTAMP, block_assignment=block_path) 
        schedule_df.to_csv(
            SAVE_PATH + '/dictionaries/' +name  +'.csv'
        )
        print("SCHEDULE DICT" , schedule_dict)

        env.dispose() 
        
        return schedule_dict, name 

    #except Exception as e:
    #    print("Objective function failed:", e)
    #
    #    return {}, (200, 40000, 3000, 200, 40000, 3000, 200, 40000)
