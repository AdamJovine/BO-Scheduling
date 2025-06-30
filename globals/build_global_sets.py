import numpy as np
import pandas as pd
from collections import defaultdict
from block_assignment.helpers import cleanup
from config.settings import SAVE_PATH


import os
import pandas as pd
import numpy as np

def normalize_and_merge(ba: pd.DataFrame, exam_df: pd.DataFrame, block_config):
    """
    1) Normalize the “Exam Group” and assign adj_block
    2) Merge with exam_df to produce by_student_block
    Returns:
      - ba (with new column 'adj_block')
      - by_student_block (DataFrame with columns ['anon-netid', 'blocks'])
    """
    seen_path = os.path.join(SAVE_PATH,'cache', 'seen.csv')
    block_config = str(block_config)
    # Load or initialize the cache of configs
    if os.path.exists(seen_path):
        seen = pd.read_csv(seen_path)  # expects one column: 'configs'
    else:
        # create an empty cache DataFrame
        seen = pd.DataFrame({'configs': []})
        seen.to_csv(seen_path, index=False)
    
    # If we've already done this config, just load the cached outputs
    if block_config in seen['configs'].values:
        ba = pd.read_csv(os.path.join(SAVE_PATH, 'cache',f"{block_config}ba.csv"))
        by_student_block = pd.read_csv(os.path.join(SAVE_PATH, 'cache',f"{block_config}by_student_block.csv"))
    else:
        ba = ba.copy()
        # Normalize "Exam Group"
        if 'exam' in ba.columns:
            ba['Exam Group'] = ba['exam'].astype(str)
        ba['Exam Group'] = ba['Exam Group'].apply(cleanup)

        # Build adj_block mapping
        block_names = np.sort(ba['Exam Block'].unique())
        adjusted_block_dict = dict(
            zip(block_names, np.arange(1, len(block_names) + 1))
        )
        if 'Exam Block' in ba.columns:
            ba['adj_block'] = ba['Exam Block'].map(adjusted_block_dict)
        if 'slot' in ba.columns:
            ba['adj_block'] = ba['slot'].map(adjusted_block_dict)

        # Merge and group by student
        exam_df_with_blocks = exam_df.merge(
            ba, how='left', left_on='Exam Key', right_on='Exam Group'
        )
        by_student_block = (
            exam_df_with_blocks
            .groupby('anon-netid')['adj_block']
            .apply(list)
            .reset_index(name='blocks')
        )

        # Cache this config
        seen.loc[len(seen), 'configs'] = block_config
        seen.to_csv(seen_path, index=False)

        # Save the computed tables for next time
        ba.to_csv(os.path.join(SAVE_PATH, 'cache',f"{block_config}ba.csv"), index=False)
        by_student_block.to_csv(
            os.path.join(SAVE_PATH,'cache', f"{block_config}by_student_block.csv"),
            index=False
        )

    return ba, by_student_block
import os
import pandas as pd
from collections import defaultdict

def compute_co_enrollment(by_student_block: pd.DataFrame, block_config: str):
    """
    For each student’s list of blocks, count how many times each pair, triple,
    and quadruple appears across the student pool.
    Uses caching in SAVE_PATH/seen_co.csv; if you've already computed for
    this block_config, loads and returns the saved results.
    Returns three dicts: pairwise, triple, quadruple.
    """
    seen_path = os.path.join(SAVE_PATH,'cache', 'seen_co.csv')
    # 1) Load or initialize the cache of configs
    if os.path.exists(seen_path):
        seen = pd.read_csv(seen_path)
        # ensure we can append strings
        seen['configs'] = seen['configs'].astype(object)
    else:
        seen = pd.DataFrame({'configs': pd.Series(dtype=object)})
        seen.to_csv(seen_path, index=False)

    # 2) If we've already done this config, load saved CSVs
    if block_config in seen['configs'].values:
        # load pairwise
        pw_df = pd.read_csv(os.path.join(SAVE_PATH, 'cache',f"{block_config}_pairwise.csv"))
        pairwise = {
            (int(row['i']), int(row['j'])): int(row['count'])
            for _, row in pw_df.iterrows()
        }
        # load triple
        t3_df = pd.read_csv(os.path.join(SAVE_PATH,'cache',f"{block_config}_triple.csv"))
        triple = {
            (int(row['i']), int(row['j']), int(row['k'])): int(row['count'])
            for _, row in t3_df.iterrows()
        }
        # load quadruple
        t4_df = pd.read_csv(os.path.join(SAVE_PATH, 'cache',f"{block_config}_quadruple.csv"))
        quadruple = {
            (int(row['i']), int(row['j']), int(row['k']), int(row['l'])): int(row['count'])
            for _, row in t4_df.iterrows()
        }

    else:
        # compute fresh
        pairwise = defaultdict(int)
        triple = defaultdict(int)
        quadruple = defaultdict(int)

        for blocks in by_student_block['blocks']:
            for i in blocks:
                for j in blocks:
                    if i != j:
                        pairwise[(i, j)] += 1
                    for k in blocks:
                        if len({i, j, k}) == 3:
                            triple[(i, j, k)] += 1
                        for l in blocks:
                            if len({i, j, k, l}) == 4:
                                quadruple[(i, j, k, l)] += 1

        # 3a) Append this config to seen and save
        seen.loc[len(seen), 'configs'] = block_config
        seen.to_csv(seen_path, index=False)

        # 3b) Dump each dict to its own CSV
        pw_df = pd.DataFrame([
            {'i': i, 'j': j, 'count': cnt}
            for (i, j), cnt in pairwise.items()
        ])
        pw_df.to_csv(
            os.path.join(SAVE_PATH,'cache', f"{block_config}_pairwise.csv"),
            index=False
        )

        t3_df = pd.DataFrame([
            {'i': i, 'j': j, 'k': k, 'count': cnt}
            for (i, j, k), cnt in triple.items()
        ])
        t3_df.to_csv(
            os.path.join(SAVE_PATH, 'cache',f"{block_config}_triple.csv"),
            index=False
        )

        t4_df = pd.DataFrame([
            {'i': i, 'j': j, 'k': k, 'l': l, 'count': cnt}
            for (i, j, k, l), cnt in quadruple.items()
        ])
        t4_df.to_csv(
            os.path.join(SAVE_PATH,'cache', f"{block_config}_quadruple.csv"),
            index=False
        )

    return dict(pairwise), dict(triple), dict(quadruple)


def compute_student_unique(by_student_block: pd.DataFrame, num_slots: int, block_config: str):
    """
    1) Count how many students have exactly one block
       -> student_unique_block[i]
    2) Count how many students have exactly two blocks
       -> student_unique_block_pairs[(a, b)] and [(b, a)]
    Uses caching in SAVE_PATH/seen_unique.csv; if you've already computed for
    this block_config, loads and returns the saved results.
    Ensures every slot i and every pair (i, j) up to num_slots appears at least once (even if zero).
    Returns two dicts: student_unique_block, student_unique_block_pairs
    """
    seen_path = os.path.join(SAVE_PATH,'cache', 'seen_unique.csv')
    # 1) Load or initialize the cache of configs
    if os.path.exists(seen_path):
        seen = pd.read_csv(seen_path)  # one column: 'configs'
    else:
        seen = pd.DataFrame({'configs': []})
        seen.to_csv(seen_path, index=False)

    # 2) If we've already done this config, load saved CSVs
    if block_config in seen['configs'].values:
        # load single-slot counts
        ub_df = pd.read_csv(os.path.join(SAVE_PATH, 'cache', f"{block_config}_student_unique_block.csv"))
        print(
            'ub_df' , ub_df
        )

        up_df = pd.read_csv(os.path.join(SAVE_PATH,'cache',f"{block_config}_student_unique_block_pairs.csv"))
        student_unique_block        = ub_df.set_index('i')['count'].to_dict()
        student_unique_block_pairs  = {(i, j): c for i, j, c in
                               zip(up_df['i'], up_df['j'], up_df['count'])}
        #student_unique_block = (
        #    ub_df
        #    .set_index('i')['count']      # index=i, values=count
        #    .astype(int)                 # make sure they’re ints
        #    .to_dict()
        #)

        #student_unique_block_pairs = (
        #    up_df
        #    .set_index(['i', 'j'])['count']
        #    .astype(int)
        #    .to_dict()
        #)
    else:
        # compute fresh
        student_unique_block = defaultdict(int)
        student_unique_block_pairs = defaultdict(int)

        for _, row in by_student_block.iterrows():
            blocks = row["blocks"]
            if len(blocks) == 1:
                student_unique_block[blocks[0]] += 1
            elif len(blocks) == 2:
                a, b = blocks
                student_unique_block_pairs[(a, b)] += 1
                student_unique_block_pairs[(b, a)] += 1

        # Ensure keys exist for all slots/pairs
        for i in range(1, num_slots + 1):
            _ = student_unique_block[i]
            for j in range(1, num_slots + 1):
                _ = student_unique_block_pairs[(i, j)]

        # 3a) Append this config to seen and save
        seen.loc[len(seen), 'configs'] =block_config
        seen.to_csv(seen_path, index=False)

        # 3b) Dump each dict to its own CSV
        ub_out = pd.DataFrame([
            {'i': i, 'count': cnt}
            for i, cnt in student_unique_block.items()
        ])
        ub_out.to_csv(
            os.path.join(SAVE_PATH,'cache', f"{block_config}_student_unique_block.csv"),
            index=False
        )

        up_out = pd.DataFrame([
            {'i': i, 'j': j, 'count': cnt}
            for (i, j), cnt in student_unique_block_pairs.items()
        ])
        up_out.to_csv(
            os.path.join(SAVE_PATH,'cache', f"{block_config}_student_unique_block_pairs.csv"),
            index=False
        )

    return dict(student_unique_block), dict(student_unique_block_pairs)


def compute_slot_structures(slots: list, slots_per_day: int = 3):
    """
    Each pattern is defined on raw slot times (1-indexed). A “day” is any trio {d, d+1, d+2}
    where (d−1) % slots_per_day == 0 (i.e. d ∈ {1, 4, 7, 10, 13, 16, 19, 22,…}). 

    1) triple_in_day: any d ∈ slots such that d is a “start‐of‐day” ( (d−1)%3==0 )
       AND {d, d+1, d+2} ⊆ slots.

    2) triple_in_24hr: any s ∈ slots for which {s, s+1, s+2} ⊆ slots, but s is NOT a start‐of‐day.

    3) tripi_tropi = sorted(triple_in_day ∪ triple_in_24hr).

    4) eve_morn_start: any s ∈ slots with s%3==0 (end‐of‐day) AND (s+1) ∈ slots.

    5) other_b2b_start: any s ∈ slots with (s+1) ∈ slots AND s%3 ≠ 0.

    6) quad_start: any s ∈ slots with {s, s+1, s+2, s+3} ⊆ slots.

    7) quint_start: any s ∈ slots with {s, s+1, s+2, s+3, s+4} ⊆ slots.

    8) two_in_three_start: any s ∈ slots with {s, s+2} ⊆ slots.

    9) three_in_four_start: any s ∈ slots where, among the four consecutive times {s, s+1, s+2, s+3}, at least three of them lie in slots.

   10) four_in_five_start: any s ∈ slots where, among {s, s+1, s+2, s+3, s+4}, at least four lie in slots.

   11) triple_slots: all raw times d for which d+2 ≤ max_slot.  (We generate these simply as range(1, max_slot−1).)

   12) The Cartesian products (block_pair, block_slot, block_sequence_trip, block_sequence_quad, block_sequence_slot) are unchanged, just done over `slots`.

   13) next_slot: each slot → the next element in the sorted list (rolled around).

    Returns a dict containing all of these fields.
    """
    
    slots_sorted = sorted(slots)
    slots_set = set(slots_sorted)
    if not slots_sorted:
        return {}

    max_slot = slots_sorted[-1]

    # Helper: True iff s is the first slot of a “day” (1,4,7,10,…)
    def is_start_of_day(s: int) -> bool:
        return (s - 1) % slots_per_day == 0

    # 1) triple_slots = all d in [1..max_slot−2]
    triple_slots = list(range(1, max_slot - 1))

    # 2) triple_in_day
    triple_in_day = [
        s for s in slots_sorted
        if is_start_of_day(s) and {s, s + 1, s + 2}.issubset(slots_set)
    ]

    # 3) triple_in_24hr (must have s, s+1, s+2 all in slots, but s is NOT a start‐of‐day)
    triple_in_24hr = [
        s for s in slots_sorted
        if {s, s + 1, s + 2}.issubset(slots_set) and not is_start_of_day(s)
    ]

    # 4) tripi_tropi = union of the above two
    tripi_tropi = sorted(set(triple_in_day + triple_in_24hr))

    # 5) eve_morn_start: s%3==0 AND (s+1) in slots
    eve_morn_start = [
        s for s in slots_sorted
        if (s % slots_per_day == 0) and ((s + 1) in slots_set)
    ]

    # 6) other_b2b_start: (s+1) in slots AND s%3 != 0
    other_b2b_start = [
        s for s in slots_sorted
        if ((s + 1) in slots_set) and (s % slots_per_day != 0)
    ]

    # 7) quad_start: {s, s+1, s+2, s+3} ⊆ slots
    quad_start = [
        s for s in slots_sorted
        if {s, s + 1, s + 2, s + 3}.issubset(slots_set)
    ]

    # 8) quint_start: {s..s+4} ⊆ slots
    quint_start = [
        s for s in slots_sorted
        if {s, s + 1, s + 2, s + 3, s + 4}.issubset(slots_set)
    ]

    # 9) two_in_three_start: {s, s+2} ⊆ slots
    two_in_three_start = [
        s for s in slots_sorted
        if {s, s + 2}.issubset(slots_set)
    ]

    # 10) three_in_four_start: among {s, s+1, s+2, s+3}, at least 3 are in slots
    three_in_four_start = []
    for s in slots_sorted:
        window = [s, s+1, s+2, s+3]
        count = sum(1 for w in window if w in slots_set)
        if count >= 3:
            three_in_four_start.append(s)

    # 11) four_in_five_start: among {s..s+4}, at least 4 in slots
    four_in_five_start = []
    for s in slots_sorted:
        window = [s, s+1, s+2, s+3, s+4]
        count = sum(1 for w in window if w in slots_set)
        if count >= 4:
            four_in_five_start.append(s)

    # 12) next_slot mapping (roll the sorted list)
    shifted = np.roll(slots_sorted, -1)
    next_slot = {slots_sorted[i]: int(shifted[i]) for i in range(len(slots_sorted))}

    # 13) Cartesian‐product lists
    block_pair = [(i, j) for i in slots_sorted for j in slots_sorted]
    block_slot = [(i, s) for i in slots_sorted for s in slots_sorted]
    block_sequence_trip = [
        (i, j, k) for i in slots_sorted for j in slots_sorted for k in slots_sorted
    ]
    block_sequence_quad = [
        (i, j, k, l)
        for i in slots_sorted
        for j in slots_sorted
        for k in slots_sorted
        for l in slots_sorted
    ]
    block_sequence_slot = [
        (i, j, k, s)
        for i in slots_sorted
        for j in slots_sorted
        for k in slots_sorted
        for s in slots_sorted
    ]

    return {
        "slots": slots_sorted,
        "next_slot": next_slot,
        "block_pair": block_pair,
        "block_slot": block_slot,
        "block_sequence_trip": block_sequence_trip,
        "block_sequence_quad": block_sequence_quad,
        "block_sequence_slot": block_sequence_slot,
        "triple_slots": triple_slots,
        "triple_in_day": triple_in_day,
        "triple_in_24hr": triple_in_24hr,
        "tripi_tropi": tripi_tropi,
        "eve_morn_start": eve_morn_start,
        "other_b2b_start": other_b2b_start,
        "quad_start": quad_start,
        "quint_start": quint_start,
        "three_in_four_start": three_in_four_start,
        "four_in_five_start": four_in_five_start,
        "two_in_three_start": two_in_three_start,
    }

def compute_early_slot_penalties(num_slots: int = 24):
    """
    Assign penalty values to the last 3 slots of each day, next 3, and next 3:
      - For slots in [num_slots-3 .. num_slots-1], penalty = 3
      - For [num_slots-6 .. num_slots-4], penalty = 2
      - For [num_slots-9 .. num_slots-7], penalty = 1
      - Others = 0
    Returns:
      - first_list: length num_slots, first_list[j] is penalty for slot index j (0-based).
    """
    first_list = [0] * num_slots
    for j in range(num_slots - 3, num_slots):
        first_list[j] = 3
    for j in range(num_slots - 6, num_slots - 3):
        first_list[j] = 2
    for j in range(num_slots - 9, num_slots - 6):
        first_list[j] = 1
    return first_list


def build_global_sets(ba: pd.DataFrame,
                      exam_df: pd.DataFrame,
                      exam_sizes: pd.DataFrame = None,
                      num_slots: int = 24,
                      slots_per_day: int = 3):
    """
    Wrapper that stitches together all the above steps and returns exactly the same
    dictionary as before. The prints mark the “checkpoints” from the original.
    """
    print('BUILD GLOBAL ')
    # 1) Normalize and merge
    ba_adj, by_student_block = normalize_and_merge(ba, exam_df)
    print('here 0 ')

    # 2) Compute co-enrollment counts
    pairwise, triple, quadruple = compute_co_enrollment(by_student_block)
    print('here 1 ')

    # 3) Compute “unique block” counts
    student_unique_block, student_unique_block_pairs = compute_student_unique(
        by_student_block,
        num_slots=num_slots
    )
    print('here 2')

    # 4) Compute slot structures and special patterns
    slot_structures = compute_slot_structures(num_slots=num_slots, slots_per_day=slots_per_day)
    print('here 3')

    # 5) Compute early-slot penalties
    first_list = compute_early_slot_penalties(num_slots=num_slots)

    # Collect everything into one dictionary
    result = {
        **slot_structures,
        "first_list": first_list,
        "student_unique_block": student_unique_block,
        "student_unique_block_pairs": student_unique_block_pairs,
        "pairwise": pairwise,
        "triple": triple,
        "quadruple": quadruple,
        "block_assignment": ba_adj
    }
    return result
