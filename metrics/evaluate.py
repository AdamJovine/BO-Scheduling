import pandas as pd
import numpy as np
from config.settings import (
    PARAM_NAMES,
    SAVE_PATH,
    BLOCK_BOUNDS,
    SEQ_BOUNDS,
    PP_BOUNDS,
    DATA_PATH,
)
from sqlalchemy import create_engine, text


# srom myproject.data.db_accessor import add_metric
def compute_average_max_slot(by_student_block):
    """Calculate the average maximum slot per student."""
    max_values = by_student_block["slots"].apply(lambda x: max(x) if x else np.nan)
    return max_values.mean()


def compute_lateness(schedule_df, exam_sizes):
    """Calculate the lateness metric."""
    # Merge schedule with exam sizes
    # schedule_df = schedule_df.merge(
    #    exam_sizes[['exam', 'size']],
    #    how='left',
    #    left_on='Exam Group',
    #    right_on='exam'
    # )
    # Calculate lateness
    schedule_df["lateness"] = schedule_df["slot"] - 15
    schedule_df["weighted"] = schedule_df["lateness"] * schedule_df["size"].apply(
        lambda x: 0 if x < 100 else x
    )
    schedule_df["weighted"] = schedule_df["weighted"].apply(lambda x: 0 if x < 0 else x)

    return schedule_df["weighted"].sum(), schedule_df


def compute_conflicts(by_student_block):
    """Calculate the number of exam conflicts."""
    conflicts = 0
    # Copy to avoid modifying the original
    blocks_copy = by_student_block.copy()

    for s in range(len(blocks_copy)):
        for b in range(len(blocks_copy["slots"][s])):
            if (
                blocks_copy["slots"][s][b] != -1
                and blocks_copy["slots"][s].count(blocks_copy["slots"][s][b]) > 1
            ):
                conflicts += (
                    blocks_copy["slots"][s].count(blocks_copy["slots"][s][b]) - 1
                )
                # Mark this block as counted
                blocks_copy["slots"][s][b] = -1

    return conflicts


def compute_pattern_count(by_student_block, pattern_starts, pattern_length):
    """Generic function to count occurrences of exam patterns."""
    pattern_count = 0
    blocks_copy = by_student_block.copy()

    patterns = [[i] + [i + j for j in range(1, pattern_length)] for i in pattern_starts]

    for s in range(len(blocks_copy)):
        for p in patterns:
            if all(b in blocks_copy["slots"][s] for b in p):
                # Remove the blocks from consideration for future patterns
                for b in p:
                    blocks_copy["slots"][s].remove(b)
                pattern_count += 1

    return pattern_count, blocks_copy


def compute_density_count(by_student_block, window_starts, window_length, density):
    """Count occurrences where at least 'density' exams occur within a window."""
    count = 0
    blocks_copy = by_student_block.copy()

    windows = [[i] + [i + j for j in range(1, window_length)] for i in window_starts]

    for s in range(len(blocks_copy)):
        for w in windows:
            if sum(b in blocks_copy["slots"][s] for b in w) >= density:
                # Remove the blocks that are part of this window
                for b in w:
                    if b in blocks_copy["slots"][s]:
                        blocks_copy["slots"][s].remove(b)
                count += 1

    return count, blocks_copy


def compute_two_exams_large_gap(by_student_block, gap_threshold=15):
    """Count students with only two exams that have a large gap between them."""
    count = 0

    for s in range(len(by_student_block)):
        if len(by_student_block["slots"][s]) == 2:
            mini = min(by_student_block["slots"][s])
            maxi = max(by_student_block["slots"][s])
            if maxi - mini > gap_threshold:
                count += 1

    return count


def compute_late_singular_exam(by_student_block, late_threshold=18):
    """Count students with only one exam that is scheduled late."""
    count = 0

    for s in range(len(by_student_block)):
        if len(by_student_block["slots"][s]) == 1:
            if by_student_block["slots"][s][0] > late_threshold:
                count += 1

    return count


def evaluate_schedule(
    schedule, exam_sizes, params, global_sets, sched_name, slots_per_day=3
):
    """
    Evaluates an exam schedule for conflicts, lateness, and student burden.

    Args:
        schedule_dict (dict): Mapping of exam blocks to slots.
        exam_df (DataFrame): Full student enrollment data with ['anon-netid', 'Exam Key'].
        exam_sizes (DataFrame): Contains ['exam', 'size'].
        param_dict (dict): Parameters including global_sets.
        slots_per_day (int): Number of exam slots per day (e.g., 3).

    Returns:
        metrics_df (DataFrame): A dataframe of key evaluation metrics.
    """
    # print('PARAM_DICT IN E_S: ' , params)
    # print('schedule : ' , schedule)
    # print('block_assignment : ' , block_assignment)
    schedule = schedule.loc[:, ~schedule.columns.str.contains("Unnamed")]
    enrl_df = pd.read_csv(DATA_PATH + "/enrl.csv")
    enrl_df = enrl_df.merge(schedule, left_on="Exam Key", right_on="Exam Group")
    print("enrl_df : ", enrl_df)
    # Create by_student_block dataframe
    by_student_block = (
        enrl_df.groupby("anon-netid")["slot"].apply(list).reset_index(name="slots")
    )  # create_by_student_slot_df(exam_df, schedule_dict)
    print(" by_student_block : ", by_student_block)
    # Calculate average maximum slot
    average_max = compute_average_max_slot(by_student_block)
    print("The average maximum is:", average_max)

    # Calculate lateness
    lateness, schedule_df_updated = compute_lateness(schedule, exam_sizes)
    print("lateness:", lateness)

    # Calculate conflicts
    conflicts = compute_conflicts(by_student_block)
    print("conflicts:", conflicts)

    # Calculate quints
    quint_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["quint_start"], 5
    )
    print("quints:", quint_count)

    # Calculate quads
    quad_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["quad_start"], 4
    )
    print("quads:", quad_count)

    # Calculate triple in 24 hours
    triple_24_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["triple_in_24hr"], 3
    )
    print("triple in 24h (no gaps):", triple_24_count)

    # Calculate triple in same day
    triple_day_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["triple_in_day"], 3
    )
    print("triple in same day (no gaps):", triple_day_count)

    # Calculate four in five slots
    four_in_five_count, by_student_block = compute_density_count(
        by_student_block, global_sets["quint_start"], 5, 4
    )

    # Add pattern-based four in five
    extra_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["four_in_five_start"], 4
    )
    four_in_five_count += extra_count
    print("four in five slots:", four_in_five_count)

    # Calculate three in four slots
    three_in_four_count, by_student_block = compute_density_count(
        by_student_block, global_sets["quad_start"], 4, 3
    )

    # Add pattern-based three in four
    extra_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["three_in_four_start"], 3
    )
    three_in_four_count += extra_count
    print("three in four slots:", three_in_four_count)

    # Calculate evening/morning back-to-back
    eve_morn_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["eve_morn_start"], 2
    )
    print("evening/morning b2b:", eve_morn_count)

    # Calculate other back-to-backs
    other_b2b_count, by_student_block = compute_pattern_count(
        by_student_block, global_sets["other_b2b_start"], 2
    )
    print("other b2b:", other_b2b_count)

    # Calculate two in three slots
    two_in_three_count = 0

    for i, slot_list in enumerate(by_student_block["slots"]):
        for start in global_sets["two_in_three_start"]:
            # both start and start+2 must still be in the list
            if start in slot_list and (start + 2) in slot_list:
                # remove by value (order doesn’t matter with remove())
                slot_list.remove(start)
                slot_list.remove(start + 2)
                two_in_three_count += 1

    print("two in three slots:", two_in_three_count)

    # Calculate singular late exam
    late_exam_count = compute_late_singular_exam(by_student_block)
    print("singular late exam count:", late_exam_count)

    # Calculate two exams with large gap
    two_exams_large_gap = compute_two_exams_large_gap(by_student_block)
    print("two exams, large gap:", two_exams_large_gap)

    # Create metrics dataframe
    resul = {
        "conflicts": conflicts,
        "quints": quint_count,
        "quads": quad_count,
        "four in five slots": four_in_five_count,
        "triple in 24h (no gaps)": triple_24_count,
        "triple in same day (no gaps)": triple_day_count,
        "three in four slots": three_in_four_count,
        "evening/morning b2b": eve_morn_count,
        "other b2b": other_b2b_count,
        "two in three slots": two_in_three_count,
        "singular late exam": late_exam_count,
        "two exams, large gap": two_exams_large_gap,
        "avg_max": average_max,
        "lateness": lateness,
    }

    # 2) Now `params` is a list/1D‐tensor of the same length as param_names,
    #    so you can either zip into a dict:
    # metrics = {}
    for name, val in zip(PARAM_NAMES, params):
        resul[name] = float(val)
    met = pd.DataFrame(resul, index=[0])
    met.to_csv(f"{SAVE_PATH}/metrics/{sched_name}.csv", index=False)
    # add_metric(met, sched_name)
    print("met! ", met)
    return met
