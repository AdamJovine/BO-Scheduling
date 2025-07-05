import numpy as np
import pandas as pd
from datetime import datetime

SEMESTER = "sp25"
NUM_SLOTS = 24
FIRST_LIST = [0] * 15 + [1, 1, 1, 2, 2, 2, 3, 3, 3]
ASSIGNMENT_TYPE = "block"
OPTIM_MODE = "carbo"
POST_PROCESSING = True
EMPTY_BLOCKS = []
NORM = True
run_name = "thomp:"
print("run_name: ", run_name)
print("ASSIGNMENT_TYPE: ", ASSIGNMENT_TYPE)
print("semester , ", SEMESTER)
GLOBAL_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DATA_PATH = "/home/asj53/BOScheduling/data/" + SEMESTER
SAVE_PATH = "/home/asj53/BOScheduling/results/" + SEMESTER
UI_PATH = "/home/asj53/BOScheduling/UI/metrics/plots/"
INDEX_WEIGHTS = {
    0: 5,
}
"""
Maximum values for each column:
conflicts                           4.000000
quints                              4.000000
quads                              26.000000
four in five slots                 21.000000
triple in 24h (no gaps)            99.000000
triple in same day (no gaps)      135.000000
three in four slots               575.000000
evening/morning b2b               999.000000
other b2b                        1995.000000
two in three slots               4994.000000
singular late exam               1421.000000
two exams, large gap             1080.000000
avg_max                            29.987783
lateness                        57474.000000
"""
REF_POINT = (-250, -600, -3000, -5000, -1500, -1200, -20, -60000)

BA_TIME = 7200
SEQ_TIME = 300
PP_TIME = 360
BLOCK_BOUNDS = {"size_cutoff": [200, 300], "reserved": [0, 7], "num_blocks": [20, 24]}

SEQ_BOUNDS = {
    "alpha": [10, 100],  # alpha
    "gamma": [5, 50],  # gamma
    "delta": [0, 10],  # delta
    "vega": [0, 10],  # vega
    "theta": [0, 10],  # theta
    "large_block_size": [1000, 2000],  # large_block_size
    "large_exam_weight": [0, 100],  # large_exam_weight
    "large_block_weight": [0, 100],  # large_block_weight
    "large_size_1": [200, 400],  # large_size_1
    "large_cutoff_freedom": [0, 4],
}  # large_size_2}}

PP_BOUNDS = {"tradeoff": [0, 1], "flpens": [0, 1]}

# {
#    "WLSACCESSID": "0fb92c9f-9175-4f1c-a107-ab835fc599b7",  # Josh
#    "WLSSECRET": "680f2cd4-fbe5-432b-86b5-e8b14e8c73ef",
#    "LICENSEID": 2554057
# },
PARAM_NAMES = (
    list(BLOCK_BOUNDS.keys()) + list(SEQ_BOUNDS.keys()) + list(PP_BOUNDS.keys())
)
LICENSES = [
    {
        "WLSACCESSID": "0fb92c9f-9175-4f1c-a107-ab835fc599b7",  # Josh
        "WLSSECRET": "680f2cd4-fbe5-432b-86b5-e8b14e8c73ef",
        "LICENSEID": 2554057,
    },
    {
        "WLSACCESSID": "7820242a-1059-4e41-be5e-249bf3b03c9f",  # JK
        "WLSSECRET": "1b840509-d813-4b67-a764-fbb5b30fa693",
        "LICENSEID": 2471364,
    },
    # {
    #    "WLSACCESSID":"99a11dbd-9220-4ba7-a91b-8b476ab9d37f", # Tajesh
    #    "WLSSECRET":"abe10b17-fc7f-4536-aef6-6284d5e1a9e1",
    #    "LICENSEID":2616425
    # },
    {
        "WLSACCESSID": "17b613a5-d70f-4885-94aa-c54cf45cf07a",  # Caleb
        "WLSSECRET": "d053d738-acec-41e3-bd66-bbc460d6b31e",
        "LICENSEID": 2654947,
    },
    # {
    #    "WLSACCESSID": "94f291a0-38b1-475c-b39f-c50e2ef84f39",  # Selina
    #    "WLSSECRET": "3365e397-bda9-4721-ba58-acd6ea9dfc23",
    #    "LICENSEID": 2409153
    # },
    # {
    #    "WLSACCESSID": "2954ef62-8fa8-4f22-a875-c2ebc1136e22",  # Adam
    #    "WLSSECRET": "08ab134c-9a73-4212-a3c0-d625c9f8a662",
    #   "LICENSEID": 931025
    # },
    {
        "WLSACCESSID": "03b0faba-5a45-4416-90e2-a7421e127ffa",  # Hedy
        "WLSSECRET": "04a2c9d8-04ae-476a-ac16-6a404d44e992",
        "LICENSEID": 2618329,
    },
]

from datetime import datetime
import hashlib

import json

_get_name_call_count = 0


def get_name(param_dict, optim_mode=OPTIM_MODE, global_ts=None, block_assignment=None):
    if global_ts is None:
        global_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    global _get_name_call_count

    # Custom JSON serializer to handle non-serializable types like numpy.int64
    def json_serializer(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        # Add other types here if you encounter more serialization issues
        # For example, if you have numpy floats:
        # if isinstance(obj, (np.float64, np.float32)):
        #     return float(obj)
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    # pack everything into a canonical JSON string
    payload = {
        "mode": optim_mode,
        "block": block_assignment or "",
        "params": param_dict,
    }
    # Use the custom serializer for json.dumps
    data = json.dumps(payload, sort_keys=True, default=json_serializer).encode("utf-8")

    _get_name_call_count += 1
    name = hashlib.md5(data).hexdigest()
    # take the MD5 (you can swap in sha1/sha256 if you like)
    return global_ts + "i" + str(_get_name_call_count) + run_name + "-" + str(name)


def load_exam_data(semester):
    exam_df = pd.read_csv(f"{DATA_PATH}/exam_df.csv", low_memory=False, dtype=str)
    exam_sizes = pd.read_csv(f"{DATA_PATH}/exam_sizes.csv")
    return exam_df, exam_sizes
