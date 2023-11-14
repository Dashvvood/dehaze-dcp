import sys
sys.path.append("..")

from dehaze import (
    dcp,
    constant,
)
from dehaze.dataset import TinySOTS

import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import os
from itertools import chain
import argparse

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
input_root = r"D:\lab\dataset\IMA201\SOTS\output\task12"
output_root = r"D:\lab\dataset\IMA201\SOTS\output\task11"
if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
     

data = TinySOTS(root=r"D:\lab\dataset\IMA201\SOTS\outdoor")
print(len(data.index))

elapsed_time = {}

for prefix in tqdm(data.index):
    tmp = os.path.join(input_root, prefix)
    info_path = os.path.join(tmp, "info.json")
    with open(info_path, 'r') as f:
        time_used = json.load(f)['time']
    elapsed_time[prefix] = time_used
    
output_path = os.path.join(output_root, f"fast_time.json")
out = {
    "elapsed_time": elapsed_time
}

with open(output_path, 'w') as f:
    json.dump(out, f, indent=4)

print(f"Average: {sum(elapsed_time.values()) / len(elapsed_time)}")
