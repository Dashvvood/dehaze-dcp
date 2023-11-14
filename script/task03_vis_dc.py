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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input_root',
    type=str,
    default=r"D:\lab\dataset\IMA201\SOTS\outdoor"
)

parser.add_argument(
    '-o', '--output_root',
    type=str,
    default=r"D:\lab\dataset\IMA201\SOTS\output\task03"
)

parser.add_argument(
    '-s', '--subset',
    type=str,
    default="../data/sots_subset/subset01.json"
)

parser.add_argument(
    '-p', '--patch',
    type=int,
    default=15
)

args = parser.parse_args()

input_root = args.input_root
output_root = args.output_root
subset = json.load(open(args.subset, 'r'))['prefix']
data = TinySOTS(root=input_root)
p = (args.patch, args.patch)

for i in tqdm(range(len(subset))):
    prefix = subset[i]
    gt, hazy, *_ = data.get_by_prefix(prefix=prefix)
    dc = dcp.get_dark_channel(gt, patch_size=p)
    filepath = os.path.join(output_root, f"{prefix}_dc_{p[0]}_{p[1]}.png")
    plt.imsave(filepath, dc, **constant.gray_mode)

