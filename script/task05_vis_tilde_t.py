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

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input_root',
    type=str,
    default=r"D:\lab\dataset\IMA201\SOTS\outdoor"
)

parser.add_argument(
    '-o', '--output_root',
    type=str,
    default=r"D:\lab\dataset\IMA201\SOTS\output\task05"
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

if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True) 
    
subset = json.load(open(args.subset, 'r'))['prefix']
data = TinySOTS(root=input_root)
p = (args.patch, args.patch)


for i in tqdm(range(len(subset))):
    prefix = subset[i]
    hazy = data.get_by_prefix(prefix=prefix).hazy
    dc = dcp.get_dark_channel(hazy, patch_size=(15,15))
    A = dcp.get_atmos_light(hazy, dc, top_ratio=1e-3)
    t_tilde = dcp.get_tilde_t(hazy, A)
    
    output_path = os.path.join(output_root, f"{prefix}_tilde_t.png")
    plt.imsave(output_path, t_tilde, **constant.gray_mode)