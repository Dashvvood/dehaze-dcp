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
    default=r"D:\lab\dataset\IMA201\SOTS\output\task07"
)

parser.add_argument(
    '-s', '--subset',
    type=str,
    default="../data/sots_subset/subset02.json"
)

parser.add_argument(
    '-p', '--patch',
    type=int,
    default=15
)

parser.add_argument(
    '--lam',
    type=float,
    default=1e-4
)

args = parser.parse_args()

input_root = args.input_root
output_root = args.output_root

if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True) 
    
subset = json.load(open(args.subset, 'r'))['prefix']
data = TinySOTS(root=input_root)
p = (args.patch, args.patch)
lam = args.lam


for i in tqdm(range(len(subset))):
    prefix = subset[i]
    gt, hazy, *_ = data.get_by_prefix(prefix=prefix)
    I = hazy
    dc = dcp.get_dark_channel(I)
    A = dcp.get_atmos_light(I, dc)
    tilde_t = dcp.get_tilde_t(I, A)
    t = dcp.soft_matting(I, p=tilde_t, lam=lam)
    
    filepath = os.path.join(output_root, f"{prefix}_t_{lam}.png")

    plt.imsave(filepath, t, **constant.gray_mode)

