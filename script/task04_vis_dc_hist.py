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

input_root = r"D:\lab\dataset\IMA201\SOTS\output\task01"
filelist = os.listdir(input_root)

def cumulative(hist):
    res = [0]
    for x in hist:
        res.append(res[-1]+x)
    return res

titles = [
    "SOTS-outdoor-gt",
    "SOTS-outdoor-hazy",
    "SOTS-indoor-gt",
    "SOTS-indoor-hazy",
    "O-HAZE-gt",
    "O-HAZE-hazy"
]


fig, axs = plt.subplots(3, 2, figsize=(8, 8), tight_layout=True, sharey=True)
axs_= axs.flatten()

for i, file in enumerate(filelist):
    filepath = os.path.join(input_root, file)
    d = json.load(open(filepath, 'r'))
    bins = np.array(d['bins'])
    hist = np.array(d['res'])
    hist = hist / np.sum(hist)
    cdf = cumulative(hist)
    axs_[i].bar(bins[:10] + 0.05, hist, width=0.085, facecolor=constant.COLOR_05)
    # ax2 = axs_[i].twinx()
    axs_[i].step(bins, cdf, c=constant.COLOR_04, linewidth=2)
    axs_[i].set_xticks(bins)
    axs_[i].set_title(titles[i])
    axs_[i].grid()
axs_[0].legend(["histogramme", "répartition"])

fig.savefig("./tmp.png")

