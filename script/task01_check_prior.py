"""
Check the dark channel prior images under a folder
"""
import sys
sys.path.append("..")
from dehaze import dcp

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import time
import functools


BINS = np.linspace(0, 1, 11)

def __imread(path):
    I = plt.imread(path)
    if I.dtype == np.uint8:
        I = I / 255
    return I

elapsed_time = 0

def timer(func):
    # global elapsed_time
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        global elapsed_time
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer
    
@timer
def main(input_dir, output_dir):
    res = np.zeros((len(BINS) - 1))
    # for filename in os.listdir(input_dir)
    for filename in tqdm(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        I = __imread(filepath)
        dc = dcp.get_dark_channel(I, patch_size=(15, 15))
        hist, _ = np.histogram(dc.ravel(), bins=BINS)
    res += hist

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default="."
    )

    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default="."
    )

    args = parser.parse_args()


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")
  

    res = main(input_dir=args.input_dir, output_dir=args.output_dir)

    tmp = {
        "input": args.input_dir,
        "output": args.output_dir,
        "res": list(res),
        "bins": list(np.around(BINS, 2)),
        "time": elapsed_time
    }

    with open(os.path.join(args.output_dir, dt + ".json"), 'w') as f:
        json.dump(tmp, f, indent=4)


    

    
