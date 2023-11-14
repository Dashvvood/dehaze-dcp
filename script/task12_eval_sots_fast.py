import sys
sys.path.append("..")
from dehaze import (
    dcp,
    dataset,
    constant
)

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
import time
import functools
from itertools import product

import IPython

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


def main(sots_root, output_dir):
    data = dataset.TinySOTS(sots_root)
    params = {
        'method': "fast",
        'patch_size': (15,15),
        'top_ratio': 1e-3,
        'eps': 1e-3,
        'ks': (41, 41),
        's': 4
    }
    
    for i in tqdm(range(len(data))):
    # for i in tqdm(range(1)):
        X = data[i]
        prefix, _ = os.path.splitext(os.path.basename(X.path))
        
        tic = time.perf_counter()
        res = dcp.dehaze_image(X.hazy, **params)
        toc = time.perf_counter()

        tmp_dir = os.path.join(output_dir, prefix)
        os.makedirs(tmp_dir, exist_ok=True)
        with open(os.path.join(tmp_dir, "info.json"), 'w') as f:
            info = {
                'params': params,
                'path': X.path,
                'prefix': prefix,
                'A': list(res.A),
                'time': toc-tic
            }
            json.dump(info, f, indent=4)

        d = res._asdict()
        d.pop('A')
        for k, v in d.items():
            if len(v.shape) == 3 and v.shape[-1] == 3:
                plt.imsave(os.path.join(tmp_dir, f"{k}.jpg"), v, **constant.rgb_mode)
            else:
                plt.imsave(os.path.join(tmp_dir, f"{k}.jpg"), v, **constant.gray_mode)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir',
        type=str,
        default=r"D:\lab\dataset\IMA201\SOTS\outdoor"
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        default=r"D:\lab\dataset\IMA201\SOTS\output\task12"
    )

    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    dt = datetime.now().strftime("%Y%m%d-%H%M%S")

    res = main(sots_root=args.input_dir, output_dir=args.output_dir)



