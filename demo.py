import argparse
from PIL import Image
from dcp import *
import matplotlib.pyplot as plt
import os
import re


def _get_prefix(filename):
    pattern = r"(\d{4})"
    match = re.search(pattern, filename)
    return match.group(1)



def run(input_dir,output_dir):
    for i, filename in enumerate(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, filename)
        prefix = _get_prefix(filename)
        I = plt.imread(filepath)
        I = np.array()
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="dcp",
        description="Using dcp to dehaze",
    )

    parser.add_argument(
        '-i', '--input_dir', type=str, 
        default=argparse.SUPPRESS,
        help="image path"
    )

    parser.add_argument(
        '-o', '--output_dir', type=str, 
        default=argparse.SUPPRESS,
        help="image path"
    )

    opts = parser.parse_args()
    
    run(opts.input_dir, opts.output_dir)


