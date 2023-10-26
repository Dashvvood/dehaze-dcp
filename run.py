import argparse
from PIL import Image
from dcp import *
import matplotlib.pyplot as plt
import os
import re


opt_byte = {
"cmap": "gray",
"vmin": 0,
"vmax" : 1,
}

opt_rgb = {
"vmin": 0,
"vmax" : 1,
}


def _get_prefix(filename):
    pattern = r"(\d{4})"
    match = re.search(pattern, filename)
    return match.group(1)


def _imread_float(filepath):
    I = plt.imread(filepath)
    if I.dtype == np.uint8:
        I = I / 255
    return I

def run(input_dir,output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for i, filename in enumerate(os.listdir(input_dir)):
        prefix, ext = os.path.splitext(filename)
        filepath = os.path.join(input_dir, filename)
        I = _imread_float(filepath)
        
        dc = get_dark_channel(I, patch_size=(15, 15))
        mask = get_mask(dc, top_ratio=1e-3)
        A = get_atmos_light(I, dc, top_ratio=1e-3)
        tilde_t = get_tilde_t(I, A)

        t1 = soft_matting(I, p=tilde_t)
        t2 = guided_filter(I, p=tilde_t, ks=(5,5))

        J1 = get_J(I, A, t1)
        J2 = get_J(I, A, t2)
        D = get_depth(t1)

        plt.imsave(os.path.join(output_dir, prefix + "_I.png"), I, **opt_rgb)
        plt.imsave(os.path.join(output_dir, prefix + "_tilde_t.png"),tilde_t, **opt_byte)
        plt.imsave(os.path.join(output_dir, prefix + "_t1.png"), t1, **opt_byte)
        plt.imsave(os.path.join(output_dir, prefix + "_t2.png"), t2, **opt_byte)
        plt.imsave(os.path.join(output_dir, prefix + "_J1.png"), J1, **opt_rgb)
        plt.imsave(os.path.join(output_dir, prefix + "_J2.png"), J2, **opt_rgb)
        plt.imsave(os.path.join(output_dir, prefix + "_D.png"), D, cmap="hot")


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


