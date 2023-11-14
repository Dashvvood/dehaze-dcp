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
from skimage.metrics import structural_similarity as ssim


def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    psnr = 10 * np.log10(1 / mse) 
    return psnr 

def _imread(path):
    I = plt.imread(path)
    if I.dtype == np.uint8:
        I = I / 255
    return I

def SSIM(A, B):
    ssim_r = ssim(A[:, :, 0], B[:, :, 0], data_range=1.0)
    ssim_g = ssim(A[:, :, 1], B[:, :, 1], data_range=1.0)
    ssim_b = ssim(A[:, :, 2], B[:, :, 2], data_range=1.0)
    return  (ssim_r + ssim_g + ssim_b) / 3

dt = datetime.now().strftime("%Y%m%d-%H%M%S")
input_root = r"D:\lab\dataset\IMA201\SOTS\output\task12"
output_root = r"D:\lab\dataset\IMA201\SOTS\output\task10"
if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
     

data = TinySOTS(root=r"D:\lab\dataset\IMA201\SOTS\outdoor")
print(len(data.index))


gt_hazy_psnr = {}
gt_J_psnr = {}
gt_hazy_ssim = {}
gt_J_ssim = {}

for prefix in tqdm(data.index):
    gt, hazy, *_ = data.get_by_prefix(prefix=prefix)
    tmp = os.path.join(input_root, prefix)
    J_path = os.path.join(tmp, "J.jpg")
    J = _imread(J_path)
    gt_hazy_psnr[prefix] = PSNR(gt, hazy)
    gt_J_psnr[prefix] = PSNR(gt, J)
    gt_hazy_ssim[prefix] = SSIM(gt, hazy)
    gt_J_ssim[prefix] = SSIM(gt, J)
    
output_path = os.path.join(output_root, f"{dt}_fast.json")
out = {
    "gt_hazy_psnr": gt_hazy_psnr,
    "gt_J_psnr": gt_J_psnr,
    "gt_hazy_ssim": gt_hazy_ssim,
    "gt_J_ssim": gt_J_ssim,
}

with open(output_path, 'w') as f:
    json.dump(out, f, indent=4)

print(f"gt_hazy_psnr: {sum(gt_hazy_psnr.values()) / len(gt_hazy_psnr)}")
print(f"gt_J_psnr: {sum(gt_J_psnr.values()) / len(gt_J_psnr)}")
print(f"gt_hazy_ssim: {sum(gt_hazy_ssim.values()) / len(gt_hazy_ssim)}")
print(f"gt_J_ssim: {sum(gt_J_ssim.values()) / len(gt_J_ssim)}")

