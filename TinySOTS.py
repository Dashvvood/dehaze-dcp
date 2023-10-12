import matplotlib.pyplot as plt
import os
from collections import OrderedDict, namedtuple
import re
import numpy as np

SOTS_DATA = namedtuple('data', ['gt', 'hazy'])

class TinySOTS(object):
    def __init__(self, root:str) -> None:
        self.root = root
        self.gt_dir = os.path.join(self.root, "gt")
        # self.gt_list = os.listdir(self.gt_dir)

        self.hazy_dir = os.path.join(self.root, "hazy")
        # self.hazy_list = os.listdir(self.hazy_dir)

        self.gt = OrderedDict()
        self.hazy = OrderedDict()
        self.index = []

        self._prepare()

    def _get_prefix(self, filename):
        pattern = r"(\d{4})"
        match = re.search(pattern, filename)
        return match.group(1)

    def _prepare(self):
        D1 = {}
        for name in os.listdir(self.gt_dir):
            pre = self._get_prefix(name)
            D1[pre] = os.path.join(self.gt_dir, name)
       
        D2 = {}
        for name in os.listdir(self.hazy_dir):
            pre = self._get_prefix(name)
            D2[pre] = os.path.join(self.hazy_dir, name)

        if D1.keys() != D2.keys():
            raise ValueError
        
        self.gt = OrderedDict(sorted(D1.items(), key=lambda s:s[0]))
        self.hazy = OrderedDict(sorted(D2.items(), key=lambda s:s[0]))
        self.index = list(self.gt.keys())
        return len(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        k = self.index[idx]
        gt = plt.imread(self.gt[k])
        if gt.dtype == np.uint8:
            gt = gt / 255
        hazy = plt.imread(self.hazy[k])
        if hazy.dtype == np.uint8:
            hazy = hazy / 255

        return SOTS_DATA(gt, hazy)
