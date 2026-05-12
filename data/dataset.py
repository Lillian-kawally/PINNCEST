import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils.utils import f2mM

class dataset(Dataset):
    """CEST MRI z-spectra dataset"""
    def __init__(self, datapath, mode='train', k_mode='log', debug=False, debug_size=None):
        super().__init__()
        self.mode = mode
        self.k_mode = k_mode

        data_dir = os.path.join(datapath, mode)
        self.zspecs = loadmat(os.path.join(data_dir, 'zspecs.mat'),
                              simplify_cells=True)['zspecs']
        self.zparams = loadmat(os.path.join(data_dir, 'zparams.mat'),
                               simplify_cells=True)['zparams']

        for i in range(len(self.zparams)):
            if 'mt_pool' in self.zparams[i]:
                del self.zparams[i]['mt_pool']['lineshape']

        if debug:
            total = len(self.zparams)
            sample_size = debug_size if debug_size else total // 4
            indices = np.random.choice(total, sample_size, replace=False)
            self.zspecs = self.zspecs[indices]
            self.zparams = [self.zparams[i] for i in indices]

    def __len__(self):
        return len(self.zparams)

    def __getitem__(self, idx):
        zspec = torch.from_numpy(self.zspecs[idx]).float()
        param = self.zparams[idx]

        label = []
        for name in param['cest_pool'].keys():
            # f: concentration
            f = f2mM(param['cest_pool'][name]['f'])
            frng = param['cest_pool'][name]['frng']
            f_norm = (f - frng[0]) / (frng[1] - frng[0])

            # k: exchange rate (log scale)
            k = param['cest_pool'][name]['k']
            krng = param['cest_pool'][name]['krng']

            C = 10.0
            if self.k_mode == 'loglog':
                # log10(log10(k))
                k_loglog = np.log10(np.log10(k + C))
                k_min_loglog = np.log10(np.log10(krng[0] + C))
                k_max_loglog = np.log10(np.log10(krng[1] + C))
                k_norm = (k_loglog - k_min_loglog) / (k_max_loglog - k_min_loglog)
            elif self.k_mode == 'linear':
                k_norm = (k - krng[0]) / (krng[1] - krng[0])
            else:
                k_min_l = np.log10(krng[0] + C)
                k_max_l = np.log10(krng[1] + C)
                k_norm = (np.log10(k + C) - k_min_l) / (k_max_l - k_min_l)

            label.extend([f_norm, k_norm])
        return zspec, param, torch.tensor(label)