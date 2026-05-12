# define utility functions
import numpy as np
import torch
import scipy.io as spio
from torch.optim.lr_scheduler import LambdaLR

def f2mM(f, n=1):
    m = f * 1000 * 111 / n
    return m

def mM2f(m, n=1):
    f = n * m / 1000 / 111
    return f

def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = torch.sum(x**2, dim=-1, keepdim=True) / x.shape[-1]
    npower = xpower / snr
    noise = torch.randn_like(x) * torch.sqrt(npower)
    return x + noise


def dynamic_awgn(x, snr_range=(30.0, 100.0)):
    batch_size = x.shape[0]
    snr_db = torch.empty((batch_size, 1, 1), device=x.device).uniform_(*snr_range)
    snr = 10 ** (snr_db / 10.0)

    xpower = torch.sum(x ** 2, dim=-1, keepdim=True) / x.shape[-1]
    npower = xpower / snr
    noise = torch.randn_like(x) * torch.sqrt(npower)

    return x + noise

def add_rician(zspec, std=2e-3):
    rdm1 = torch.randn(zspec.shape).to(zspec.device)
    rdm2 = torch.randn(zspec.shape).to(zspec.device)
    
    x = std * rdm1 + zspec
    y = std * rdm2

    module = torch.sqrt(torch.pow(x,2) + torch.pow(y,2))
    return module


def refactor_zparams(zparams):
    # [x, y, n_b1]
    zparams_tmp = np.array(zparams, dtype=object).transpose(1, 2, 0)
    xn, yn, _ = zparams_tmp.shape
    zparams_refined = np.empty((xn, yn), dtype=object)

    for r in range(xn):
        for c in range(yn):
            p_list = zparams_tmp[r, c]
            if not isinstance(p_list[0], dict) or not p_list[0]:
                zparams_refined[r, c] = {}
                continue
            base_p = p_list[0].copy()
            base_p['b1'] = np.array([p['b1'] for p in p_list], dtype=np.float64)

            zparams_refined[r, c] = base_p

    return zparams_refined

class PolyScheduler(LambdaLR):
    def __init__(self, optimizer, t_total, exponent=0.9, last_epoch=-1):
        self.t_total = t_total
        self.exponent = exponent
        super(PolyScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return (1 - step / self.t_total)**self.exponent

def check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    assert len(dict.shape) == 2, 'only 2d array is supported'
    xn, yn = dict.shape
    for x in range(xn):
        for y in range(yn):
            if isinstance(dict[x,y], spio.matlab._mio5_params.mat_struct):
                dict[x,y] = _todict(dict[x,y])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    by mergen, https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab._mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict