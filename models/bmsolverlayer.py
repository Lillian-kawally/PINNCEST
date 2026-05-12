import torch
import torch.nn as nn
from einops import rearrange
from torch import tensor, zeros
from torch import sign, sin, cos, sqrt, exp, abs, sum, cat, pi
from torch.linalg import lstsq, matrix_exp

class BMSolverLayer(nn.Module):
    """
    This layer solves Bloch-McConnell equations to generate CEST Z-spectrum, supports cw saturation only
    Params:
        - scanparams : scanning parameters (i.e., b0, b1, ts)
    Input:
        - cestparams : paramters of cest pools (i.e., t1, t2, f, k, dw)
    """
    def __init__(self, scanparams):
        super().__init__()
        self.phi    = tensor(0.0)                       # reset phase of rf pulse to 0 each time
        self.pi     = tensor(pi)                        # everything should be wrapped to tensor
        self.gamma  = 267.5152538329683                 # gyro-magnetization ratio of proton [rad/s]
        
        self.xZspec = tensor(scanparams['offs'])        # frequency offset list
        self.b0 = tensor(scanparams['b0'])              # [T]
        self.b1 = tensor(scanparams['b1'])              # list, [uT]
        self.ts = tensor(scanparams['ts'])              # [s]

    def _SuperLorentzian(self, t2, w1, dw):
        """
        t2:  [noffs, nb1, batch]
        w1:  [noffs, nb1, batch]
        dw:  [noffs, nb1, batch]
        """
        du = 0.0001
        u = torch.arange(0, 1 + du, du, device=t2.device, dtype=t2.dtype)  # [n_u]
        # t2[:,:,:,None]  [noffs, nb1, batch, 1] & u[None,None,None,:] [1,1,1,n_u]
        t2_exp = t2.unsqueeze(-1)  # [noffs, nb1, batch, 1]
        dw_exp = dw.unsqueeze(-1)  # [noffs, nb1, batch, 1]
        u_exp = u.view(1, 1, 1, -1)  # [1, 1, 1, n_u]

        denom = torch.abs(3 * u_exp ** 2 - 1)
        denom = torch.clamp(denom, min=1e-12)

        integrand2 = torch.sqrt(2 / self.pi) * t2_exp / denom * torch.exp(-2 * (dw_exp * t2_exp / denom) ** 2)
        integral = torch.sum(integrand2, dim=-1) * du  # [noffs, nb1, batch]
        return w1 ** 2 * self.pi * integral
    
    def forward(self, cestparams):
        batchsize = len(cestparams['water_pool']['f'])
        device = cestparams['water_pool']['f'].device

        noffs = len(self.xZspec)
        nb1   = len(self.b1)

        dw = zeros([nb1, batchsize, noffs], device=device)
        dw[:, :, ...] = self.xZspec
        dw = dw.permute(2, 0, 1) + cestparams['b0shift'][None]

        b1 = zeros([noffs, batchsize, nb1], device=device)
        b1[:, :, ...] = self.b1
        b1 = b1.permute(0, 2, 1) * cestparams['b1shift']

        # nominate water pool as A
        fA  = 1.0
        R1A = zeros([noffs, nb1, batchsize], device=device)
        R2A = zeros([noffs, nb1, batchsize], device=device)
        R1A[:, :, ...] = 1 / cestparams['water_pool']['t1']
        R2A[:, :, ...] = 1 / cestparams['water_pool']['t2']

        # nominate cest pool as B
        B = cestparams['cest_pool']
        ncest = len(B.keys())
        fB, kB   = zeros([noffs, nb1, batchsize, ncest], device=device), zeros([noffs, nb1, batchsize, ncest], device=device)
        R1B, R2B = zeros([noffs, nb1, batchsize, ncest], device=device), zeros([noffs, nb1, batchsize, ncest], device=device)
        dwB = zeros([noffs, nb1, batchsize, ncest], device=device)

        fB[:, :, ...]  = torch.stack([B[key]['f'] for key in B.keys()],-1)
        R1B[:, :, ...] = 1 / torch.stack([B[key]['t1'] for key in B.keys()],-1)
        R2B[:, :, ...] = 1 / torch.stack([B[key]['t2'] for key in B.keys()],-1)
        dwB[:, :, ...] = torch.stack([B[key]['dw'] for key in B.keys()],-1)
        kB[:, :, ...]  = torch.stack([B[key]['k'] for key in B.keys()],-1)
        kAB = fB / fA * kB

        # nominate MT pool as C
        if 'mt_pool' in cestparams.keys():
            C = cestparams['mt_pool']
            fC, kC   = zeros([noffs, nb1, batchsize], device=device), zeros([noffs, nb1, batchsize], device=device)
            R1C, T2C = zeros([noffs, nb1, batchsize], device=device), zeros([noffs, nb1, batchsize], device=device)
            fC[:, :, ...]  = C['f']
            R1C[:, :, ...] = 1 / C['t1']
            T2C[:, :, ...] = C['t2']
            kC[:, :, ...]  = C['k']
            kAC = fC / fA * kC
            nmt = 1
        else:
            nmt = 0

        # include water pool for simplicity
        ncest = ncest + 1

        # build coefficient matrix A
        matsize = 3 * ncest + nmt
        A = zeros([noffs, nb1, batchsize, matsize, matsize], device=device)

        # fill in coefficient matrix - Mx
        A[:, :, :, 0, 0]           = - (R2A + sum(kAB, dim=-1))
        A[:, :, :, 0, 1:ncest]     = kB
        A[:, :, :, 0, ncest]       = - dw * self.gamma * self.b0 * sign(b1)
        A[:, :, :, 0, ncest*2]     = - b1 * self.gamma * sin(self.phi)

        for n in range(1,ncest):
            A[:, :, :, n, n]           = - (R2B[:,:,:,n-1] + kB[:,:,:,n-1])
            A[:, :, :, n, 0]           = kAB[:,:,:,n-1]
            A[:, :, :, n, n+ncest]     = - (dw - dwB[:,:,:,n-1]) * self.gamma * self.b0 * sign(b1)
            A[:, :, :, n, n+ncest*2]   = - b1 * self.gamma * sin(self.phi)

        # fill in coefficient matrix - My
        A[:, :, :, ncest, ncest]              = - (R2A + sum(kAB, dim=-1))
        A[:, :, :, ncest, 1+ncest:ncest*2]    = kB
        A[:, :, :, ncest, ncest*2]            = b1 * self.gamma * cos(self.phi)
        A[:, :, :, ncest, 0]                  = dw * self.gamma * self.b0 * sign(b1)
        for n in range(1,ncest):
            A[:, :, :, n+ncest, n+ncest]      = - (R2B[:,:,:,n-1] + kB[:,:,:,n-1])
            A[:, :, :, n+ncest, ncest]        = kAB[:,:,:,n-1]
            A[:, :, :, n+ncest, n+ncest*2]    = b1 * self.gamma * cos(self.phi)
            A[:, :, :, n+ncest, n]            = (dw - dwB[:,:,:,n-1]) * self.gamma * self.b0 * sign(b1)

        # fill in coefficient matrix - Mz
        if nmt != 0:
            A[:, :, :, ncest*2, ncest*2]                  = - (R1A + sum(cat([kAB, kAC.unsqueeze(-1)],-1),-1))
            A[:, :, :, ncest*2, 1+ncest*2:ncest*3+nmt]    = cat([kB, kC.unsqueeze(-1)],-1)
        else:
            A[:, :, :, ncest*2, ncest*2]                  = - (R1A + sum(kAB,-1))
            A[:, :, :, ncest*2, 1+ncest*2:ncest*3+nmt]    = kB
        A[:, :, :, ncest*2, ncest]                    = - b1 * self.gamma * cos(self.phi)
        A[:, :, :, ncest*2, 0]                        = b1 * self.gamma * sin(self.phi)
        for n in range(1,ncest):
            A[:, :, :, n+ncest*2, n+ncest*2]      = - (R1B[:,:,:,n-1] + kB[:,:,:,n-1])
            A[:, :, :, n+ncest*2, ncest*2]        = kAB[:,:,:,n-1]
            A[:, :, :, n+ncest*2, n+ncest]        = - b1 * self.gamma * cos(self.phi)
            A[:, :, :, n+ncest*2, n]              = b1 * self.gamma * sin(self.phi)
        if nmt != 0:
            rfC = self._SuperLorentzian(T2C, b1 * self.gamma, dw * self.gamma * self.b0)
            A[:, :, :, ncest*3, ncest*3]  = - (R1C + kC + rfC)
            A[:, :, :, ncest*3, ncest*2]  = kAC

        # build coefficient matrix b
        b = zeros([noffs, nb1, batchsize, ncest*3+nmt], device=device)
        b[:, :, :, ncest*2] = fA*R1A
        b[:, :, :, ncest*2+1:ncest*3] = fB*R1B
        if nmt != 0:
            b[:, :, :, -1] = fC*R1C
        b = b.unsqueeze(-1)

        # initial magnetization
        m0 = zeros([noffs, nb1, batchsize, ncest*3+nmt], device=device)
        m0[:, :, :, ncest*2] = fA
        m0[:, :, :, ncest*2+1:ncest*3] = fB
        if nmt != 0:
            m0[:, :, :, -1] = fC
        m0 = m0.unsqueeze(-1)

        # solve Bloch-McConnell equations - multi shot
        m = []
        for i in range(nb1):
            Ai = A[:,i,...]
            bi = b[:,i,...]
            m0i = m0[:,i,...]
            Ai  = rearrange(Ai, 'b c h w -> (b c) h w')
            bi  = rearrange(bi, 'b c h w -> (b c) h w')
            m0i = rearrange(m0i, 'b c h w -> (b c) h w')
            Ainvb = lstsq(Ai, bi).solution
            ex = matrix_exp(Ai * self.ts)
            mi = (ex @ (m0i + Ainvb) - Ainvb)[:,ncest*2,0]
            mi = rearrange(mi, '(b c) -> b c', b = noffs).T
            m.append(mi)
        m = torch.stack(m,1)

        # m is [batchsize, nb1, noffs]
        return m