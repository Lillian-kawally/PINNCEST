# [Ablation Version] All ablation switches added; default behavior unchanged.
import math
import torch
import torch.nn as nn
from utils.utils import mM2f
from models.bmsolverlayer import BMSolverLayer


class CNN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.feature_extractor(x)


class MLP(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(indim, outdim, bias=False),
            nn.LayerNorm(outdim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class Regressor(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=1, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        curr_dim = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.Dropout(dropout))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, b1_feats):
        query = b1_feats[-1]
        context = torch.cat(b1_feats, dim=1)  # [B, N_B1 * N_off, d_model]
        attn_out, _ = self.multihead_attn(query, context, context)
        x = self.norm(query + attn_out)
        x = self.norm_ffn(x + self.ffn(x))
        return x  # [B, N_off, d_model]


class PINNCEST(nn.Module):
    """
    Physics-informed Attention Network with ablation switches.

    Ablation flags (set in modelparams):
      use_cross_attn  : True  -> CrossAttention fusion across B1
                        False -> mean pooling across B1
      use_embeddings  : True  -> b1_emb prepended + pos_emb added
                        False -> no b1_emb, no pos_emb
      decouple_heads  : True  -> per-pool independent f/k Regressors
                        False -> single shared Regressor outputs 2*num_pools
      skip_bmlayer    : True  -> skip BM solver but keep physics injection
      single_b1_idx   : None  -> use all B1 (default)
                        int   -> use only b1_list[idx] (negative index allowed)
    """
    def __init__(self, modelparams, scanparams):
        super().__init__()
        self.scanparams = scanparams
        self.k_mode = modelparams.get('k_mode', 'log')
        self.b1_list = scanparams['b1']
        self.pool_names = list(scanparams['cest_pool'].keys())
        self.num_pools = len(self.pool_names)
        self.dropout = modelparams['dropout']

        d_model = modelparams.get('d_model', 128)
        nhead = modelparams.get('nhead', 8)
        num_offs = len(scanparams['offs'])

        # Ablation switches
        self.use_cross_attn = modelparams.get('use_cross_attn', True)
        self.use_embeddings = modelparams.get('use_embeddings', True)
        self.decouple_heads = modelparams.get('decouple_heads', True)
        self.use_physics = modelparams.get('use_physics', True)
        self.skip_bmlayer = modelparams.get('skip_bmlayer', False)

        # Single-B1 ablation: None = use all B1, int = use only this index
        raw_idx = modelparams.get('single_b1_idx', None)
        if raw_idx is None:
            self.single_b1_idx = None
        else:
            n_b1 = len(self.b1_list)
            self.single_b1_idx = raw_idx + n_b1 if raw_idx < 0 else raw_idx
            assert 0 <= self.single_b1_idx < n_b1, \
                f"single_b1_idx={raw_idx} out of range for b1_list of length {n_b1}"

        self.filter = CNN(d_model)

        if self.use_embeddings:
            self.b1_emb = nn.Embedding(len(self.b1_list), d_model)
            self.pos_emb = nn.Parameter(torch.randn(1, num_offs, d_model) * 0.02)

        self.extractor = nn.ModuleList([
            Attention(d_model, nhead) for _ in range(modelparams.get('num_layers', 4))
        ])

        # CrossAttention only needed when fusing multiple B1 features
        if self.use_cross_attn and self.single_b1_idx is None:
            self.fusion_layer = CrossAttention(d_model, nhead)

        self.pooling_weight = nn.Linear(d_model, 1)

        # Regression heads
        if self.decouple_heads:
            self.f_sub_heads = nn.ModuleList([
                Regressor(d_model // 2, d_model // 4,
                          num_layers=2, dropout=self.dropout)
                for _ in range(self.num_pools)
            ])
            self.k_sub_heads = nn.ModuleList([
                Regressor(d_model // 2, d_model // 4,
                          num_layers=3, dropout=self.dropout)
                for _ in range(self.num_pools)
            ])
        else:
            self.shared_head = Regressor(
                d_model // 2, d_model // 4,
                out_dim=2 * self.num_pools,
                num_layers=3, dropout=self.dropout
            )

        # Physics injection / BM solver
        if self.use_physics:
            self.extra_proj = nn.Sequential(
                MLP(5, d_model // 2),
                MLP(d_model // 2, d_model),
            )
            self.proj = nn.Sequential(
                MLP(d_model * 2, d_model),
                MLP(d_model, d_model // 2)
            )

            f_min, f_max, k_min, k_max = [], [], [], []
            for name in self.pool_names:
                pool = scanparams['cest_pool'][name]
                f_min.append(pool['frng'][0])
                f_max.append(pool['frng'][1])

                C = 10.0
                k0, k1 = pool['krng'][0], pool['krng'][1]
                if self.k_mode == 'loglog':
                    k_min.append(math.log10(math.log10(k0 + C)))
                    k_max.append(math.log10(math.log10(k1 + C)))
                elif self.k_mode == 'linear':
                    k_min.append(k0)
                    k_max.append(k1)
                else:
                    k_min.append(math.log10(k0 + C))
                    k_max.append(math.log10(k1 + C))

            self.register_buffer('f_min', torch.tensor(f_min).view(1, -1))
            self.register_buffer('f_max', torch.tensor(f_max).view(1, -1))
            self.register_buffer('k_min_val', torch.tensor(k_min).view(1, -1))
            self.register_buffer('k_max_val', torch.tensor(k_max).view(1, -1))
            self.bmlayer = BMSolverLayer(scanparams)
        else:
            self.proj = nn.Sequential(
                MLP(d_model, d_model),
                MLP(d_model, d_model // 2)
            )

    def forward(self, x, cestparams):
        batch_size = x.shape[0]
        b1_feats = []

        # Decide which B1 channels to use
        if self.single_b1_idx is not None:
            b1_indices = [self.single_b1_idx]
        else:
            b1_indices = list(range(len(self.b1_list)))

        for i in b1_indices:
            f = self.filter(x[:, i, :].unsqueeze(1)).transpose(1, 2)

            if self.use_embeddings:
                f = f + self.pos_emb
                emb = self.b1_emb(torch.full((batch_size, 1), i,
                                             dtype=torch.long, device=x.device))
                f = torch.cat([emb, f], dim=1)
                for layer in self.extractor:
                    f = layer(f)
                b1_feats.append(f[:, 1:, :])
            else:
                for layer in self.extractor:
                    f = layer(f)
                b1_feats.append(f)

        # Cross-B1 fusion
        if len(b1_feats) == 1:
            # Single-B1: nothing to fuse
            fused = b1_feats[0]
        elif self.use_cross_attn:
            fused = self.fusion_layer(b1_feats)
        else:
            stacked = torch.stack(b1_feats, dim=1)
            fused = stacked.mean(dim=1)

        weights = torch.softmax(self.pooling_weight(fused), dim=1)
        feat = torch.sum(weights * fused, dim=1)

        if self.use_physics:
            extra = torch.stack([
                cestparams['water_pool']['t1'],
                cestparams['water_pool']['t2'],
                cestparams['mt_pool']['f'],
                cestparams['b0shift'],
                cestparams['b1shift']
            ], 1).float()
            fusion_feat = torch.cat([feat, self.extra_proj(extra)], dim=1)
            feat = self.proj(fusion_feat)
        else:
            feat = self.proj(feat)

        # Heads
        if self.decouple_heads:
            f_pred = torch.cat([h(feat) for h in self.f_sub_heads], dim=1)
            k_pred = torch.cat([h(feat) for h in self.k_sub_heads], dim=1)
        else:
            out = self.shared_head(feat)
            out = out.view(batch_size, self.num_pools, 2)
            f_pred = out[:, :, 0]
            k_pred = out[:, :, 1]

        pred = torch.stack([f_pred, k_pred], dim=2).flatten(1)

        # BM solver
        if self.use_physics:
            f_val_mM = self.f_min + f_pred * (self.f_max - self.f_min)
            C = 10.0
            inner_val = self.k_min_val + k_pred * (self.k_max_val - self.k_min_val)
            if self.k_mode == 'loglog':
                k_val = torch.pow(10.0, torch.pow(10.0, inner_val)) - C
            elif self.k_mode == 'linear':
                k_val = inner_val
            else:
                k_val = torch.pow(10.0, inner_val) - C

            if not self.skip_bmlayer:
                for i, pool_name in enumerate(self.pool_names):
                    pool_dict = cestparams['cest_pool'][pool_name]
                    pool_dict['f'] = mM2f(f_val_mM[:, i])
                    pool_dict['k'] = k_val[:, i]
                zrec = self.bmlayer(cestparams)
            else:
                zrec = torch.zeros(
                    (batch_size, len(self.scanparams['offs'])),
                    device=x.device
                )
        else:
            zrec = torch.zeros(
                (batch_size, len(self.scanparams['offs'])),
                device=x.device
            )

        return pred, zrec