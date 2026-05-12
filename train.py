import argparse
import os
from datetime import datetime
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
from scipy.io import loadmat
from tensordict import TensorDict
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import models
from data import *
from utils.utils import *

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        current_epoch = epoch + 1
        if current_epoch <= warmup_epochs:
            return float(current_epoch) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main(args, override_params=None):
    # Seed handling: priority is override_params['seed'] > args.seed > default 42
    if override_params and 'seed' in override_params:
        seed = override_params['seed']
    elif hasattr(args, 'seed') and args.seed is not None:
        seed = args.seed
    else:
        seed = 42
    print(f"[seed] Using seed = {seed}")
    seed_everything(seed)

    with open(os.path.join(args.datapath, 'params.yaml'), 'r') as f:
        scanparams = yaml.load(f, yaml.SafeLoader)

    with open(os.path.join('params', args.model + '.yaml'), 'r') as f:
        modelparams = yaml.load(f, yaml.SafeLoader)
    modelparams['k_mode'] = args.k_mode
    modelparams['use_physics'] = args.use_physics

    if override_params:
        print(f"\n!!! Overriding model parameters: {override_params} !!!\n")
        modelparams.update(override_params)

    offs = loadmat(os.path.join(args.datapath, 'CEST_SOP_OFFS.mat'), simplify_cells=True)['offs']
    offs = offs[offs < 100]

    fit_range = (offs >= modelparams['fit_range'][0]) & \
                (offs <= modelparams['fit_range'][1]) & \
                (np.abs(offs) > 0.5)
    scanparams['offs'] = offs[fit_range]

    model = getattr(getattr(models, args.model), args.model)(modelparams, scanparams).cuda()
    total_params = count_parameters(model)
    print(f"\n{'=' * 30}")
    print(f"Model: {args.model}")
    print(f"Total Trainable Params: {total_params:,}")
    print(f"{'=' * 30}\n")

    full_dataset = dataset(datapath=args.datapath, mode='train', k_mode=args.k_mode)
    trainsize = int(0.9 * len(full_dataset))
    valsize = len(full_dataset) - trainsize

    # Use fixed generator seed for reproducible split
    trainset, valset = torch.utils.data.random_split(
        full_dataset, [trainsize, valsize],
        generator=torch.Generator().manual_seed(42)
    )

    trainloader = DataLoader(trainset, modelparams['batch_size'], shuffle=True)
    valloader = DataLoader(valset, modelparams['batch_size'], shuffle=False)

    print(f"Training samples: {len(trainset)}, Validation samples: {len(valset)}")
    print(f"Batch size: {modelparams['batch_size']}")

    epochs = modelparams['epochs']
    f_w = modelparams['f_w']
    k_w = modelparams['k_w']
    z_w = modelparams.get('z_loss_weight', 100) if modelparams.get('use_physics', True) else 0.0
    params = model.parameters()
    opt_name = args.opt.lower()
    if opt_name == 'adamw':
        optimizer = torch.optim.AdamW(
            params,
            lr=float(modelparams['lr']),
            weight_decay=float(modelparams['weight_decay'])
        )
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(
            params,
            lr=float(modelparams['lr']),
            momentum=float(modelparams.get('momentum', 0.9)),
            weight_decay=float(modelparams['weight_decay']))
    elif opt_name == 'adam':
        optimizer = torch.optim.Adam(
            params,
            lr=float(modelparams['lr']),
            weight_decay=float(modelparams['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    total_steps = epochs * len(trainloader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(modelparams['lr']),
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=10,
        cycle_momentum=False
    )

    criterion_l1 = nn.L1Loss()
    criterion_smo = nn.SmoothL1Loss(beta=0.1)

    start_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if override_params and 'trial_name' in override_params:
        start_time = override_params['trial_name']

    print(f"Training start: {start_time}")
    log_path = os.path.join('runs', args.model, start_time)
    print(f"Log path: {str(log_path)}")
    writer = SummaryWriter(log_dir=log_path)

    iter = 0
    pbar_epoch = tqdm(range(epochs), desc='Total Progress', unit='epoch')
    for epoch in pbar_epoch:
        model.train()
        train_epoch_total, train_epoch_k, train_epoch_f, train_epoch_z, train_epoch_fk = 0, 0, 0, 0, 0

        with tqdm(trainloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False) as pbar_batch:
            for zspec, param, label in pbar_batch:
                zspec, param, label = zspec.cuda(), TensorDict(param).cuda(), label.cuda()
                zspec = zspec[:, :, fit_range]

                noised_zspec = dynamic_awgn(zspec, snr_range=(40.0, 100.0))
                pred, zrec = model(noised_zspec, param)

                pred_f, pred_k = pred[:, 0::2], pred[:, 1::2]
                label_f, label_k = label[:, 0::2], label[:, 1::2]
                loss_k = criterion_smo(pred_k, label_k)
                loss_f = criterion_smo(pred_f, label_f)

                if args.use_physics:
                    loss_z = criterion_l1(zrec, zspec)
                else:
                    loss_z = torch.tensor(0.0).cuda()
                total_loss = f_w * loss_f + k_w * loss_k + z_w * loss_z

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                # Accumulate for epoch average
                train_epoch_total += total_loss.item()
                train_epoch_k += loss_k.item()
                train_epoch_f += loss_f.item()
                if args.use_physics:
                    train_epoch_z += loss_z.item()

                pbar_batch.set_postfix(L=f'{total_loss.item():.4f}')
                iter += 1

        model.eval()
        val_epoch_total, val_epoch_k, val_epoch_f, val_epoch_z, val_epoch_fk = 0, 0, 0, 0, 0
        with torch.no_grad():
            with tqdm(valloader, desc='Validation', leave=False) as pbar_val:
                for zspec, param, label in pbar_val:
                    zspec, param, label = zspec.cuda(), TensorDict(param).cuda(), label.cuda()
                    zspec = zspec[:, :, fit_range]

                    noised_zspec = awgn(zspec, snr=90.0)
                    pred, zrec = model(noised_zspec, param)

                    vk = criterion_smo(pred[:, 1::2], label[:, 1::2])
                    vf = criterion_smo(pred[:, 0::2], label[:, 0::2])

                    if args.use_physics:
                        vz = criterion_l1(zrec, zspec)
                    else:
                        vz = torch.tensor(0.0).cuda()
                    vt = f_w * vf + k_w * vk + z_w * vz

                    val_epoch_total += vt.item()
                    val_epoch_k += vk.item()
                    val_epoch_f += vf.item()
                    if args.use_physics:
                        val_epoch_z += vz.item()

        tr_n = len(trainloader)
        val_n = len(valloader)

        writer.add_scalars('Loss/Total', {'train': train_epoch_total / tr_n, 'val': val_epoch_total / val_n}, epoch)
        writer.add_scalars('Loss/loss_k', {'train': train_epoch_k / tr_n, 'val': val_epoch_k / val_n}, epoch)
        writer.add_scalars('Loss/loss_f', {'train': train_epoch_f / tr_n, 'val': val_epoch_f / val_n}, epoch)
        if args.use_physics:
            writer.add_scalars('Loss/loss_z', {'train': train_epoch_z / tr_n, 'val': val_epoch_z / val_n}, epoch)

        pbar_epoch.set_postfix(L=f'{total_loss.item():.4f}', Val_Z=f'{(val_epoch_z/val_n):.4f}')

        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/Epoch', current_lr, epoch)

        if modelparams.get('plot', False) and (epoch + 1) % 100 == 0:
            plt.plot(scanparams['offs'], noised_zspec[0][0].cpu(), '.')
            plt.plot(scanparams['offs'], zrec[0][0].detach().cpu(), '-')
            plt.show(block=False); plt.pause(3); plt.close()

        if (epoch + 1) % 50 == 0:
            save_path = os.path.join('checkpoints', args.model, start_time)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), os.path.join(save_path, f'ckp_{epoch + 1}.pth'))

    writer.close()
    save_path = os.path.join('checkpoints', args.model, start_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'final.pth'))
    print(f'Final model saved in {save_path}')
    print(f'training is over at {datetime.now()}')

    return val_epoch_f / val_n, val_epoch_k / val_n

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str)
    parser.add_argument('--k_mode', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_physics', action='store_false', dest='use_physics')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'sgd', 'adam'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)