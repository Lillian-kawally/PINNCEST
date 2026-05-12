import os
import yaml
import models
import argparse
import random
from utils.utils import *
from tensordict import TensorDict
from scipy.io import loadmat, savemat
from tqdm import tqdm
import evaluate


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, override_params=None):
    TEST_SAMPLING_SEED = 42
    seed_everything(TEST_SAMPLING_SEED)

    trial_name = override_params.get('trial_name') if override_params else None
    noise_seed = override_params.get('noise_seed') if override_params else None

    with open(os.path.join(args.datapath, 'params.yaml'), 'r') as f:
        scanparams = yaml.load(f, yaml.SafeLoader)
    with open(os.path.join('params', args.model + '.yaml'), 'r') as f:
        modelparams = yaml.load(f, yaml.SafeLoader)
    modelparams['k_mode'] = args.k_mode
    modelparams['use_physics'] = args.use_physics

    if override_params:
        print(f"\n!!! Overriding model parameters: {override_params} !!!")
        modelparams.update(override_params)

    offs = loadmat(os.path.join(args.datapath, 'CEST_SOP_OFFS.mat'),
                   simplify_cells=True)['offs']
    offs = offs[offs < 100]
    fit_range_mask = (offs >= modelparams['fit_range'][0]) & \
                     (offs <= modelparams['fit_range'][1]) & \
                     (np.abs(offs) > 0.4)
    scanparams['offs'] = offs[fit_range_mask]

    # Build model with the SAME modelparams that training used
    model_fn = getattr(getattr(models, args.model), args.model)
    model = model_fn(modelparams, scanparams).cuda()
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model.eval()

    zspecs = loadmat(os.path.join(args.datapath, 'test', 'zspecs.mat'),
                     simplify_cells=True)['zspecs']
    zparams = loadmat(os.path.join(args.datapath, 'test', 'zparams.mat'),
                      simplify_cells=True)['zparams']
    total_samples = len(zspecs)
    sample_size = total_samples // 10

    rng = np.random.RandomState(TEST_SAMPLING_SEED)
    indices = np.sort(rng.choice(total_samples, sample_size, replace=False))
    print(f"Reproducibly selecting {sample_size} samples (out of {total_samples})")

    zspecs = zspecs[indices]
    zparams = [zparams[i] for i in indices]

    if noise_seed is not None:
        torch.manual_seed(int(noise_seed))
        np.random.seed(int(noise_seed))

    Result = []
    pbar = tqdm(range(len(zspecs)),
                desc=f"Testing {args.model}"
                     + (f" [{trial_name}]" if trial_name else ""))

    with torch.no_grad():
        for index in pbar:
            zspec = torch.from_numpy(zspecs[index]).float().cuda()
            param = TensorDict(zparams[index], batch_size=[]).cuda()
            zspec = zspec[:, fit_range_mask]
            if args.snr is not None:
                zspec = awgn(zspec, snr=args.snr)
            pred_cest, zrec = model(zspec[None], param[None])
            out_cest = pred_cest.squeeze(0).cpu().numpy()

            pool_names = list(scanparams['cest_pool'].keys())
            for i, name in enumerate(pool_names):
                frng = scanparams['cest_pool'][name]['frng']
                krng = scanparams['cest_pool'][name]['krng']

                f_norm = np.clip(out_cest[i * 2], 0.0, 1.0)
                k_norm = np.clip(out_cest[i * 2 + 1], 0.0, 1.0)

                C = 10.0
                f_mM = frng[0] + f_norm * (frng[1] - frng[0])
                zparams[index]['cest_pool'][name]['f'] = mM2f(f_mM)

                if args.k_mode == 'loglog':
                    k_min_ll = np.log10(np.log10(krng[0] + C))
                    k_max_ll = np.log10(np.log10(krng[1] + C))
                    k_val = 10 ** (10 ** (k_min_ll + k_norm * (k_max_ll - k_min_ll))) - C
                elif args.k_mode == 'linear':
                    k_val = krng[0] + k_norm * (krng[1] - krng[0])
                else:
                    k_min_l = np.log10(krng[0] + C)
                    k_max_l = np.log10(krng[1] + C)
                    k_val = 10 ** (k_min_l + k_norm * (k_max_l - k_min_l)) - C

                zparams[index]['cest_pool'][name]['k'] = k_val
            Result.append(zparams[index])

    if trial_name:
        result_dir = os.path.join('results', args.model, trial_name)
    else:
        result_dir = os.path.join('results', args.model)
    os.makedirs(result_dir, exist_ok=True)
    save_dict = {'Result': Result, 'test_indices': indices}
    save_path = os.path.join(result_dir, 'Result.mat')
    savemat(save_path, save_dict)
    print(f"\nDone. Results saved to {save_path}")

    print("\n>>> Starting automatic evaluation...")
    eval_args = argparse.Namespace(
        datapath=args.datapath,
        model=args.model,
        visualize=False,
        save=True,
        show_mM=True,
        trial_name=trial_name,
    )
    all_metrics = evaluate.main(eval_args)

    print(f"\n{'=' * 20} Evaluation: {trial_name or 'default'} {'=' * 20}")
    for pool, metrics in all_metrics.items():
        if pool == 'extra_metrics':
            continue
        f_unit = 'mM' if eval_args.show_mM else 'fraction'
        print(f"  -{pool} k: r={metrics['k']['r']:.4f}, "
              f"RMSE={metrics['k']['rmse']:.4f} Hz, "
              f"NRMSE={metrics['k']['nrmse']:.2f}%")
        print(f"  -{pool} f: r={metrics['f']['r']:.4f}, "
              f"RMSE={metrics['f']['rmse']:.4f} {f_unit}, "
              f"NRMSE={metrics['f']['nrmse']:.2f}%")
        print("-" * 60)
    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, required=True)
    parser.add_argument('--k_mode', type=str, default='log',
                        choices=['log', 'loglog', 'linear'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--no_physics', action='store_false', dest='use_physics')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--snr', type=float, default=60)
    args = parser.parse_args()
    main(args)