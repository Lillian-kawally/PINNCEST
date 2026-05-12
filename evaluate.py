import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy.stats import pearsonr, gaussian_kde
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

from utils.utils import f2mM

sns.set_theme(style="ticks")
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['xtick.major.width'] = 2.0
matplotlib.rcParams['ytick.major.width'] = 2.0
matplotlib.rcParams['xtick.major.size'] = 8.0
matplotlib.rcParams['ytick.major.size'] = 8.0


def draw_publication_plot(ax_main, ax_marg_x, ax_marg_y, cax, gt, res, title, unit, pred_color, gt_color, fig):
    xy = np.vstack([gt, res])
    z = gaussian_kde(xy)(xy)
    z = (z - z.min()) / (z.max() - z.min())
    idx = z.argsort()
    gt_s, res_s, z_s = gt[idx], res[idx], z[idx]

    max_limit = max(gt.max(), res.max()) * 1.05
    ax_main.set_xlim(0, max_limit)
    ax_main.set_ylim(0, max_limit)

    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [gt_color, pred_color])
    sc = ax_main.scatter(gt_s, res_s, c=z_s, s=60, cmap=cmap, alpha=0.9,
                         edgecolor='white', linewidth=0.3, zorder=3)

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label('Relative Density', rotation=270, labelpad=15, fontsize=9, fontweight='bold')
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_edgecolor('#888888')

    ax_main.plot([0, max_limit], [0, max_limit], color='k', linestyle='--',
                 linewidth=1.5, alpha=0.8, label='Ideal', zorder=4)

    r, _ = pearsonr(gt, res)
    rmse = np.sqrt(np.mean((res - gt) ** 2))
    gt_range = gt.max() - gt.min()
    nrmse = (rmse / gt_range * 100) if gt_range != 0 else 0

    stats_text = f'$r = {r:.2f}$\n$RMSE = {rmse:.2f}$ {unit}\n$NRMSE = {nrmse:.1f}\%$'

    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes, fontsize=10,
                 verticalalignment='top',
                 bbox=dict(facecolor='none', alpha=0.9, edgecolor='none'), zorder=5)

    ax_main.set_xlabel(f'Ground Truth ({unit})', fontsize=11, fontweight='bold')
    ax_main.set_ylabel(f'Prediction ({unit})', fontsize=11, fontweight='bold')
    ax_main.legend(loc='lower right', frameon=False, fontsize=11)
    ax_main.grid(True, linestyle='--', alpha=0.3, zorder=2)

    sns.histplot(x=gt, ax=ax_marg_x, color=gt_color, kde=True, alpha=0.8,
                 element="bars", edgecolor='white')
    sns.histplot(y=res, ax=ax_marg_y, color=pred_color, kde=True, alpha=0.4,
                 element="bars", edgecolor='white')
    if ax_marg_x.lines:
        ax_marg_x.lines[0].set_color('#555555')
        ax_marg_x.lines[0].set_linewidth(1.5)
    if ax_marg_y.lines:
        ax_marg_y.lines[0].set_color(pred_color)
        ax_marg_y.lines[0].set_linewidth(1.5)

    for ax in [ax_marg_x, ax_marg_y]:
        ax.tick_params(axis='both', which='both', bottom=False, top=False,
                       left=False, right=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')


def plot_parameter(gt, res, pool_name, param_name, unit, pred_color, gt_color,
                   save_path=None, visualize=False):
    fig = plt.figure(figsize=(4, 3.5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 4], width_ratios=[4, 1, 0.3],
                           hspace=0.05, wspace=0.05)

    ax_marg_x = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_marg_y = fig.add_subplot(gs[1, 1], sharey=ax_main)
    cax = fig.add_subplot(gs[1, 2])
    ax_marg_x.sharex(ax_main)

    draw_publication_plot(ax_main, ax_marg_x, ax_marg_y, cax, gt, res,
                          f'{pool_name} {param_name}', unit, pred_color, gt_color, fig)

    param_title = 'Exchange Rate' if param_name == 'k' else 'Concentration'
    ax_marg_x.set_title(f'{pool_name} {param_title}', fontsize=12, pad=10)

    if save_path:
        plt.savefig(save_path + '.svg', format='svg', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}.svg")

    if visualize:
        plt.show()
    plt.close(fig)


def get_metrics(g, r):
    pearson_r, _ = pearsonr(g, r)
    rmse = np.sqrt(np.mean((r - g) ** 2))
    rng = g.max() - g.min()
    nrmse = (rmse / rng * 100) if rng != 0 else 0
    return {'r': float(pearson_r), 'rmse': float(rmse), 'nrmse': float(nrmse)}


def main(args):
    trial_name = getattr(args, 'trial_name', None)
    if trial_name:
        res_dir = os.path.join('results', args.model, trial_name)
    else:
        res_dir = os.path.join('results', args.model)
    respath = os.path.join(res_dir, 'Result.mat')
    plot_dir = os.path.join(res_dir, 'plots')
    data_dir = os.path.join(res_dir, 'data')

    if args.save:
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

    if not os.path.exists(respath):
        print(f"Error: File not found {respath}")
        return {}

    zparams_all = sio.loadmat(f'{args.datapath}/test/zparams.mat',
                              simplify_cells=True)['zparams']
    res_data = sio.loadmat(respath, simplify_cells=True)
    Result = res_data['Result']
    indices = res_data['test_indices']
    zparams = [zparams_all[i] for i in indices]
    poolnames = list(zparams[0]['cest_pool'].keys())

    color_gt, color_k_pred, color_f_pred = '#EAEAEA', '#3498db', '#e67e22'
    metrics_summary = {}

    for name in poolnames:
        is_new_format = 'water_pool' in Result[0]

        if is_new_format:
            gt_k = np.array([p['cest_pool'][name]['k'] for p in zparams])
            res_k = np.array([r['cest_pool'][name]['k'] for r in Result])
            gt_f = np.array([p['cest_pool'][name]['f'] for p in zparams])
            res_f = np.array([r['cest_pool'][name]['f'] for r in Result])
        else:
            gt_k = np.array([p['cest_pool'][name]['k'] for p in zparams])
            res_k = np.array([r[name]['k'] for r in Result])
            gt_f = np.array([p['cest_pool'][name]['f'] for p in zparams])
            res_f = np.array([r[name]['f'] for r in Result])

        f_unit = 'fraction'
        if args.show_mM:
            f_unit = 'mM'
            gt_f, res_f = f2mM(gt_f), f2mM(res_f)

        metrics_summary[name] = {'k': get_metrics(gt_k, res_k),
                                 'f': get_metrics(gt_f, res_f)}

        if args.visualize or args.save:
            plot_parameter(gt_k, res_k, name, 'k', 'Hz', color_k_pred, color_gt,
                           save_path=os.path.join(plot_dir, f'Eval_{name}_k') if args.save else None,
                           visualize=args.visualize)
            plot_parameter(gt_f, res_f, name, 'f', f_unit, color_f_pred, color_gt,
                           save_path=os.path.join(plot_dir, f'Eval_{name}_f') if args.save else None,
                           visualize=args.visualize)

    if args.save:
        with open(os.path.join(data_dir, 'metrics_summary.txt'), 'w') as f:
            for pool, m in metrics_summary.items():
                f.write(f"\n[{pool} Pool]\n")
                f.write(f"  k: r={m['k']['r']:.4f}, "
                        f"RMSE={m['k']['rmse']:.2f} Hz, "
                        f"NRMSE={m['k']['nrmse']:.1f}%\n")
                f.write(f"  f: r={m['f']['r']:.4f}, "
                        f"RMSE={m['f']['rmse']:.6f} {f_unit}, "
                        f"NRMSE={m['f']['nrmse']:.1f}%\n")

    trial_label = trial_name if trial_name else 'default'
    print(f"\nEvaluation complete for model: {args.model} [{trial_label}]")
    return metrics_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate CEST ANN Model')
    parser.add_argument('--datapath', type=str, default='../SimData/data')
    parser.add_argument('--model', type=str, default='MBCN')
    parser.add_argument('--visualize', action='store_true', help='Show plots on screen')
    parser.add_argument('--save', action='store_true', help='Save plots as vector graphics')
    parser.add_argument('--show_mM', action='store_true')
    parser.add_argument('--trial_name', type=str, default=None,
                        help='Trial name to read results from results/{model}/{trial_name}/')
    args = parser.parse_args()
    main(args)