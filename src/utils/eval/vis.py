import json
import math
import os

import pandas as pd
import torchvision
import matplotlib as mpl


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_per_run_metrics(obj):
    keys = ["runs", "results", "per_run", "all_results", "metrics", "experiments"]
    candidates = None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and isinstance(obj[k], list) and len(obj[k]) > 0:
                candidates = obj[k]; break
    if candidates is None:
        candidates = obj if isinstance(obj, list) and len(obj) > 0 else [obj]
    rows = []
    for item in candidates:
        metrics = item.get("metrics", item) if isinstance(item, dict) else item
        if isinstance(metrics, dict):
            row = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and np.isfinite(v)}
            if row:
                rows.append(row)
    return rows

def summarize_runs(rows):
    if len(rows) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    df = pd.DataFrame(rows).select_dtypes(include=[np.number])
    n = max(len(df), 1)
    mean = df.mean(axis=0, skipna=True)
    se = (df.std(axis=0, ddof=1) / math.sqrt(n)).fillna(0.0) if n > 1 else pd.Series(0.0, index=df.columns)
    return mean, se

def compute_y_limits(means, ses):
    arr_low = np.array(means) - np.array(ses)
    arr_high = np.array(means) + np.array(ses)
    if np.all(np.isnan(arr_low)) or np.all(np.isnan(arr_high)):
        return None
    minv = np.nanmin(arr_low)
    maxv = np.nanmax(arr_high)
    pad = 0.5 * (maxv - minv) if maxv > minv else 0.1 * abs(maxv) if maxv != 0 else 0.1
    return float(minv - pad), float(maxv + pad)

def choose_accuracy_metric(columns):
    cols = [c.lower() for c in columns]
    for pattern in ("accuracy", "acc", "top1"):
        for i, c in enumerate(cols):
            if pattern in c:
                return list(columns)[i]
    return None

import re

def sanitize_latex(s: str) -> str:
    """
    Safely escape LaTeX special characters in string `s`.
    - Prevents LaTeX errors from `_`, `%`, `$`, etc.
    - Avoids double-escaping already sanitized input (e.g., '\\_' stays '\\_').
    - Works well for Matplotlib PGF output and LaTeX tables.
    """
    if not isinstance(s, str):
        return s

    # Mapping of special LaTeX characters to their escaped forms
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '_': r'\_',
        '%': r'\%',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '~': r'\textasciitilde{}',
    }

    # Regex pattern to find LaTeX special characters
    pattern = re.compile(r'([\\{}_%$&#^~])')

    def repl(m):
        char = m.group(0)
        # If the character is already escaped (preceded by a backslash), skip escaping
        if m.start() > 0 and s[m.start() - 1] == '\\':
            return char
        return replacements.get(char, char)

    return pattern.sub(repl, s)


def plot_group_short(title, labels, means, ses, ylabel, save_path=None, use_pgf=True,rot=True):
    """
    Plot a short grouped bar plot with LaTeX-safe text.
    If save_path is provided, save as PGF (if use_pgf) or fallback by extension and do not show interactively.
    """
    safe_title = sanitize_latex(title)
    safe_ylabel = sanitize_latex(ylabel)
    safe_labels = [sanitize_latex(lbl) for lbl in labels]

    x = np.arange(len(safe_labels))
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(safe_labels))]

    fig, ax = plt.subplots()
    ax.bar(x, means, yerr=ses, capsize=5, color=colors, edgecolor="black", alpha=1.0,zorder=2)

    ax.set_xticks(x)

    if rot:
        rotation=20
        ha = "right"
    else:
        rotation=0
        ha = "center"
    ax.set_xticklabels(safe_labels, rotation=rotation, ha=ha)
    ax.set_ylabel(safe_ylabel)
    ax.set_title(safe_title, pad=6)

    ax.grid(axis="y", linestyle="--", alpha=0.7,zorder=-1)

    ylims = compute_y_limits(means, ses)
    if ylims is not None:
        ax.set_ylim(ylims)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.show()
    else:
        plt.show()



def build_and_write_latex(base_results_dir,architecture,group_key, group_title, labels, summaries):
    # common metrics across all variants
    common = None
    for s in summaries.values():
        cols = set(s["mean"].index.tolist())
        common = cols if common is None else (common & cols)
    if not common:
        print(f"[{group_key}] No common metrics for LaTeX. Skipping.")
        return
    metrics_sorted = sorted(common)
    def fmt(mu, se): return f"{mu:.4f} ± {se:.4f}"
    latex_df = pd.DataFrame(
        {lbl: [fmt(summaries[lbl]["mean"][m], summaries[lbl]["se"][m]) for m in metrics_sorted]
         for lbl in labels},
        index=metrics_sorted,
    )
    latex_df.index.name = "Metric"
    latex_str = latex_df.to_latex(escape=True, index=True,
                                  caption=f"{architecture} {group_title} (mean ± SE)",
                                  label=f"tab:{architecture}_{group_key}")
    out_path = os.path.join(base_results_dir, f"{architecture}_{group_key}_summary.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"\nLaTeX written to: `{out_path}`\n")
    print(latex_str)


def _default_pgf_path(root_dir, architecture, group_key, metric):
    """
    Autogenerate a PGF output path from inputs.
    Example: <root_dir>/plots/<architecture>/<architecture>_<group_key>_<metric>.pgf
    """
    safe_metric = str(metric).replace("/", "_").replace(" ", "_")
    fname = f"{architecture}_{group_key}_{safe_metric}.pgf"
    return os.path.join(root_dir, "plots", architecture, fname)
def _default_pdf_path(root_dir, architecture, group_key, metric):
    """
    Autogenerate a PDF output path from inputs.
    Example: <root_dir>/plots/<architecture>/<architecture>_<group_key>_<metric>.pdf
    """
    safe_metric = str(metric).replace("/", "_").replace(" ", "_")
    fname = f"{architecture}_{group_key}_{safe_metric}.pdf"
    return os.path.join(root_dir, "plots", architecture, fname)


def vis_dataset_image(train,val,test):
    # test images
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
    train_data = next(iter(train))[0]
    batch_size = train_data.shape[0]

    axs[0].imshow(torchvision.utils.make_grid(train_data, nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[1].imshow(torchvision.utils.make_grid(next(iter(val))[0], nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[2].imshow(
        torchvision.utils.make_grid(next(iter(test))[0], nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[0].set_title('Training')
    axs[1].set_title('Validation')
    axs[2].set_title('Test')

    for ax in axs.flat:
        ax.axis('off')

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def vis_dataset_image(train, val, test):
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(10, 5))
    train_data = next(iter(train))[0]
    batch_size = train_data.shape[0]

    axs[0].imshow(torchvision.utils.make_grid(train_data, nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[1].imshow(torchvision.utils.make_grid(next(iter(val))[0], nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[2].imshow(torchvision.utils.make_grid(next(iter(test))[0], nrow=batch_size // 4).permute(1, 2, 0).cpu())
    axs[0].set_title('Training')
    axs[1].set_title('Validation')
    axs[2].set_title('Test')

    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def _to_point_array(item):
    """Convert dataset item to (N,3) numpy array."""
    if isinstance(item, (tuple, list)):
        x = item[0]
    else:
        x = item

    # PyG Data object
    if hasattr(x, "pos"):
        pts = x.pos
    else:
        pts = x

    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy()
    pts = np.asarray(pts)

    # handle (3,N) -> (N,3)
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    # keep only xyz
    if pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    return pts


def vis_dataset_pointcloud(train, val, test, n_samples=4, figsize=(12, 8)):
    """
    Visualize a dataset point cloud. train val and test are dataloaders.
    N_sumples is the number of samples to take. figsize is the size of the matplotlib figure.

    """
    rows = 3
    cols = n_samples
    fig = plt.figure(figsize=figsize)
    loaders = [("train", train), ("val", val), ("test", test)]

    for r, (name, loader) in enumerate(loaders):
        batch = next(iter(loader))
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1, projection="3d")
            if c < len(batch[0]):
                item = batch[0][c] if isinstance(batch, (tuple, list)) else batch[c]
                pts = _to_point_array(item)
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
                ax.set_xticks([]);
                ax.set_yticks([]);
                ax.set_zticks([])
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(name)
    plt.tight_layout()
    plt.show()


def _plot_stroke_on_ax(seq, ax, linewidth=1.5, color="black"):
    """Plot single stroke sequence (N,4): dx,dy,pen,mask."""
    if isinstance(seq, torch.Tensor):
        seq = seq.detach().cpu().numpy()
    seq = np.asarray(seq)
    if seq.size == 0:
        return

    # Use mask to filter real points
    mask = seq[:, 3] if seq.shape[1] >= 4 else np.ones(seq.shape[0], dtype=bool)
    real_idx = mask == 1
    if not np.any(real_idx):
        return

    dx = seq[real_idx, 0]
    dy = seq[real_idx, 1]
    pen = seq[real_idx, 2] if seq.shape[1] >= 3 else np.zeros_like(dx)

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    xs, ys = [], []
    for xi, yi, p in zip(x, y, pen):
        xs.append(xi)
        ys.append(-yi)
        if p == 1:
            ax.plot(xs, ys, linewidth=linewidth, color=color)
            xs, ys = [], []
    if xs:
        ax.plot(xs, ys, linewidth=linewidth, color=color)

    ax.axis("equal")
    ax.axis("off")


def vis_dataset_stroke(train, val, test, n_samples=6, figsize=(12, 8)):
    rows = 3
    cols = n_samples
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    loaders = [("train", train), ("val", val), ("test", test)]

    for r, (name, loader) in enumerate(loaders):
        batch = next(iter(loader))
        sequences = batch[0] if isinstance(batch, (tuple, list)) else batch

        for c in range(cols):
            ax = axes[r, c] if rows > 1 else axes[c]
            if c < len(sequences):
                _plot_stroke_on_ax(sequences[c], ax)
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(name)
    plt.tight_layout()
    plt.show()


def vis_dataset(train, val, test):
    data_shape = train.dataset[0][0].shape
    if len(data_shape) == 3 and data_shape[0] in [1, 3]:  # image data (C, H, W)
        vis_dataset_image(train, val, test)
    elif len(data_shape) == 2 and data_shape[-1] == 3:  # point cloud data (N, 3)
        vis_dataset_pointcloud(train, val, test)
    elif len(data_shape) == 2 and data_shape[1] == 4:  # stroke data (N, 4)
        vis_dataset_stroke(train, val, test)
    else:
        raise ValueError(f"Unsupported data shape for visualization: {data_shape}")



def process_and_plot_groups(groups, base_results_dir, architecture, save_root=None, save_pgf=True,width=None,height=None):
    """
    Process and plot each group in `groups`.
    Each group is (group_key, group_title, items) where items is list of (label, filename).
    Files are looked up under `base_results_dir`.

    If save_root is provided (or None to use base_results_dir), figures are saved as PGF under:
      <save_root or base_results_dir>/plots/<architecture>/<architecture>_<group_key>_<metric>.pgf
    """
    for group_key, group_title, items in groups:
        # keep only files that exist, in the declared order
        existing = [(lbl, os.path.join(base_results_dir, fname)) for (lbl, fname) in items
                    if os.path.isfile(os.path.join(base_results_dir, fname))]

        # load and summarize
        summaries = {}
        for lbl, fpath in existing:
            obj = load_json(fpath)
            rows = extract_per_run_metrics(obj)
            mean, se = summarize_runs(rows)
            summaries[lbl] = {"mean": mean, "se": se}

        # choose plotting metric from intersection (prefer accuracy-like)
        common = None
        for s in summaries.values():
            cols = set(s["mean"].index.tolist())
            common = cols if common is None else (common & cols)
        if not common:
            print(f"[{group_key}] No common numeric metrics. Skipping plot.")
            continue
        metric = choose_accuracy_metric(sorted(common)) or sorted(common)[0]

        labels = [lbl for (lbl, _) in existing]
        means = [summaries[lbl]["mean"].get(metric, np.nan) for lbl in labels]
        ses   = [summaries[lbl]["se"].get(metric, 0.0) for lbl in labels]

        # decide save path (PGF) if requested
        root_for_save = (save_root if save_root is not None else base_results_dir)
        save_path = _default_pgf_path(root_for_save, architecture, group_key, metric) if save_pgf else _default_pdf_path(root_for_save, architecture, group_key, metric)

        # plot (and optionally save)
        plot_group_short(None, labels, means, ses, ylabel=metric.capitalize(),
                         save_path=save_path, use_pgf=save_pgf)

        # LaTeX (full common metrics)
        build_and_write_latex(base_results_dir, architecture,
                              group_key, group_title, labels, summaries)


def plt_setup_latex(W=4.9823):
    import matplotlib.pyplot as plt
    mpl.rcdefaults()  # resets to default style
    mpl.pyplot.close('all')  # close figures

    plt.style.use('default')
    plt.rcParams.update({
        # --- General text settings ---
        "text.usetex": True,
        "font.family": "serif",  # Use a generic family (avoids 'unknown font' warnings)

        'font.size': 9.5,  # Set font size to 11pt
        'axes.labelsize': 9.5,  # -> axis labels
        'legend.fontsize': 9.5,  # -> legends
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        #set dpi
        "figure.dpi": 300,

        # --- PGF (LaTeX) compatibility ---
        "pgf.texsystem": "pdflatex",  # or 'xelatex' / 'lualatex' if you prefer
        "pgf.rcfonts": False,  # Use LaTeX fonts, not matplotlib's
        "pgf.preamble": r"""
            \usepackage[T1]{fontenc}
            \usepackage[utf8]{inputenc}
            \usepackage{lmodern}       % Use Latin Modern for both pgf and pdf outputs
        """,
    })

    # Options
    plt.rcParams.update({
        'figure.figsize': (W, W / (6 / 3)),  # 4:3 aspect ratio
    })

    return W



def plt_setup_paper(W=5.5): #4.803 W would be correct reexport?
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcdefaults()
    plt.close('all')

    plt.style.use('default')

    plt.rcParams.update({
        # --- LaTeX + Font ---
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,

        # --- Font sizes ---
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # --- Figure DPI ---
        "figure.dpi": 300,

        # --- Grid & line clarity ---
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.8,

        # --- LaTeX preamble (CM + T1 encoding) ---
        "pgf.preamble": r"""
        \usepackage[T1]{fontenc}
        \usepackage[utf8]{inputenc}
        """
    })


    plt.rcParams.update({
        "figure.figsize": (W, W * 0.75),
    })

    return W
