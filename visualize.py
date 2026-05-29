"""
Visualize benchmark results produced by step_benchmark.py.

Reads the `results.pkl` written by a benchmark run and emits PNGs.

Usage:
    python visualize.py path/to/results.pkl [--out-dir DIR]
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# StepTimings must be importable for pickle to deserialize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step_benchmark import StepTimings  # noqa: F401


STAGE_SPECS = [
    ("Preprocess",       "preprocess_ms",             "#9e9e9e"),
    ("Duration pred.",   "duration_predictor_ms",     "#e57373"),
    ("Text encoder",     "text_encoder_ms",           "#ffb74d"),
    ("Noisy latent",     "noisy_latent_ms",           "#fff176"),
    ("Vector estimator", "vector_estimator_total_ms", "#4fc3f7"),
    ("Vocoder",          "vocoder_ms",                "#81c784"),
    ("Audio trim",       "audio_trim_ms",             "#ba68c8"),
]


def _mean_std(timings, attr):
    vals = [getattr(t, attr) for t in timings]
    return float(np.mean(vals)), float(np.std(vals))


def plot_stage_breakdown(data, out_path):
    """Stacked bar per voice — one subplot per (steps × text)."""
    results = data["results"]
    voices = data["voices"]
    texts = list(data["texts"].keys())
    step_counts = sorted(data["step_counts"])

    nrows, ncols = len(step_counts), len(texts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows + 0.4),
                             squeeze=False)
    x = np.arange(len(voices))
    width = 0.65

    for ri, steps in enumerate(step_counts):
        for ci, text_label in enumerate(texts):
            ax = axes[ri][ci]
            bottoms = np.zeros(len(voices))
            for _, attr, color in STAGE_SPECS:
                means = []
                for v in voices:
                    vt = results.get((steps, text_label, v), [])
                    means.append(_mean_std(vt, attr)[0] if vt else 0.0)
                ax.bar(x, means, width, bottom=bottoms, color=color,
                       edgecolor="white", linewidth=0.5)
                bottoms += np.array(means)

            # Std of TOTAL as error bar capping the stack
            total_stds = []
            for v in voices:
                vt = results.get((steps, text_label, v), [])
                total_stds.append(_mean_std(vt, "total_ms")[1] if vt else 0.0)
            ax.errorbar(x, bottoms, yerr=total_stds, fmt="none", ecolor="black",
                        capsize=4, capthick=1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(voices)
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"steps={steps}  ·  text={text_label}")
            ax.grid(axis="y", alpha=0.25)

            stds_arr = np.array(total_stds)
            ymax = max((bottoms + stds_arr).max() * 1.12, 1.0)
            ax.set_ylim(0, ymax)

            for xi, (top, sd) in enumerate(zip(bottoms, stds_arr)):
                label_y = min(top + sd + ymax * 0.01, ymax * 0.99)
                ax.text(xi, label_y, f"{top:.0f}",
                        ha="center", va="bottom", fontsize=8, clip_on=True)

    handles = [mpatches.Patch(color=c, label=n) for n, _, c in STAGE_SPECS]
    fig.legend(handles=handles, loc="lower center", ncol=len(STAGE_SPECS),
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10)
    fig.suptitle("Stage breakdown per voice  (mean over reps; error bar = total-ms std)",
                 fontsize=14, y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_vector_estimator_steps(data, out_path):
    """Per-step ms inside the vector estimator loop, pooled over voices×reps."""
    results = data["results"]
    voices = data["voices"]
    texts = list(data["texts"].keys())
    step_counts = sorted(data["step_counts"])

    nrows, ncols = len(step_counts), len(texts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for ri, steps in enumerate(step_counts):
        for ci, text_label in enumerate(texts):
            ax = axes[ri][ci]
            all_t = []
            for v in voices:
                all_t.extend(results.get((steps, text_label, v), []))
            if not all_t:
                ax.set_visible(False)
                continue
            per_step = np.array([t.vector_estimator_per_step_ms for t in all_t])
            means = per_step.mean(axis=0)
            stds = per_step.std(axis=0)

            x = np.arange(len(means))
            colors = ["#d32f2f"] + ["#4fc3f7"] * (len(means) - 1)
            ax.bar(x, means, color=colors, yerr=stds, capsize=3,
                   edgecolor="white", linewidth=0.5)
            ax.set_xticks(x)
            ax.set_xlabel("Step index")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"steps={steps}  ·  text={text_label}  ·  n={len(all_t)}")
            ax.grid(axis="y", alpha=0.25)

            if len(means) > 1:
                rest_mean = means[1:].mean()
                ax.axhline(rest_mean, color="gray", linestyle="--", alpha=0.6,
                           label=f"steps 1+ mean = {rest_mean:.0f} ms")
                ax.text(0.98, 0.92,
                        f"step 0 = {means[0]/rest_mean:.2f}× rest",
                        transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=9, color="#d32f2f", weight="bold",
                        bbox=dict(facecolor="white", edgecolor="#d32f2f",
                                  alpha=0.85, pad=2, linewidth=0.6))
                ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Vector estimator: per-step latency  "
                 "(step 0 spike = first-call kernel/buffer init)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_total_distributions(data, out_path):
    """Box plot of total ms per voice (1 box = N reps)."""
    results = data["results"]
    voices = data["voices"]
    texts = list(data["texts"].keys())
    step_counts = sorted(data["step_counts"])

    nrows, ncols = len(step_counts), len(texts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows),
                             squeeze=False)

    for ri, steps in enumerate(step_counts):
        for ci, text_label in enumerate(texts):
            ax = axes[ri][ci]
            box_data = []
            for v in voices:
                vt = results.get((steps, text_label, v), [])
                box_data.append([t.total_ms for t in vt] if vt else [0])
            bp = ax.boxplot(
                box_data, tick_labels=voices, showmeans=True, patch_artist=True,
                meanprops={"marker": "D", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 5},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor("#4fc3f7")
                patch.set_alpha(0.6)
            ax.set_ylabel("Total time (ms)")
            ax.set_title(f"steps={steps}  ·  text={text_label}")
            ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Total-time distribution per voice  "
                 f"({data['n_reps']} reps each; ◇ = mean, line = median)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cross_step(data, out_path):
    """Grouped bar: x=voice, groups=step counts, y=mean total ms ± std."""
    results = data["results"]
    voices = data["voices"]
    texts = list(data["texts"].keys())
    step_counts = sorted(data["step_counts"])

    fig, axes = plt.subplots(1, len(texts), figsize=(6 * len(texts), 4.5),
                             squeeze=False)
    x = np.arange(len(voices))
    width = 0.8 / len(step_counts)
    colors = plt.cm.viridis(np.linspace(0.25, 0.75, len(step_counts)))

    for ci, text_label in enumerate(texts):
        ax = axes[0][ci]
        for si, steps in enumerate(step_counts):
            means, stds = [], []
            for v in voices:
                vt = results.get((steps, text_label, v), [])
                if not vt:
                    means.append(0); stds.append(0); continue
                mu, sd = _mean_std(vt, "total_ms")
                means.append(mu); stds.append(sd)
            offset = (si - (len(step_counts) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                   label=f"steps={steps}", color=colors[si],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(voices)
        ax.set_ylabel("Total time (ms)")
        ax.set_title(f"text={text_label}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Step-count impact on total time per voice  (error bars = std over reps)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def generate_plots(pkl_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    plot_stage_breakdown(data,         out_dir / "1_stage_breakdown.png")
    plot_vector_estimator_steps(data,  out_dir / "2_vector_estimator_per_step.png")
    plot_total_distributions(data,     out_dir / "3_total_time_distributions.png")
    plot_cross_step(data,              out_dir / "4_cross_step_compare.png")
    print(f"Wrote 4 plots to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("pkl_path", help="Path to results.pkl from a benchmark run")
    p.add_argument("--out-dir", default=None,
                   help="Output dir for plots (default: <run_dir>/plots)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pkl_path = Path(args.pkl_path)
    out_dir = Path(args.out_dir) if args.out_dir else pkl_path.parent / "plots"
    generate_plots(pkl_path, out_dir)
