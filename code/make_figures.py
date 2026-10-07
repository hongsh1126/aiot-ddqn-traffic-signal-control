"""
Generate paper figures from results_aiot.json:
  - fig_perf_comparison.pdf : single-intersection performance bar chart
  - fig_grid_results.pdf    : grid network performance
  - fig_v2x_sensitivity.pdf : V2X channel impact
  - fig_ablation.pdf        : sensing & reward ablations
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 9,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'figure.dpi': 110, 'savefig.bbox': 'tight'
})

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(CODE_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(CODE_DIR, "results_aiot.json")) as f:
    R = json.load(f)


def fig_perf_comparison():
    methods = ["Fixed-Time", "Actuated", "MaxPressure",
               "DQN", "IntelliLight", "AIoT-DDQN"]
    demands = ["Low", "Medium", "High"]
    metrics = ["avg_wait_s", "queue_veh", "throughput_vph", "emission_g_per_veh"]
    titles = ["Avg Waiting Time (s)", "Queue Length (veh)",
              "Throughput (veh/h)", "Emission (g/veh)"]
    colors = ["#888888", "#EE9988", "#88AAEE",
              "#22AA88", "#AA88CC", "#CC4444"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    bar_w = 0.13
    x_base = np.arange(len(demands))
    for ax, met, title in zip(axes.flat, metrics, titles):
        for i, m in enumerate(methods):
            vals = [R["single"][d][m][met] for d in demands]
            ax.bar(x_base + i * bar_w, vals, width=bar_w,
                   label=m, color=colors[i], edgecolor='black', lw=0.4)
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x_base + 2.5 * bar_w)
        ax.set_xticklabels(demands)
        ax.grid(axis='y', alpha=0.3)
        if met == "throughput_vph":
            ax.set_ylabel("veh/h")
        elif met == "emission_g_per_veh":
            ax.set_ylabel("g/veh")
        else:
            ax.set_ylabel(title.split('(')[1][:-1] if '(' in title else "")
    axes[0, 0].legend(loc='upper left', ncol=2, fontsize=8)
    plt.suptitle("Performance Comparison at Three Demand Levels (Single Intersection)",
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig_perf_comparison.pdf")
    plt.savefig(out)
    plt.savefig(out.replace('.pdf', '.png'))
    plt.close()
    print(f"  -> {out}")


def fig_grid_results():
    methods = list(R["grid"].keys())
    waits = [R["grid"][m]["avg_wait_s"] for m in methods]
    thr = [R["grid"][m]["throughput_vph"] for m in methods]
    em = [R["grid"][m]["emission_g_per_veh"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(methods))
    bar_w = 0.28
    ax1.bar(x - bar_w, waits, width=bar_w, label='Wait time (s)',
            color='#3366AA', edgecolor='black', lw=0.4)
    ax1.bar(x, em, width=bar_w, label='Emission (g/veh)',
            color='#AA6633', edgecolor='black', lw=0.4)
    ax1.set_ylabel('Wait time (s) / Emission (g/veh)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15)
    ax1.grid(axis='y', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(x + bar_w, thr, width=bar_w, label='Throughput (veh/h)',
            color='#229933', edgecolor='black', lw=0.4)
    ax2.set_ylabel('Throughput (veh/h)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right',
               fontsize=8.5)
    plt.title('2x2 Grid Network Performance (Medium Demand)',
              fontweight='bold')
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_grid_results.pdf')
    plt.savefig(out)
    plt.savefig(out.replace('.pdf', '.png'))
    plt.close()
    print(f"  -> {out}")


def fig_v2x_sensitivity():
    conditions = list(R["v2x"].keys())
    delays = [R["v2x"][c]["delay_ms"] for c in conditions]
    losses = [R["v2x"][c]["loss"] for c in conditions]
    waits = [R["v2x"][c]["avg_wait_s"] for c in conditions]
    thrs = [R["v2x"][c]["throughput_vph"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(delays, waits, 'o-', color='#CC4444', lw=2, ms=8,
             label='Avg waiting time')
    ax1.set_xlabel('V2X mean delay (ms)')
    ax1.set_ylabel('Avg waiting time (s)', color='#CC4444')
    ax1.tick_params(axis='y', labelcolor='#CC4444')
    ax1.grid(alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(delays, thrs, 's--', color='#229933', lw=2, ms=8,
              label='Throughput')
    ax1b.set_ylabel('Throughput (veh/h)', color='#229933')
    ax1b.tick_params(axis='y', labelcolor='#229933')
    for x, y, c in zip(delays, waits, conditions):
        ax1.annotate(c, (x, y), textcoords='offset points', xytext=(5, 8),
                     fontsize=8)
    ax1.set_title('Effect of V2X Channel Delay', fontweight='bold')

    ax2.plot([l * 100 for l in losses], waits, 'o-', color='#3366AA',
             lw=2, ms=8)
    ax2.set_xlabel('V2X packet loss rate (%)')
    ax2.set_ylabel('Avg waiting time (s)')
    ax2.grid(alpha=0.3)
    ax2.set_title('Effect of V2X Packet Loss', fontweight='bold')
    for x, y, c in zip([l * 100 for l in losses], waits, conditions):
        ax2.annotate(c, (x, y), textcoords='offset points', xytext=(5, 8),
                     fontsize=8)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_v2x_sensitivity.pdf')
    plt.savefig(out)
    plt.savefig(out.replace('.pdf', '.png'))
    plt.close()
    print(f"  -> {out}")


def fig_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # sensing ablation
    sens = R["state_ablation"]
    cfg = list(sens.keys())
    waits_s = [sens[c]["avg_wait_s"] for c in cfg]
    thr_s = [sens[c]["throughput_vph"] for c in cfg]
    em_s = [sens[c]["emission_g_per_veh"] for c in cfg]
    x = np.arange(len(cfg))
    bw = 0.28
    ax = axes[0]
    ax.bar(x - bw, waits_s, width=bw, label='Wait (s)', color='#3366AA', edgecolor='k', lw=0.4)
    ax.bar(x, em_s, width=bw, label='Emission (g/veh)', color='#AA6633', edgecolor='k', lw=0.4)
    ax2 = ax.twinx()
    ax2.bar(x + bw, thr_s, width=bw, label='Throughput', color='#229933', edgecolor='k', lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('(', '\n(') for c in cfg], fontsize=8.5)
    ax.set_ylabel('Wait (s) / Emission (g/veh)')
    ax2.set_ylabel('Throughput (veh/h)')
    ax.set_title('Sensing-Modality Ablation', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    # reward weight ablation
    rew = R["reward_ablation"]
    labels = list(rew.keys())
    waits_r = [rew[l]["avg_wait_s"] for l in labels]
    thr_r = [rew[l]["throughput_vph"] for l in labels]
    em_r = [rew[l]["emission_g_per_veh"] for l in labels]
    x = np.arange(len(labels))
    ax = axes[1]
    ax.bar(x - bw, waits_r, width=bw, label='Wait (s)', color='#3366AA', edgecolor='k', lw=0.4)
    ax.bar(x, em_r, width=bw, label='Emission (g/veh)', color='#AA6633', edgecolor='k', lw=0.4)
    ax3 = ax.twinx()
    ax3.bar(x + bw, thr_r, width=bw, label='Throughput', color='#229933', edgecolor='k', lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('Wait (s) / Emission (g/veh)')
    ax3.set_ylabel('Throughput (veh/h)')
    ax.set_title('Reward-Weight Ablation', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h3, l1 + l3, loc='upper right', fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_ablation.pdf')
    plt.savefig(out)
    plt.savefig(out.replace('.pdf', '.png'))
    plt.close()
    print(f"  -> {out}")


if __name__ == "__main__":
    print("Generating paper figures...")
    fig_perf_comparison()
    fig_grid_results()
    fig_v2x_sensitivity()
    fig_ablation()
    print("Done.")
