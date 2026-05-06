"""
=============================================================================
Entanglement-Assisted Decision Making for VNF Migration in 6G Networks
=============================================================================
Simulation of classical and quantum algorithmic approaches for binary
decision making in VNF migration, based on:

  Maheshwari et al., "Entanglement-assisted decision making for VNF
  migration in 6G Communication Networks", IEEE NFV-SDN 2024.

The simulation models:
  - Network topology (central node + N normal nodes)
  - Classical solution latency (with and without anonymity constraint)
  - Quantum solution latency (inherently anonymous via Bell pairs)
  - Monte Carlo analysis over randomised network conditions
  - Comparative plots across all three regimes

Author: Simulation based on paper by Maheshwari, Raman, Bassoli, Fitzek (TU Dresden / CeTI)
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import List, Tuple
import random

# ─────────────────────────────────────────────────────────────
# 1.  PHYSICAL / NETWORK PARAMETERS
# ─────────────────────────────────────────────────────────────

@dataclass
class NetworkParams:
    """
    Tunable physical parameters for the simulation.
    All times in microseconds (µs) unless noted.
    """
    # --- Topology ---
    n_normal_nodes: int = 10          # number of normal nodes in the neighbourhood
    n_vnf_to_migrate: int = 2         # paper fixes this at 2 (VNF1, VNF2)

    # --- Classical communication latencies ---
    tau_CN_proc_ov: float = 5.0       # central node overload detection processing time
    tau_CN_inode_base: float = 2.0    # base CN → node_i communication latency
    tau_CN_inode_jitter: float = 0.5  # ± jitter on CN-node link (random per run)
    tau_inode_proc: float = 1.5       # node_i processing time (status check)
    tau_inode_CN: float = 2.0         # node_i → CN reply latency
    tau_CN_proc: float = 3.0          # CN processing to choose candidate nodes A, B
    tau_CN_dec: float = 1.0           # CN internal decision latency (non-anon case)
    tau_CN_jnode: float = 2.0         # CN → chosen node j decision communication
    tau_jnode_prep: float = 2.5       # node j preparation to receive VNF
    tau_jnode_CN: float = 2.0         # node j → CN confirmation
    tau_CN_jnode_migrate: float = 10.0  # VNF actual migration transfer latency

    # --- Classical anonymity overhead (outsource to 3rd-party RNG) ---
    tau_CN_3RNG: float = 1.5          # CN → RNG request latency
    tau_3RNG_proc: float = 2.0        # RNG processing / random bit generation
    tau_3RNG_jnode: float = 2.5       # RNG → node j distribution latency

    # --- Quantum-specific latencies ---
    tau_CN_EP: float = 0.8            # Bell pair generation at entanglement source
    tau_EP_CN_jnode: float = 1.2      # Entangled photon distribution CN → nodes A, B
    tau_jnode_meas: float = 0.3       # Photon measurement at node (near-instantaneous)
    # tau_jnode_prep, tau_jnode_CN, tau_CN_jnode_migrate reused from classical


# ─────────────────────────────────────────────────────────────
# 2.  LATENCY MODELS  (direct from paper's flowchart, Fig. 3)
# ─────────────────────────────────────────────────────────────

def latency_classical_no_anon(p: NetworkParams, rng: np.random.Generator) -> dict:
    """
    Classical solution WITHOUT anonymity constraint.

    Flowchart stages (paper Fig. 3, left column):
      Stage 1: CN overload detection
      Stage 2: CN ↔ neighbours (rounds of communication) — taken as max over nodes
      Stage 3: CN processes info, selects candidate nodes A & B
      Stage 4: CN decides + communicates to nodes + nodes prepare + confirm + migrate
    """
    # Stage 1 — overload detection
    T1 = p.tau_CN_proc_ov

    # Stage 2 — CN ↔ all neighbouring nodes (parallel broadcasts, so take MAX)
    # Each node has slightly different link quality (jitter)
    link_latencies = (
        p.tau_CN_inode_base
        + rng.uniform(-p.tau_CN_inode_jitter, p.tau_CN_inode_jitter, p.n_normal_nodes)
    )
    T2 = float(np.max(link_latencies + p.tau_inode_proc + p.tau_inode_CN))

    # Stage 3 — CN selects candidates
    T3 = p.tau_CN_proc

    # Stage 4 — Classical decision + notify j-nodes + prep + confirm + migrate
    # Paper expression: max(τ_CNdec + τ_CN-jnode + τ_jnode_prep + τ_jnode-CN + τ_CN-jnode_migrate)
    T4 = p.tau_CN_dec + p.tau_CN_jnode + p.tau_jnode_prep + p.tau_jnode_CN + p.tau_CN_jnode_migrate

    T_total = T1 + T2 + T3 + T4
    return {
        "T_total": T_total,
        "T1_overload_detect": T1,
        "T2_neighbour_comm": T2,
        "T3_candidate_select": T3,
        "T4_decision_migrate": T4,
    }


def latency_classical_anon(p: NetworkParams, rng: np.random.Generator) -> dict:
    """
    Classical solution WITH anonymity constraint.

    The central node must outsource the decision to a trusted 3rd-party RNG.
    Stage 4 changes:
      (τ_CNdec + τ_CN-jnode)  →  (τ_CN-3RNG + τ_3RNG_proc + τ_3RNG-jnode)
    """
    T1 = p.tau_CN_proc_ov

    link_latencies = (
        p.tau_CN_inode_base
        + rng.uniform(-p.tau_CN_inode_jitter, p.tau_CN_inode_jitter, p.n_normal_nodes)
    )
    T2 = float(np.max(link_latencies + p.tau_inode_proc + p.tau_inode_CN))

    T3 = p.tau_CN_proc

    # Anonymity overhead replaces CN decision with 3rd-party RNG round-trip
    T4_anon = (
        p.tau_CN_3RNG
        + p.tau_3RNG_proc
        + p.tau_3RNG_jnode
        + p.tau_jnode_prep
        + p.tau_jnode_CN
        + p.tau_CN_jnode_migrate
    )

    T_total = T1 + T2 + T3 + T4_anon
    return {
        "T_total": T_total,
        "T1_overload_detect": T1,
        "T2_neighbour_comm": T2,
        "T3_candidate_select": T3,
        "T4_anon_rng_migrate": T4_anon,
    }


def latency_quantum(p: NetworkParams, rng: np.random.Generator) -> dict:
    """
    Quantum (entanglement-assisted) solution.

    Stages 1–3 are identical to classical.
    Stage 4 replaces CN decision + RNG with Bell pair generation,
    distribution, and measurement — which *inherently* ensures anonymity.

    Paper expression for quantum Stage 4:
      max(τ_CNEP + τ_EP_CN-jnode + τ_jnode_meas + τ_jnode_prep
          + τ_jnode-CN + τ_CN-jnode_migrate)
    """
    T1 = p.tau_CN_proc_ov

    link_latencies = (
        p.tau_CN_inode_base
        + rng.uniform(-p.tau_CN_inode_jitter, p.tau_CN_inode_jitter, p.n_normal_nodes)
    )
    T2 = float(np.max(link_latencies + p.tau_inode_proc + p.tau_inode_CN))

    T3 = p.tau_CN_proc

    # Quantum Stage 4 — entanglement generation + distribution + measurement
    T4_quantum = (
        p.tau_CN_EP
        + p.tau_EP_CN_jnode
        + p.tau_jnode_meas          # near-instantaneous quantum measurement
        + p.tau_jnode_prep
        + p.tau_jnode_CN
        + p.tau_CN_jnode_migrate
    )

    T_total = T1 + T2 + T3 + T4_quantum
    return {
        "T_total": T_total,
        "T1_overload_detect": T1,
        "T2_neighbour_comm": T2,
        "T3_candidate_select": T3,
        "T4_quantum_entangle_migrate": T4_quantum,
    }


# ─────────────────────────────────────────────────────────────
# 3.  BELL PAIR SIMULATION (quantum decision mechanism)
# ─────────────────────────────────────────────────────────────

def simulate_bell_pair_decision(n_trials: int = 1000, seed: int = 42) -> dict:
    """
    Simulate the quantum decision mechanism using Bell state |Ψ+⟩.

    |Ψ+⟩ = (1/√2)(|10⟩ + |01⟩)

    When measured, nodes A and B always get complementary outcomes:
      Node A = 0  →  Node B = 1  (A hosts VNF1, B hosts VNF2)
      Node A = 1  →  Node B = 0  (A hosts VNF2, B hosts VNF1)

    This guarantees:
      (a) Each VNF goes to exactly one node (no collision)
      (b) Central node does not know the outcome until nodes report back
          (inherent anonymity — no 3rd party RNG needed)
    """
    rng = np.random.default_rng(seed)

    results_A = []
    results_B = []
    vnf_assignments = []  # (vnf_at_A, vnf_at_B)

    for _ in range(n_trials):
        # Collapse of |Ψ+⟩ — uniform 50/50, always anti-correlated
        node_A_result = rng.integers(0, 2)   # 0 or 1
        node_B_result = 1 - node_A_result    # perfectly anti-correlated

        # Convention: 0 → VNF1, 1 → VNF2
        vnf_at_A = "VNF1" if node_A_result == 0 else "VNF2"
        vnf_at_B = "VNF1" if node_B_result == 0 else "VNF2"

        results_A.append(node_A_result)
        results_B.append(node_B_result)
        vnf_assignments.append((vnf_at_A, vnf_at_B))

    results_A = np.array(results_A)
    results_B = np.array(results_B)

    # Verify Bell-state properties
    correlation = np.corrcoef(results_A, results_B)[0, 1]
    collision_rate = np.mean(results_A == results_B)   # should be 0
    vnf1_at_A = np.mean(results_A == 0)

    return {
        "n_trials": n_trials,
        "correlation_AB": correlation,         # expected ≈ -1.0
        "collision_rate": collision_rate,       # expected = 0.0
        "vnf1_at_A_fraction": vnf1_at_A,       # expected ≈ 0.5
        "results_A": results_A,
        "results_B": results_B,
        "vnf_assignments": vnf_assignments,
    }


# ─────────────────────────────────────────────────────────────
# 4.  MONTE CARLO SIMULATION  (vary network size & link quality)
# ─────────────────────────────────────────────────────────────

def run_monte_carlo(
    n_nodes_range: List[int] = None,
    n_trials: int = 500,
    seed: int = 0,
) -> dict:
    """
    Run all three latency models across a range of network sizes.
    Returns mean ± std for each scenario.
    """
    if n_nodes_range is None:
        n_nodes_range = list(range(2, 51, 2))

    rng = np.random.default_rng(seed)
    p = NetworkParams()

    results = {
        "n_nodes": n_nodes_range,
        "classical_no_anon": {"mean": [], "std": []},
        "classical_anon":    {"mean": [], "std": []},
        "quantum":           {"mean": [], "std": []},
    }

    for n in n_nodes_range:
        p.n_normal_nodes = n
        c_na, c_a, q = [], [], []
        for _ in range(n_trials):
            c_na.append(latency_classical_no_anon(p, rng)["T_total"])
            c_a.append(latency_classical_anon(p, rng)["T_total"])
            q.append(latency_quantum(p, rng)["T_total"])

        for key, lst in [("classical_no_anon", c_na),
                         ("classical_anon", c_a),
                         ("quantum", q)]:
            results[key]["mean"].append(float(np.mean(lst)))
            results[key]["std"].append(float(np.std(lst)))

    return results


def run_sensitivity(param_name: str, values: np.ndarray, seed: int = 1) -> dict:
    """
    Vary a single NetworkParams field and record total latency for all three models.
    """
    rng = np.random.default_rng(seed)
    p = NetworkParams()

    out = {"param_values": list(values),
           "classical_no_anon": [],
           "classical_anon": [],
           "quantum": []}

    for v in values:
        setattr(p, param_name, float(v))
        # Average over 200 trials to reduce jitter noise
        c_na = np.mean([latency_classical_no_anon(p, rng)["T_total"] for _ in range(200)])
        c_a  = np.mean([latency_classical_anon(p, rng)["T_total"]    for _ in range(200)])
        q    = np.mean([latency_quantum(p, rng)["T_total"]            for _ in range(200)])
        out["classical_no_anon"].append(float(c_na))
        out["classical_anon"].append(float(c_a))
        out["quantum"].append(float(q))
        # Reset to default
        setattr(p, param_name, getattr(NetworkParams(), param_name))

    return out


# ─────────────────────────────────────────────────────────────
# 5.  PLOTTING
# ─────────────────────────────────────────────────────────────

COLORS = {
    "classical_no_anon": "#2196F3",   # blue
    "classical_anon":    "#F44336",   # red
    "quantum":           "#4CAF50",   # green
}
LABELS = {
    "classical_no_anon": "Classical (no anonymity)",
    "classical_anon":    "Classical + Anonymity (3rd-party RNG)",
    "quantum":           "Quantum Entanglement (inherently anonymous)",
}


def plot_all(mc: dict, bell: dict, sens_link: dict, sens_ep: dict,
             single_run: dict, out_path: str = r"C:\Users\SHASHANK\Downloads\vnf_migration_simulation.png"):

    fig = plt.figure(figsize=(18, 14), facecolor="#0f0f1a")
    fig.suptitle(
        "VNF Migration: Classical vs Quantum Entanglement-Assisted Decision Making\n"
        "Based on Maheshwari et al., IEEE NFV-SDN 2024 — TU Dresden / CeTI",
        fontsize=13, color="white", y=0.98, fontweight="bold"
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97, top=0.93, bottom=0.05)

    ax_style = dict(facecolor="#1a1a2e", grid_color="#333355")

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor(ax_style["facecolor"])
        ax.set_title(title, color="white", fontsize=10, pad=6)
        ax.set_xlabel(xlabel, color="#aaaacc", fontsize=8)
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=8)
        ax.tick_params(colors="#aaaacc", labelsize=7)
        ax.spines[:].set_color("#333355")
        ax.grid(color="#333355", linestyle="--", alpha=0.6, linewidth=0.5)

    # ── (A) Total latency vs network size ──────────────────────────────
    ax0 = fig.add_subplot(gs[0, :2])
    style_ax(ax0, "Total Migration Latency vs. Number of Neighbouring Nodes",
             "Number of Normal Nodes", "Latency (µs)")
    ns = mc["n_nodes"]
    for k in ["classical_no_anon", "classical_anon", "quantum"]:
        m = np.array(mc[k]["mean"])
        s = np.array(mc[k]["std"])
        ax0.plot(ns, m, color=COLORS[k], label=LABELS[k], linewidth=2)
        ax0.fill_between(ns, m - s, m + s, color=COLORS[k], alpha=0.18)
    ax0.legend(fontsize=7.5, facecolor="#111133", labelcolor="white",
               framealpha=0.9, loc="upper left")

    # ── (B) Stage-by-stage breakdown (single run) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 2])
    style_ax(ax1, "Stage Breakdown (Single Run, 10 Nodes)",
             "Scenario", "Latency (µs)")
    scenarios = ["Classical\n(no anon)", "Classical\n(anon)", "Quantum"]
    stage_data = {
        "Stage 1\nOverload detect": [
            single_run["c_na"]["T1_overload_detect"],
            single_run["c_a"]["T1_overload_detect"],
            single_run["q"]["T1_overload_detect"],
        ],
        "Stage 2\nNeighbour comm": [
            single_run["c_na"]["T2_neighbour_comm"],
            single_run["c_a"]["T2_neighbour_comm"],
            single_run["q"]["T2_neighbour_comm"],
        ],
        "Stage 3\nCandidate select": [
            single_run["c_na"]["T3_candidate_select"],
            single_run["c_a"]["T3_candidate_select"],
            single_run["q"]["T3_candidate_select"],
        ],
        "Stage 4\nDecision/migrate": [
            single_run["c_na"]["T4_decision_migrate"],
            single_run["c_a"]["T4_anon_rng_migrate"],
            single_run["q"]["T4_quantum_entangle_migrate"],
        ],
    }
    bottom = np.zeros(3)
    stage_colors = ["#9C27B0", "#FF9800", "#009688", "#E91E63"]
    x_pos = np.arange(3)
    for (label, vals), col in zip(stage_data.items(), stage_colors):
        ax1.bar(x_pos, vals, bottom=bottom, color=col, label=label,
                width=0.5, edgecolor="#0f0f1a", linewidth=0.5)
        bottom += np.array(vals)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(scenarios, color="#aaaacc", fontsize=7)
    ax1.legend(fontsize=6.5, facecolor="#111133", labelcolor="white", framealpha=0.9)

    # ── (C) Bell pair correlation histogram ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2, "Bell Pair Measurement: Node A vs Node B",
             "Measurement Outcome (Node A)", "Count")
    ra = bell["results_A"]
    rb = bell["results_B"]
    bins = [-0.5, 0.5, 1.5]
    ax2.hist(ra, bins=bins, color=COLORS["quantum"], alpha=0.7, label="Node A", rwidth=0.4)
    ax2.hist(rb, bins=bins, color=COLORS["classical_anon"], alpha=0.7, label="Node B",
             rwidth=0.4, align="right")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["0 (VNF1)", "1 (VNF2)"], color="#aaaacc", fontsize=8)
    ax2.legend(fontsize=7.5, facecolor="#111133", labelcolor="white")
    ax2.text(0.5, 0.88,
             f"Correlation: {bell['correlation_AB']:.3f}\nCollision rate: {bell['collision_rate']:.3f}",
             transform=ax2.transAxes, color="white", fontsize=8,
             ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222244", alpha=0.85))

    # ── (D) VNF assignment distribution from Bell pairs ───────────────
    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3, "VNF Assignment Distribution (Quantum, 1000 trials)",
             "Assignment", "Count")
    from collections import Counter
    assign_counts = Counter(bell["vnf_assignments"])
    labels_assign = ["A=VNF1\nB=VNF2", "A=VNF2\nB=VNF1"]
    counts_assign = [assign_counts.get(("VNF1", "VNF2"), 0),
                     assign_counts.get(("VNF2", "VNF1"), 0)]
    ax3.bar(labels_assign, counts_assign,
            color=[COLORS["quantum"], COLORS["classical_anon"]],
            width=0.4, edgecolor="#0f0f1a")
    ax3.text(0.5, 0.9,
             f"Each outcome ~50% → uniform randomness\nNo bias, no central knowledge",
             transform=ax3.transAxes, color="white", fontsize=8,
             ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#222244", alpha=0.85))

    # ── (E) Anonymity overhead: quantum advantage ─────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    style_ax(ax4, "Anonymity Overhead: Classical RNG vs Quantum Entanglement",
             "Number of Normal Nodes", "Extra Latency vs. Classical No-Anon (µs)")
    ns_arr = np.array(mc["n_nodes"])
    overhead_anon = np.array(mc["classical_anon"]["mean"]) - np.array(mc["classical_no_anon"]["mean"])
    overhead_q    = np.array(mc["quantum"]["mean"]) - np.array(mc["classical_no_anon"]["mean"])
    ax4.plot(ns_arr, overhead_anon, color=COLORS["classical_anon"],
             label="Classical anonymity overhead", linewidth=2)
    ax4.plot(ns_arr, overhead_q, color=COLORS["quantum"],
             label="Quantum overhead (vs classical, no anon)", linewidth=2, linestyle="--")
    ax4.axhline(0, color="white", linewidth=0.8, linestyle=":")
    ax4.legend(fontsize=7.5, facecolor="#111133", labelcolor="white")

    # ── (F) Sensitivity: τ_CN_inode_base ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    style_ax(ax5, "Sensitivity: CN↔Node Base Link Latency",
             "τ_CN_inode_base (µs)", "Mean Total Latency (µs)")
    for k in ["classical_no_anon", "classical_anon", "quantum"]:
        ax5.plot(sens_link["param_values"], sens_link[k],
                 color=COLORS[k], linewidth=2, label=LABELS[k][:12] + "…")
    ax5.legend(fontsize=6.5, facecolor="#111133", labelcolor="white")

    # ── (G) Sensitivity: τ_CN_EP (entanglement generation time) ───────
    ax6 = fig.add_subplot(gs[2, 1])
    style_ax(ax6, "Sensitivity: Entanglement Generation Time (τ_CN_EP)",
             "τ_CN_EP (µs)", "Mean Total Latency (µs)")
    ax6.plot(sens_ep["param_values"], sens_ep["classical_no_anon"],
             color=COLORS["classical_no_anon"], linewidth=2, linestyle="--",
             label="Classical no-anon (flat baseline)")
    ax6.plot(sens_ep["param_values"], sens_ep["classical_anon"],
             color=COLORS["classical_anon"], linewidth=2, linestyle="--",
             label="Classical anon (flat baseline)")
    ax6.plot(sens_ep["param_values"], sens_ep["quantum"],
             color=COLORS["quantum"], linewidth=2,
             label="Quantum (grows with τ_CN_EP)")
    ax6.legend(fontsize=6.5, facecolor="#111133", labelcolor="white")
    ax6.text(0.5, 0.15,
             "Quantum advantage disappears if\nentanglement generation is too slow",
             transform=ax6.transAxes, color="#ffcc44", fontsize=7.5,
             ha="center", bbox=dict(boxstyle="round,pad=0.3", facecolor="#222244", alpha=0.85))

    # ── (H) Summary table ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor("#1a1a2e")
    ax7.axis("off")
    ax7.set_title("Summary: Key Properties", color="white", fontsize=10, pad=6)

    p_default = NetworkParams()
    rng_tmp = np.random.default_rng(99)
    sr_q  = np.mean([latency_quantum(p_default, rng_tmp)["T_total"]           for _ in range(300)])
    sr_ca = np.mean([latency_classical_anon(p_default, rng_tmp)["T_total"]    for _ in range(300)])
    sr_cn = np.mean([latency_classical_no_anon(p_default, rng_tmp)["T_total"] for _ in range(300)])

    table_data = [
        ["Property",                  "Classical\n(no anon)", "Classical\n(anon)", "Quantum"],
        ["Anonymity",                 "✗ No",                 "✓ Yes (RNG)",       "✓ Yes (native)"],
        ["Extra resource needed",     "—",                    "3rd-party RNG",     "Entangl. source"],
        ["Decision made by",          "Central node",         "RNG",               "Bell measurement"],
        ["Latency (µs, 10 nodes)",    f"{sr_cn:.1f}",         f"{sr_ca:.1f}",      f"{sr_q:.1f}"],
        ["Quantum overhead",          "—",                    "—",                 "τ_EP + τ_meas"],
        ["Scalable to N nodes?",      "✓",                    "✓",                 "Partial (GHZ)"],
    ]

    col_colors = [
        ["#222244"] * 4,
        ["#222244", "#113344", "#442222", "#113322"],
        ["#222244", "#113344", "#442222", "#113322"],
        ["#222244", "#113344", "#442222", "#113322"],
        ["#222244", "#113344", "#442222", "#113322"],
        ["#222244", "#113344", "#442222", "#113322"],
        ["#222244", "#113344", "#442222", "#113322"],
    ]

    tbl = ax7.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center", loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2a2a5a")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor(col_colors[row][col])
            cell.set_text_props(color="white")
        cell.set_edgecolor("#333366")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    print(f"[✓] Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────
# 6.  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("VNF Migration: Classical vs Quantum Latency Simulation")
    print("=" * 65)

    # Default parameters
    p = NetworkParams()
    rng = np.random.default_rng(42)

    # ── Single-run stage breakdown ──────────────────────────────────
    print("\n[1] Single-run latency breakdown (10 normal nodes):")
    p.n_normal_nodes = 10
    c_na = latency_classical_no_anon(p, rng)
    c_a  = latency_classical_anon(p, rng)
    q    = latency_quantum(p, rng)

    for name, d in [("Classical (no anon)", c_na),
                    ("Classical (anon)",    c_a),
                    ("Quantum",             q)]:
        print(f"\n  {name}:")
        for k, v in d.items():
            print(f"    {k:40s}: {v:.2f} µs")

    # ── Bell pair simulation ────────────────────────────────────────
    print("\n[2] Bell pair quantum decision simulation (1000 trials):")
    bell = simulate_bell_pair_decision(n_trials=1000, seed=7)
    print(f"    Correlation A↔B     : {bell['correlation_AB']:.4f}  (expected ≈ -1.0)")
    print(f"    Collision rate       : {bell['collision_rate']:.4f}  (expected = 0.0)")
    print(f"    VNF1 at Node A       : {bell['vnf1_at_A_fraction']:.3f}  (expected ≈ 0.5)")

    # ── Monte Carlo ─────────────────────────────────────────────────
    print("\n[3] Monte Carlo simulation (500 trials × 25 network sizes)...")
    mc = run_monte_carlo(n_nodes_range=list(range(2, 52, 2)), n_trials=500, seed=0)
    print("    Done.")

    # Print a summary table at 10/20/50 nodes
    print("\n    n_nodes | Classical (no anon) | Classical (anon) | Quantum")
    print("    " + "-" * 62)
    for n_target in [10, 20, 50]:
        idx = mc["n_nodes"].index(n_target)
        print(f"    {n_target:7d} | "
              f"{mc['classical_no_anon']['mean'][idx]:17.2f}   | "
              f"{mc['classical_anon']['mean'][idx]:14.2f}   | "
              f"{mc['quantum']['mean'][idx]:.2f}")

    # ── Sensitivity analyses ────────────────────────────────────────
    print("\n[4] Sensitivity analysis: τ_CN_inode_base (link latency)...")
    p.n_normal_nodes = 10  # reset
    sens_link = run_sensitivity("tau_CN_inode_base", np.linspace(0.5, 10.0, 30), seed=3)
    print("    Done.")

    print("\n[5] Sensitivity analysis: τ_CN_EP (entanglement generation time)...")
    sens_ep = run_sensitivity("tau_CN_EP", np.linspace(0.1, 8.0, 30), seed=5)
    print("    Done.")

    # ── Plot everything ─────────────────────────────────────────────
    print("\n[6] Generating combined figure...")
    single_run = {"c_na": c_na, "c_a": c_a, "q": q}
    plot_all(mc, bell, sens_link, sens_ep, single_run)

    # ── Quantum advantage summary ───────────────────────────────────
    print("\n" + "=" * 65)
    print("QUANTUM ADVANTAGE SUMMARY")
    print("=" * 65)
    idx10 = mc["n_nodes"].index(10)
    q_lat  = mc["quantum"]["mean"][idx10]
    ca_lat = mc["classical_anon"]["mean"][idx10]
    cn_lat = mc["classical_no_anon"]["mean"][idx10]

    saving_vs_anon = ca_lat - q_lat
    overhead_vs_no_anon = q_lat - cn_lat

    print(f"  At 10 nodes:")
    print(f"    Classical (no anon)  : {cn_lat:.2f} µs")
    print(f"    Classical (anon)     : {ca_lat:.2f} µs")
    print(f"    Quantum              : {q_lat:.2f} µs")
    print(f"")
    print(f"    Quantum saves {saving_vs_anon:.2f} µs vs classical-anon (same security level)")
    print(f"    Quantum overhead vs classical-no-anon: {overhead_vs_no_anon:.2f} µs")
    print(f"")
    print(f"  Bell-pair mechanism guarantees:")
    print(f"    ✓ Zero collision rate ({bell['collision_rate']:.0f})")
    print(f"    ✓ Perfect anti-correlation ({bell['correlation_AB']:.4f})")
    print(f"    ✓ Uniform VNF assignment (no bias)")
    print(f"    ✓ Central node anonymity — no 3rd-party RNG needed")
    print("=" * 65)
    print("\n[✓] Simulation complete.")


if __name__ == "__main__":
    main()
