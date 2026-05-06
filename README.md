# VNF_DecisionMaking
# Entanglement-Assisted Decision Making for VNF Migration in 6G Networks

A Python simulation comparing **classical** and **quantum entanglement-based** approaches to binary decision making for Virtual Network Function (VNF) migration, inspired by:

> S. Maheshwari, V. Raman, R. Bassoli, and F. H. P. Fitzek, "Entanglement-assisted decision making for VNF migration in 6G Communication Networks," *2024 IEEE Conference on Network Function Virtualization and Software Defined Networks (NFV-SDN)*, pp. 227–229, 2024. DOI: [10.1109/NFV-SDN61811.2024.10807502](https://doi.org/10.1109/NFV-SDN61811.2024.10807502)

---

## Overview

Network Function Virtualization (NFV) is essential for network slicing in 6G, but it introduces additional latency. When a central node needs to migrate VNFs to neighbouring nodes, it must make a binary decision — which VNF goes where — while potentially preserving **anonymity** (the central node shouldn't know the assignment until the nodes confirm).

This simulation models and compares three approaches:

| Approach | Anonymity | Decision Mechanism |
|---|---|---|
| **Classical (no anonymity)** | ✗ | Central node decides directly |
| **Classical (with anonymity)** | ✓ | Outsourced to a trusted 3rd-party RNG |
| **Quantum (entanglement)** | ✓ (inherent) | Bell pair measurement at destination nodes |

The quantum approach leverages the Bell state |Ψ⁺⟩ = (1/√2)(|10⟩ + |01⟩). When each node measures its entangled photon, the outcomes are guaranteed to be anti-correlated — ensuring complementary VNF assignment with zero collisions and no need for a third-party random source.

## Features

- **Latency modelling** for all three decision regimes, following the analytical expressions from the paper (Fig. 3)
- **Bell pair simulation** verifying anti-correlation, zero collision rate, and uniform assignment
- **Monte Carlo analysis** across varying network sizes (2–50 nodes, 500 trials each)
- **Sensitivity analysis** on key parameters (link latency, entanglement generation time)
- **Combined 8-panel visualisation** summarising all results

## Repository Structure

```
.
├── vnf_migration_sim.py    # Main simulation script
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/vnf-migration-quantum-sim.git
cd vnf-migration-quantum-sim
pip install -r requirements.txt
```

### Running the Simulation

```bash
python vnf_migration_sim.py
```

This will:
1. Run a single-run latency breakdown (10 nodes)
2. Simulate 1 000 Bell pair measurements
3. Execute the Monte Carlo analysis (500 trials × 25 network sizes)
4. Perform sensitivity sweeps on link latency and entanglement generation time
5. Generate and save an 8-panel figure as `vnf_migration_simulation.png`
6. Print a summary of quantum vs classical latency advantages

### Example Output

```
VNF Migration: Classical vs Quantum Latency Simulation
=================================================================

  At 10 nodes:
    Classical (no anon)  : 23.50 µs
    Classical (anon)     : 24.50 µs
    Quantum              : 19.80 µs

    Quantum saves ~4.7 µs vs classical-anon (same security level)

  Bell-pair mechanism guarantees:
    ✓ Zero collision rate
    ✓ Perfect anti-correlation (-1.0)
    ✓ Uniform VNF assignment (no bias)
    ✓ Central node anonymity — no 3rd-party RNG needed
```

*(Exact numbers depend on random seed and default parameter values.)*

## Simulation Parameters

All tunable parameters are in the `NetworkParams` dataclass. Key defaults (in µs):

| Parameter | Default | Description |
|---|---|---|
| `n_normal_nodes` | 10 | Number of neighbouring normal nodes |
| `tau_CN_proc_ov` | 5.0 | Central node overload detection time |
| `tau_CN_inode_base` | 2.0 | Base CN → node communication latency |
| `tau_CN_EP` | 0.8 | Bell pair generation time |
| `tau_jnode_meas` | 0.3 | Quantum measurement time at node |
| `tau_CN_jnode_migrate` | 10.0 | Actual VNF migration transfer latency |

You can modify these in the script or pass different values programmatically.

## How It Works

### Classical Solution
1. Central node detects overload
2. Broadcasts status request to all neighbours; collects replies
3. Selects two candidate nodes (A, B)
4. Either decides the assignment directly (no anonymity) or outsources to a 3rd-party RNG (with anonymity), then communicates the decision and initiates migration

### Quantum Solution
1. Steps 1–3 are identical to the classical approach
2. Central node generates a Bell pair and distributes one photon to each candidate node
3. Each node measures its photon — outcomes are automatically anti-correlated
4. Nodes know their assignment *before* the central node does (inherent anonymity)

## Citation

If you use this code in your work, please cite the original paper:

```bibtex
@inproceedings{maheshwari2024entanglement,
  author    = {Maheshwari, Shivam and Raman, Vignesh and Bassoli, Riccardo and Fitzek, Frank H. P.},
  title     = {Entanglement-assisted decision making for {VNF} migration in {6G} Communication Networks},
  booktitle = {2024 IEEE Conference on Network Function Virtualization and Software Defined Networks (NFV-SDN)},
  pages     = {227--229},
  year      = {2024},
  doi       = {10.1109/NFV-SDN61811.2024.10807502}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

This simulation is based on research conducted at the Deutsche Telekom Chair of Communication Networks, TU Dresden, and the Centre for Tactile Internet with Human-in-the-Loop (CeTI). The original work was funded by the German Research Foundation (DFG) under Germany's Excellence Strategy (EXC 2050/1, Project ID 390696704) and the Federal Ministry of Education and Research (BMBF) through the 6G-life project (16KISK001K).

