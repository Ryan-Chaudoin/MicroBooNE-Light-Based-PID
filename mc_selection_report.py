"""Generate MC selection and contamination summaries.

Example:
    python mc_selection_report.py data/simulated/sample --output reports/mc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from selection import add_truth_signal, cutflow, select

BRANCHES = [
    "nslice", "n_tracks", "trk_llr_pid_score_v",
    "trk_sce_start_x_v", "trk_sce_start_y_v", "trk_sce_start_z_v",
    "trk_sce_end_x_v", "trk_sce_end_y_v", "trk_sce_end_z_v",
    "flash_pe_flash_matching", "flash_z_flash_matching", "flash_y_flash_matching",
    "flash_zwidth_flash_matching", "nu_centerY", "nu_centerZ",
    "ng2hip_r1cm", "ng2hip_r3cm", "ng2hip_r5cm", "ng2hip_r10cm",
    "nmuon", "nproton", "nelec", "npion", "npi0",
    "muon_e", "proton_e", "elec_e", "pion_e", "pi0_e",
]


def load_frame(path: Path) -> pd.DataFrame:
    root_path = path if path.suffix == ".root" else path.with_suffix(".root")
    tree = uproot.open(root_path)["nuselection"]["NeutrinoSelectionFilter"]
    available = [name for name in BRANCHES if name in tree.keys()]
    frame = pd.DataFrame(tree.arrays(available, library="np"))
    return frame[frame["trk_llr_pid_score_v"].map(len) == 1].copy()


def _energy(value: object, mass: float) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        value = value[0] if len(value) else 0.0
    return max(float(value) - mass, 0.0)


def contamination_summary(frame: pd.DataFrame, selected: pd.DataFrame, particle: str) -> pd.DataFrame:
    if particle == "proton":
        signal = selected["proton_e"].map(lambda value: _energy(value, 0.938))
    else:
        signal = selected["muon_e"].map(lambda value: _energy(value, 0.105))
    columns = {
        "muon_e": 0.105, "proton_e": 0.938, "elec_e": 0.000511,
        "pion_e": 0.140, "pi0_e": 0.135,
    }
    total = sum(selected[column].map(lambda value, mass=mass: _energy(value, mass)) for column, mass in columns.items())
    output = selected[["nmuon", "nproton", "nelec", "npion", "npi0"]].copy()
    output["signal_ke_mev"] = signal
    output["total_visible_ke_mev"] = total
    output["contamination_fraction"] = 1.0 - signal / total.replace(0, np.nan)
    output["particle"] = particle
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="ROOT file or file stem")
    parser.add_argument("--output", type=Path, default=Path("reports/mc_selection"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.input)
    report = {}
    for particle in ("proton", "muon"):
        labeled = add_truth_signal(frame, particle)
        flow = cutflow(labeled, particle)
        selected = select(labeled, particle)
        contamination = contamination_summary(frame, selected, particle)
        flow.to_csv(args.output / f"{particle}_cutflow.csv", index=False)
        contamination.to_csv(args.output / f"{particle}_contamination.csv", index=False)
        report[particle] = {"input_events": len(frame), "selected_events": len(selected)}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(contamination["contamination_fraction"].dropna(), bins=30, range=(0, 1), color="#356f8f")
        axes[0].set(xlabel="Contamination fraction", ylabel="Events", title=f"{particle.title()} candidates")
        axes[1].hist2d(contamination["signal_ke_mev"], contamination["total_visible_ke_mev"], bins=30, cmap="viridis")
        axes[1].set(xlabel="Signal KE (MeV)", ylabel="Total visible KE (MeV)")
        fig.tight_layout()
        fig.savefig(args.output / f"{particle}_contamination.png", dpi=160)
        plt.close(fig)

    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
