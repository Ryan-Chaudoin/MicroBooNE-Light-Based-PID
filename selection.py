#single-track proton/muon selection

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SelectionConfig:
    proton_pid_max: float = 0.0
    muon_pid_min: float = 0.2
    track_length_max_cm: float = 100.0
    flash_distance_max_cm: float = 200.0
    flash_zwidth_sigma_max: float = 1.5
    hip_limits: Tuple[Tuple[str, float], ...] = (
        ("ng2hip_r1cm", 10.0),
        ("ng2hip_r3cm", 10.0),
        ("ng2hip_r5cm", 15.0),
        ("ng2hip_r10cm", 15.0),
    )


FIDUCIAL = {
    "x": (10.0, 250.0),
    "y": (-105.0, 105.0),
    "z": (30.0, 1000.0),
}


def scalar(value: object) -> float:
    #Return the first scalar from a ROOT scalar/one-element vector value
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return np.nan
        return float(value[0])
    return float(value)


def scalar_series(series: pd.Series) -> pd.Series:
    return series.map(scalar).astype(float)


def _track_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for axis in "xyz":
        result[f"start_{axis}"] = scalar_series(result[f"trk_sce_start_{axis}_v"])
        result[f"end_{axis}"] = scalar_series(result[f"trk_sce_end_{axis}_v"])
    result["track_length_cm"] = np.sqrt((result["start_x"] - result["end_x"]) ** 2 + (result["start_y"] - result["end_y"]) ** 2 + (result["start_z"] - result["end_z"]) ** 2)
    result["z_fraction"] = ( -(result["start_z"] - result["end_z"]) / result["track_length_cm"].clip(lower=1e-9))
    result["midpoint_x"] = (result["start_x"] + result["end_x"]) / 2.0
    result["midpoint_y"] = (result["start_y"] + result["end_y"]) / 2.0
    result["midpoint_z"] = (result["start_z"] + result["end_z"]) / 2.0
    result["flash_y"] = scalar_series(result["flash_y_flash_matching"])
    result["flash_z"] = scalar_series(result["flash_z_flash_matching"])
    result["flash_zwidth"] = scalar_series(result["flash_zwidth_flash_matching"])
    result["flash_pe"] = scalar_series(result["flash_pe_flash_matching"])
    result["charge_flash_y"] = (result["nu_centerY"] - result["flash_y"]).abs()
    result["charge_flash_z"] = (result["nu_centerZ"] - result["flash_z"]).abs()
    result["flash_distance_cm"] = np.sqrt((result["flash_z"] - result["midpoint_z"]) ** 2 + (result["flash_y"] - result["midpoint_y"]) ** 2)
    result["flash_zwidth_sigma"] = ((result["flash_z"] - result["midpoint_z"]).abs() / result["flash_zwidth"].replace(0, np.nan))
    return result


def _base_mask(df: pd.DataFrame) -> pd.Series:
    return (df["nslice"] == 1) & (df["n_tracks"] == 1)


def _fiducial_mask(df: pd.DataFrame) -> pd.Series:
    return (
        df["start_x"].between(*FIDUCIAL["x"])
        & df["start_y"].between(*FIDUCIAL["y"])
        & df["start_z"].between(*FIDUCIAL["z"])
        & df["end_x"].between(*FIDUCIAL["x"])
        & df["end_y"].between(*FIDUCIAL["y"])
        & df["end_z"].between(*FIDUCIAL["z"])
    )


def _cut_masks(df: pd.DataFrame, particle: str, config: SelectionConfig) -> List[Tuple[str, pd.Series]]:
    pid = scalar_series(df["trk_llr_pid_score_v"])
    masks: List[Tuple[str, pd.Series]] = [("single_track", _base_mask(df))]
    masks.append(("pid", pid < config.proton_pid_max if particle == "proton" else pid > config.muon_pid_min))
    masks.append(("fiducial_volume", _fiducial_mask(df)))
    masks.append(("positive_flash", df["flash_pe"] > 0))
    masks.append(("track_length", df["track_length_cm"] < config.track_length_max_cm))
    masks.append(("forward_z", df["z_fraction"] > 0.4))
    masks.append(
        ("flash_track_match", (df["flash_distance_cm"] < config.flash_distance_max_cm)
         & (df["flash_zwidth_sigma"] <= config.flash_zwidth_sigma_max))
    )
    masks.append(("charge_flash_match", (df["charge_flash_y"] < 50) & (df["charge_flash_z"] < 50)))
    if particle == "muon":
        for column, limit in config.hip_limits:
            masks.append((column, df[column] < limit))
    return masks


def cutflow(df: pd.DataFrame, particle: str, config: SelectionConfig | None = None) -> pd.DataFrame:
    #Return cumulative counts, efficiencies, and truth purity
    if particle not in {"proton", "muon"}:
        raise ValueError("particle must be 'proton' or 'muon'")
    config = config or SelectionConfig(
        track_length_max_cm=100.0 if particle == "proton" else 300.0,
        flash_distance_max_cm=200.0 if particle == "proton" else 160.0,
        flash_zwidth_sigma_max=1.5 if particle == "proton" else 1.0,
    )
    work = _track_columns(df)
    running = np.ones(len(work), dtype=bool)
    total = len(work)
    rows = []
    for name, mask in _cut_masks(work, particle, config):
        running &= mask.fillna(False).to_numpy()
        selected = int(running.sum())
        row = {"cut": name, "events": selected, "efficiency": selected / total if total else np.nan}
        if "truth_signal" in work:
            truth = work["truth_signal"].to_numpy(dtype=bool)
            row["purity"] = float(truth[running].mean()) if selected else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, particle: str, config: SelectionConfig | None = None) -> pd.DataFrame:
    #Apply the cumulative selection and return a copy of selected rows
    flow = cutflow(df, particle, config)
    work = _track_columns(df)
    running = np.ones(len(work), dtype=bool)
    config = config or SelectionConfig(
        track_length_max_cm=100.0 if particle == "proton" else 300.0,
        flash_distance_max_cm=200.0 if particle == "proton" else 160.0,
        flash_zwidth_sigma_max=1.5 if particle == "proton" else 1.0,
    )
    for _, mask in _cut_masks(work, particle, config):
        running &= mask.fillna(False).to_numpy()
    return work.loc[running].copy()


def add_truth_signal(df: pd.DataFrame, particle: str) -> pd.DataFrame:
    #Add a conservative pure-particle truth label when truth branches exist
    result = df.copy()
    if particle == "proton":
        result["truth_signal"] = (
            (result["nproton"] == 1) & (result["nmuon"] == 0) & (result["nelec"] == 0) & (result["npion"] == 0) & (result["npi0"] == 0)
        )
    else:
        result["truth_signal"] = (
            (result["nmuon"] == 1) & (result["nproton"] == 0) & (result["nelec"] == 0) & (result["npion"] == 0) & (result["npi0"] == 0)
        )
    return result
