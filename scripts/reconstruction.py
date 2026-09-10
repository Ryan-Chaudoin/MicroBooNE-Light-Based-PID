#Photon yield reconstruction helpers 

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import uproot

from physics import bragg_profile
from selection import scalar

# MicroBooNE PMT positions in detector coordinates, in cm.
PMT_POSITIONS = np.array([
    [-11.6415, 55.313, 951.861], [-11.8345, 55.822, 911.065], [-11.4175, 27.607, 989.712],
    [-12.1765, -0.722, 865.599], [-11.4545, -28.625, 990.356], [-11.7755, -56.514, 951.865],
    [-12.0585, -56.309, 911.939], [-12.5405, 55.625, 751.884], [-12.6615, 55.8, 711.073],
    [-12.3045, -0.502, 796.208], [-12.6245, -0.051, 664.203], [-12.6045, -56.284, 751.905],
    [-12.6125, -56.408, 711.274], [-12.8735, 55.822, 540.929], [-12.9835, 55.771, 500.134],
    [-12.6515, -0.549, 585.284], [-12.6185, -0.875, 453.096], [-12.6205, -56.205, 540.616],
    [-12.5945, -56.323, 500.221], [-13.1865, 54.693, 328.212], [-13.4175, 54.646, 287.976],
    [-13.0855, -0.706, 373.839], [-13.1505, -0.829, 242.014], [-12.6485, -57.022, 328.341],
    [-13.0075, -56.261, 287.639], [-13.3965, 55.249, 128.354], [-13.5415, 55.249, 87.761],
    [-13.4345, 27.431, 51.102], [-13.4415, -0.303, 173.743], [-13.1525, -28.576, 50.475],
    [-13.2784, -56.203, 128.180], [-13.2375, -56.615, 87.870],
])

X_MIN, X_MAX = -64.825, 321.175
Y_MIN, Y_MAX = -193.0, 193.0
Z_MIN, Z_MAX = -127.24, 1164.24
NX, NY, NZ = 75, 75, 400
DX, DY, DZ = (X_MAX - X_MIN) / NX, (Y_MAX - Y_MIN) / NY, (Z_MAX - Z_MIN) / NZ

RECONSTRUCTION_BRANCHES = [
    "nslice", "n_tracks", "trk_llr_pid_score_v",
    "trk_sce_start_x_v", "trk_sce_start_y_v", "trk_sce_start_z_v",
    "trk_sce_end_x_v", "trk_sce_end_y_v", "trk_sce_end_z_v",
    "flash_pe_flash_matching", "flash_pe_flash_matching_v",
    "flash_z_flash_matching", "flash_y_flash_matching", "flash_zwidth_flash_matching",
    "nu_centerY", "nu_centerZ", "trk_energy_proton_v", "trk_energy_muon",
    "ng2hip_r1cm", "ng2hip_r3cm", "ng2hip_r5cm", "ng2hip_r10cm",
    "nmuon", "nproton", "nelec", "npion", "npi0",
]


def load_reconstruction_frame(path):
    # Load only the branches needed by selection, calibration, and reconstruction.
    root_path = Path(path) if str(path).endswith(".root") else Path(f"{path}.root")
    tree = uproot.open(root_path)["nuselection"]["NeutrinoSelectionFilter"]
    available = [name for name in RECONSTRUCTION_BRANCHES if name in tree.keys()]
    frame = pd.DataFrame(tree.arrays(available, library="np"))
    return frame[frame["trk_llr_pid_score_v"].map(len) == 1].copy()


def load_reconstruction_frames(paths):
    # Combine several data files while preserving the original event rows.
    return pd.concat([load_reconstruction_frame(path) for path in paths], ignore_index=True)


def load_visibility(library_path, cache_path=None):
    #Load the visibility library, caching the voxels
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)

    tree = uproot.open(library_path)["pmtresponse"]["PhotonLibraryData"]
    table = pd.DataFrame(tree.arrays(library="np"))
    visibility = {}
    for _, row in table.iterrows():
        voxel = int(row["Voxel"])
        pmt = int(row["OpChannel"])
        visibility.setdefault(voxel, np.zeros(32))[pmt] = float(row["Visibility"])

    if cache_path:
        with open(cache_path, "wb") as handle:
            pickle.dump(visibility, handle)
    return visibility


def coordinates_to_voxel(x, y, z):
    #Convert detector coordinates to the visibility-library voxel index
    ix, iy, iz = int((x - X_MIN) / DX), int((y - Y_MIN) / DY), int((z - Z_MIN) / DZ)
    if not (0 <= ix < NX and 0 <= iy < NY and 0 <= iz < NZ):
        raise ValueError("track point lies outside the visibility-library volume")
    return iz * (NX * NY) + iy * NX + ix


def emission_weighted_visibility(start, end, particle, ke_mev, visibility, n_bins=30):
    #Average visibility along a track using the particle's bragg profile
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    profile = bragg_profile(ke_mev, particle, n_bins)
    points = start + np.outer((np.arange(n_bins) + 0.5) / n_bins, end - start)
    weighted = np.zeros(32)
    for weight, point in zip(profile, points):
        try:
            weighted += weight * visibility.get(coordinates_to_voxel(*point), np.zeros(32))
        except ValueError:
            # A small number of edge tracks can sample outside the library.
            continue
    return weighted


def reconstruct_event(start, end, pe, calibration, visibility, ke_mev, particle, pe_cut=8, n_bins=30):
    #Return total emitted photons and the number of PMTs used for one event
    if ke_mev <= 0:
        return np.nan, 0
    weighted = emission_weighted_visibility(start, end, particle, ke_mev, visibility, n_bins)
    midpoint = 0.5 * (np.asarray(start) + np.asarray(end))
    estimates = []
    for index, photoelectrons in enumerate(np.asarray(pe, dtype=float)):
        if photoelectrons < pe_cut or weighted[index] <= 0:
            continue
        if not np.isfinite(calibration[index]) or calibration[index] <= 0:
            continue
        distance = np.linalg.norm(PMT_POSITIONS[index] - midpoint)
        if photoelectrons > 5.0 * distance ** 2:
            continue
        estimates.append(photoelectrons / (weighted[index] * calibration[index]))
    if not estimates:
        return np.nan, 0
    estimates = np.asarray(estimates)
    if len(estimates) < 5:
        return float(np.median(estimates)), len(estimates)
    low, high = np.percentile(estimates, [10, 90])
    kept = estimates[(estimates >= low) & (estimates <= high)]
    return float(np.mean(kept)), len(kept)


def derive_pmt_calibration(muon_frame, visibility, pe_cut=8, n_bins=30,
                           ke_low_mev=400, ke_high_mev=600, min_events=10):
    # Use a relatively flat 400-600 MeV muon sample to compare PMT response.
    pmt_values = [[] for _ in range(32)]
    for _, row in muon_frame.iterrows():
        start, end, pe, ke_mev = track_row(row, "muon")
        if not (ke_low_mev < ke_mev < ke_high_mev):
            continue
        weighted = emission_weighted_visibility(start, end, "muon", ke_mev, visibility, n_bins)
        for pmt, photoelectrons in enumerate(pe):
            if photoelectrons >= pe_cut and weighted[pmt] > 0:
                pmt_values[pmt].append(photoelectrons / (weighted[pmt] * ke_mev))

    medians = np.array([
        np.median(values) if len(values) >= min_events else np.nan
        for values in pmt_values
    ])
    return medians / np.nanmean(medians)


def track_row(row, particle):
    start = [scalar(row[f"trk_sce_start_{axis}_v"]) for axis in "xyz"]
    end = [scalar(row[f"trk_sce_end_{axis}_v"]) for axis in "xyz"]
    pe = np.asarray(row["flash_pe_flash_matching_v"], dtype=float)
    energy_key = "trk_energy_proton_v" if particle == "proton" else "trk_energy_muon"
    ke_mev = scalar(row[energy_key]) * 1000.0
    return start, end, pe, ke_mev
