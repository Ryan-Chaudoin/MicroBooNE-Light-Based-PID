#Physics helper functions for LY simulation

import numpy as np

# Mass stopping powers in MeV cm^2 / g.
MUON_KE = np.array([10, 14, 20, 30, 40, 80, 100, 140, 200, 300, 400, 800, 1000, 1400, 2000])
MUON_DEDX = np.array([5.687, 4.461, 3.503, 2.731, 2.340, 1.771, 1.670, 1.570, 1.519, 1.510, 1.526, 1.610, 1.645, 1.700, 1.761])

PROTON_KE = np.array([10, 12.5, 15, 17.5, 20, 25, 27.5, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
PROTON_DEDX = np.array([30.61, 25.81, 22.42, 19.89, 17.93, 15.07, 13.99, 13.07, 11.59, 10.44, 9.529, 8.783, 8.162, 7.636, 7.184, 6.792, 6.449, 6.145, 5.874, 5.631, 5.412, 5.213, 4.445, 3.919, 3.537, 3.246, 3.018, 2.834, 2.683, 2.557, 2.358, 2.210, 2.095, 2.004, 1.931, 1.871, 1.821, 1.779, 1.744, 1.714, 1.688, 1.665, 1.646, 1.629])

LAR_DENSITY = 1.38
ALPHA = 0.21
A_BOX = 0.800
K_BOX = 0.0486e3
E_FIELD = 273.0
W_PH = 19.5e-6


def interp_dedx(ke_mev, particle="proton"):
    #Interpolate stopping power and convert it to MeV/cm in liquid argon
    energies, values = (PROTON_KE, PROTON_DEDX) if particle == "proton" else (MUON_KE, MUON_DEDX)
    if ke_mev <= energies[0]:
        slope = (values[1] - values[0]) / (energies[1] - energies[0])
        value = max(values[0] + slope * (ke_mev - energies[0]), 0.0)
    else:
        value = max(float(np.interp(ke_mev, energies, values)), 0.0)
    return value * LAR_DENSITY


def recombination_factor(dedx_mev_cm):
    # The Box denominator describes the fraction of ionization charge that
    # escapes recombination. The complementary fraction produces light.
    charge_survival = A_BOX / (1.0 + K_BOX * dedx_mev_cm / E_FIELD)
    return 1.0 - charge_survival


def photons_per_mev(dedx_mev_cm):
    #Calculate scintillation photons per deposited MeV
    return (ALPHA + recombination_factor(dedx_mev_cm)) / ((1.0 + ALPHA) * W_PH)


def simulate_yield(ke_mev, particle="proton", delta_e=0.1):
    #Integrate the photon yield while a particle loses all of its kinetic energy"
    remaining = float(ke_mev)
    total = 0.0
    while remaining > 0:
        deposited = min(delta_e, remaining)
        total += photons_per_mev(interp_dedx(remaining, particle)) * deposited
        remaining -= deposited
    return total


def bragg_profile(ke_mev, particle="proton", n_bins=30):
    #Return normalized emission weights along a stopping track
    remaining = float(ke_mev)
    if remaining <= 0:
        return np.ones(n_bins) / n_bins

    step = max(remaining / 500.0, 0.01)
    distance = [0.0]
    stopping_power = []
    while remaining > 0:
        deposited = min(step, remaining)
        dedx = interp_dedx(remaining, particle)
        if dedx <= 0:
            break
        distance.append(distance[-1] + deposited / dedx)
        stopping_power.append(dedx)
        remaining -= deposited

    if not stopping_power:
        return np.ones(n_bins) / n_bins
    distance = np.asarray(distance)
    stopping_power = np.asarray(stopping_power)
    centers = (np.arange(n_bins) + 0.5) / n_bins * distance[-1]
    profile = np.interp(centers, 0.5 * (distance[:-1] + distance[1:]), stopping_power)
    return profile / (profile.sum() + 1e-12)
