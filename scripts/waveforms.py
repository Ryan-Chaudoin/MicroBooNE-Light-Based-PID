#Prompt/late light helpers for waveforms (data and MC)

import numpy as np


def waveform_prompt_fraction(waveform, prompt_ticks=13, total_ticks=64, baseline_ticks=100):
    #Find pulse -> integrate prompt and total windows -> give ratio
    waveform = np.asarray(waveform, dtype=float)
    if len(waveform) < baseline_ticks + total_ticks:
        return np.nan
    signal = waveform - np.mean(waveform[:baseline_ticks])
    peak = int(np.argmax(signal))
    if peak + total_ticks > len(signal):
        return np.nan
    total = np.sum(signal[peak:peak + total_ticks])
    if total <= 0:
        return np.nan
    prompt = np.sum(signal[peak:peak + prompt_ticks])
    return float(prompt / total)


def event_prompt_fraction(row, pmt_indices, prompt_ticks=13, total_ticks=64, baseline_ticks=100):
    #Average the waveform prompt fraction over selected PMTs for one event
    values = []
    for pmt in pmt_indices:
        key = f"waveform_{pmt:02d}"
        if key in row:
            value = waveform_prompt_fraction(row[key], prompt_ticks, total_ticks, baseline_ticks)
            if np.isfinite(value):
                values.append(value)
    return float(np.mean(values)) if values else np.nan


def truth_prompt_fraction(simulated_photon_times, prompt_window_ns=100.0):
    #Calculate a prompt fraction from simulated photon arrival times
    times = np.asarray(simulated_photon_times, dtype=float)
    valid = times[times >= 0]
    return float(np.mean(valid < prompt_window_ns)) if len(valid) else np.nan
