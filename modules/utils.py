import numpy as np


def compute_peaks_fraction(peak_values):
    peak_values_sum = np.sum(peak_values, axis=-1)
    peak_values_sum = np.repeat(peak_values_sum.reshape(peak_values_sum.shape + (1,)),
                                peak_values.shape[-1], axis=-1)
    peaks_fraction = peak_values / peak_values_sum
    return peaks_fraction


def nb_peaks_factor(delta_m_max_fct, peak_fraction):
    nb_peaks_factor = delta_m_max_fct(peak_fraction)
    return np.clip(nb_peaks_factor, 0, 1)