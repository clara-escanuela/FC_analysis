import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import convolve1d
from ctapipe.image.extractor import deconvolve, deconvolution_parameters
import astropy.units as u
from ctapipe.io import EventSource
import pandas as pd

def ctapipe_subarray(tel_id=10, url="dataset://gamma_prod5.simtel.zst"):
    source = EventSource(url)
    subarray = source.subarray
    subarray = subarray.select_subarray([tel_id])

    return subarray

def __filtfilt_fast(signal, filt):
    """
    Apply a linear filter forward and backward to a signal, based on scipy.signal.filtfilt.
    filtfilt has some speed issues (https://github.com/scipy/scipy/issues/17080)
    """
    forward = convolve1d(signal, filt, axis=-1, mode="nearest")
    backward = convolve1d(forward[:, ::-1], filt, axis=-1, mode="nearest")
    return backward[:, ::-1]

def upsampling(waveforms, upsampling = 4):
    """
    Returns upsampled waveforms
    :param waveforms:
    :param upsampling:
    :return: waveform after upsampling
    """
    filt = np.ones(upsampling)
    filt_weighted = filt / upsampling
    signal = np.repeat(waveforms, upsampling, axis=-1)

    return __filtfilt_fast(signal, filt_weighted)

def remove_baseline(waveforms, baselines):
    waveforms = np.atleast_2d(waveforms) - np.atleast_2d(baselines).T
    return waveforms

def pole_zero(waveforms):
    def time_profile_pdf_gen(std_dev: float):
        if std_dev == 0:
            return None
        return scipy.stats.norm(0.0, std_dev).pdf

    deconvolution_pars = deconvolution_parameters(ctapipe_subarray().tel[10].camera, 4, 7,
            3, True, 0.05, time_profile_pdf_gen(2),
        )

    pole_zeros, gains, shifts, pz2ds = deconvolution_pars
    pz, gain, shift, pz2d = pole_zeros[0], gains[0], shifts[0], pz2ds[0]

    deconvolved_waveforms = deconvolve(waveforms, 0.0, 4, pz)

    return deconvolved_waveforms

def diff(waveforms):
    diff_wvs = np.diff(waveforms)
    return diff_wvs

def time_pars(waveforms, thr):
    n_wv = len(waveforms)

    time_over_thr = []
    for i in range(0, n_wv):
        waveform = waveforms[i]
        between_ind = np.where(waveform > thr[i])[0]  #count number of samples with amplitude > thr
        time_over_thr.append(max(between_ind, default=0) - min(between_ind, default=0))

    return time_over_thr


def integral_pars(waveforms, thr):
    n_wv = len(waveforms)

    integral_over_thr = []
    for i in range(0, n_wv):
        waveform = waveforms[i]
        integral_over_thr.append(np.sum(waveform[waveform > thr[i]]))

    return integral_over_thr

def integral(waveforms, low, high):
    summ = np.sum(waveforms[..., low:high], axis=-1)
    sum_std = np.std(waveforms[..., low:high], axis=-1)

    return summ, sum_std

def maximum(waveforms):
    maxs = np.max(waveforms, axis=-1)
    return maxs

def read_dynodes():
    filename = 'dynodes.txt'
    df = pd.read_csv(filename, delimiter=' ')
    dynodes = np.repeat(np.array(df["pmt_dynodes"]), 12)
    return dynodes

def broken_pixels():
    broken_pix = []
    pixels = np.arange(0, 1768)
    broken_pixels = np.array([219, 576, 578, 807, 1395, 1684, 1764, 1765, 1766, 1767, 577, 1685, 1686])

    for i in range(0, len(pixels)):
        if i in broken_pixels:
            broken_pix.append(True)
        else:
            broken_pix.append(False)

    return broken_pix

def masked_pixels():
    pixels = np.arange(0, 1768)
    masked_pixels = np.array([576, 578, 1684])

    masked_pix = []
    for i in range(0, len(pixels)):
        if i in masked_pixels:
            masked_pix.append(True)
        else:
            masked_pix.append(False)

    return masked_pix
