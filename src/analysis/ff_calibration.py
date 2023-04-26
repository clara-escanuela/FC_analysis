"""
Calibration of flat-fielding data using the thre masked pixels
"""
from fcio import FCIO
import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy.optimize
from parameters import *
from files_0NSB import files
import re

files = files(0)

#int_str = re.findall(r'\d+', file)
#npixels = int_str[0]
#pdptemp = int_str[2]
#nsb = int_str[5]
#amplitude = int_str[7]
#nsamples = int_str[10]
#nevents = int_str[11]

# Pixels
masked_pixels = masked_pixels()
broken_pix = broken_pixels()
dynodes = read_dynodes()

peaks = []
integral = []
tot = []
sigma_integral = []

amplitudes = []

for j, filename in enumerate(files):
    int_str = re.findall(r'\d+', filename)
    nsamples = int_str[14]
    amplitude = int_str[11]

    with FCIO(filename) as io:
        for i, e in enumerate(io.events):  # loop over events
            if i > 100:
                continue

            traces = np.array(e.traces)
            baselines = np.array(e.baseline)
            waveforms = remove_baseline(traces, baselines)

            up_wvs = upsampling(waveforms)
            pz_wvs = pole_zero(waveforms)
            up_diff_wvs = diff(up_wvs)

            tot.append(time_pars(up_wvs, baselines + 3000))
            peaks.append(maximum(up_wvs))
            wv_int, wv_std = integral(waveforms, 0, nsamples)
            integral.append(wv_int)
            sigma_integral.append(wv_std)
            amplitudes.append(amplitude)

plt.figure(figsize=(10, 8))

plt.plot(amplitudes[..., (~broken_pixels)], peaks[..., (~broken_pixels)])

plt.xlabel("Filter wheel amplitudes")
plt.ylabel("umax")
plt.show()

