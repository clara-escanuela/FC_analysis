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
integrals = []
tot = []
sigma_integral = []

amplitudes = []

for j, filename in enumerate(files):
    int_str = re.findall(r'\d+', filename)
    nsamples = int(int_str[16])
    amplitude = int(int_str[13])

    with FCIO(filename) as io:
        for i, e in enumerate(io.events):  # loop over events
            if i > 0:
                continue

            traces = np.array(e.traces)
            baselines = np.array(e.baseline)
            waveforms = np.array(remove_baseline(traces, baselines))

            up_wvs = np.array(upsampling(waveforms))
            pz_wvs = np.array(pole_zero(waveforms))
            up_diff_wvs = np.array(diff(up_wvs))

            tot.append(time_pars(up_wvs, baselines + 3000))
            peaks.append(maximum(up_wvs))
            wv_int, wv_std = integral(up_wvs, 0, nsamples)
            integrals.append(wv_int)
            sigma_integral.append(wv_std)
            amplitudes.append([amplitude]*len(maximum(up_wvs)))

amplitude = np.array(amplitudes)[..., 0:1764][..., (~np.array(broken_pix))][0]
peak = np.array(peaks)[..., 0:1764][..., (~np.array(broken_pix))][0]

amplitude = np.sort(amplitude)
peak = peak[np.argsort(amplitude)]

plt.figure(figsize=(10, 8))

plt.plot(amplitude, peak)

plt.yscale("log")
plt.xlabel("Filter wheel amplitudes")
plt.ylabel("umax")
plt.savefig('/lfs/l1/cta/nieves/FC_analysis/figures/umax_unmasked.pdf', bbox_inches='tight')

#Fit unmasked pixels only in linear region
from scipy.optimize import curve_fit

def f(x, A, B):
    return A*x + B

amplitude = np.array(amplitudes)[..., 0:1764][..., (~np.array(broken_pix)) & (np.array(dynodes) == 7)].flatten()
peak = np.log10(np.array(peaks)[..., 0:1764][..., (~np.array(broken_pix)) & (np.array(dynodes) == 7)]).flatten()

linear_region = (peak>2.7) & (peak<3.4)
amplitude = amplitude[linear_region]
peak = peak[linear_region]

popt, pcov = curve_fit(f, amplitude, peak)

plt.figure(figsize=(10, 8))

all_ampls = np.arange(np.min(amplitude), np.max(amplitude), 1)

plt.plot(amplitude[np.argsort(amplitude)], peak[np.argsort(amplitude)])
plt.plot(all_ampls, popt[0]*np.array(all_ampls)+popt[1], color='black')

peak = np.log10(np.array(peaks)[..., 0:1764][..., 576])
linear_region = (peak>2.7) & (peak<3.4)
amplitude = np.array(amplitudes)[..., 0:1764][..., 576][linear_region]
peak = peak[linear_region]

popt1, pcov1 = curve_fit(f, amplitude, peak)

plt.plot(amplitude[np.argsort(amplitude)], peak[np.argsort(amplitude)], color='orange')
plt.plot(all_ampls, popt1[0]*np.array(all_ampls)+popt1[1], color='orange')

peak = np.log10(np.array(peaks)[..., 0:1764][..., 578])
linear_region = (peak>2.7) & (peak<3.4)
amplitude = np.array(amplitudes)[..., 0:1764][..., 578][linear_region]
peak = peak[linear_region]

popt2, pcov2 = curve_fit(f, amplitude, peak)

plt.plot(amplitude[np.argsort(amplitude)], peak[np.argsort(amplitude)], color='green')
plt.plot(all_ampls, popt2[0]*np.array(all_ampls)+popt2[1], color='green')

peak = np.log10(np.array(peaks)[..., 0:1764][..., 1684])
linear_region = (peak>2.7) & (peak<3.4)
amplitude = np.array(amplitudes)[..., 0:1764][..., 1684][linear_region]
peak = peak[linear_region]

popt3, pcov3 = curve_fit(f, amplitude, peak)

plt.plot(amplitude[np.argsort(amplitude)], peak[np.argsort(amplitude)], color='blue')
plt.plot(all_ampls, popt3[0]*np.array(all_ampls)+popt3[1], color='blue')

plt.xlabel("Filter wheel amplitude", fontsize=20)
plt.ylabel("log10(umax)", fontsize=20)
plt.savefig('/lfs/l1/cta/nieves/FC_analysis/figures/umax_fit.pdf', bbox_inches='tight')

model_ampls = np.arange(0, 3000, 10)

plt.figure(figsize=(10, 8))

plt.plot(model_ampls, (popt[0]*model_ampls + popt[1]) - (popt1[0]*model_ampls + popt1[1]))
plt.plot(model_ampls, (popt[0]*model_ampls + popt[1]) - (popt2[0]*model_ampls + popt2[1]))
plt.plot(model_ampls, (popt[0]*model_ampls + popt[1]) - (popt3[0]*model_ampls + popt3[1]))

plt.xlabel("Amplitudes", fontsize=16)
plt.ylabel("Diff", fontsize=16)
plt.savefig('/lfs/l1/cta/nieves/FC_analysis/figures/umax_diff.pdf', bbox_inches='tight')

from astropy.table import QTable

t = QTable([np.array([popt[0]]), np.array([popt[1]]), np.array([popt1[0]]), np.array([popt1[1]]), np.array([popt2[0]]), np.array([popt2[1]]), np.array([popt3[0]]), np.array([popt3[1]])],
     names=('slope', 'intercept', 'slope1', 'intercept1', 'slope2', 'intercept2', 'slope3', 'intercept3'),
     meta={'name': 'linear'})

t.write('linear.ecsv', overwrite=True)




