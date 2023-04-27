import numpy as np
import matplotlib.pyplot as plt
from fcio import FCIO
from ctapipe.image.extractor import FlashCamExtractor
from parameters import *
from files_0NSB import *
from numpyext import *
import re
from astropy.table import Table, QTable

files = [files(0)[0]]

broken_pix = broken_pixels()
masked_pix = masked_pixels()
recon_charge = []
true_charge = []

t = Table.read("linear.ecsv")
slope = np.array(t["slope1"])
intercept = np.array(t["intercept1"])
dynodes = read_dynodes()
for j, filename in enumerate(files):
    int_str = re.findall(r'\d+', filename)
    nsamples = int(int_str[16])
    amplitude = int(int_str[13])
    ulin = slope[0]*amplitude + intercept[0]
    with FCIO(filename) as io:
        for i, e in enumerate(io.events):  # loop over events
            if i > 5:
                continue

            traces = np.array(e.traces)
            baselines = np.array(e.baseline)
            waveforms = np.array(remove_baseline(traces, baselines))[0:1764]
            waveforms = waveforms/10

            subarray = ctapipe_subarray()
            extractor = FlashCamExtractor(subarray)
            dl1 = extractor(waveforms = waveforms, tel_id=10, selected_gain_channel=0, broken_pixels=np.array(broken_pix), )
            true_charge.append(np.repeat(ulin, len(dl1.image))/10) 
            recon_charge.append(dl1.image)

# Charge bias

p_mean, p_std, p_n, p_x_edges = profile(np.log10(np.array(true_charge).flatten()), (np.array(recon_charge).flatten() - np.array(true_charge).flatten())/np.array(true_charge).flatten(), bins=40, sigma_cut=3)
p_x_edges = 10**p_x_edges
p_x = p_x_edges[:-1] + np.diff(p_x_edges) / 2

plt.figure(figsize=(10, 8))

plt.scatter(np.array(true_charge).flatten(), (np.array(recon_charge).flatten() - np.array(true_charge).flatten())/np.array(true_charge).flatten(), s=6, color='steelblue')
plt.plot(p_x, p_mean, color='black', linestyle='--', linewidth=2)

plt.ylabel(r"Bias, $\frac{q_{recon} - q_{calib}}{q_{calib}}$", fontsize=18)
plt.xlabel(r"Calibrated charge, $q_{calib}$ (p.e.)", fontsize=18)
plt.show()  

# Charge resolution

p_mean, p_std, p_n, p_x_edges = profile(np.array(true_charge).flatten(), (np.array(recon_charge).flatten() - np.array(true_charge).flatten())**2, bins=40, sigma_cut=3)

plt.figure(figsize=(10, 8))

plt.plot(p_x_edges[:-1], np.sqrt(p_mean)/p_x_edges[:-1], color='blue', linewidth=2)
plt.plot(p_x_edges[:-1], 1/np.sqrt(p_x_edges[:-1]), color='black', linestyle='--', linewidth=2, label='Poisson limit')

plt.legend(fontsize=16)
plt.ylabel(r"Charge resolution", fontsize=18)
plt.xlabel(r"Calibrated charge, $q_{calib}$ (p.e.)", fontsize=18)
plt.show()


