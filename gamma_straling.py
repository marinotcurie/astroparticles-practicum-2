import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import pandas as pd

pulseheight = np.array([1691.02943, 4049.5588235])  # in mV
energy = np.array([511, 1274])  # in eV


def linear_function(P, m, b): #function for calibration
    return m * P + b

def gauss_function(x, amp, cen, width):
    return amp * np.exp(-(x - cen))**2 / (2 * width**2)

def fwhm(x, y):
    p0 = [max(y), x[np.argmax(y)], 1.0]
    coeff, _ = curve_fit(gauss_function, x, y, p0=p0)
    return 2 * np.sqrt(2 * np.log(2)) * coeff[2]

params, covariance = curve_fit(linear_function, pulseheight, energy)

df = pd.read_csv('spectrum_cesium137_1.csv', sep=';')

pulseheight_cesium_list = []
counts_cesium = []

for col in df:
    pulseheight_cesium_list.append(col[0])
    counts_cesium.append(col[1])

print(pulseheight_cesium_list, counts_cesium)


# fwhm_value = fwhm(pulseheight_cesium, counts_cesium)


m_fit, b_fit = params
print(f"slope: {m_fit:.5f} eV/mV")
print(f"(b): {b_fit:.5f} eV")

def fit_function(P):
    return m_fit * P + b_fit


plt.scatter(pulseheight, energy, color='red', label='Data')
plt.plot(pulseheight, fit_function(pulseheight), color='blue')
plt.xlabel('pulseheight (mV)')
plt.ylabel('energy (eV)')
plt.legend()
plt.title('Lineair fit between pulseheight and energy')
plt.show()




