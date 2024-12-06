import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
import pandas as pd

pulseheight = np.array([1691.02943, 4049.5588235, 2186.02948])  # in mV
energy = np.array([511, 1274, 662])  # in keV
energy_error = np.array([19.2] * len(energy)) #foutmarge van halve bin




def linear_function(P, a, b): #function for calibration
    return a * P + b

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


a_fit, b_fit = params
print(f"slope: {a_fit:.5f} keV/mV")
print(f"(b): {b_fit:.5f} keV")

def fit_function(P):
    return a_fit * P + b_fit

predicted_energy = fit_function(pulseheight)
residuals = energy - predicted_energy
chi_squared = np.sum((residuals / energy_error) ** 2)
print(f"Chi-squared value: {chi_squared:.5f}")



param_errors = np.sqrt(np.diagonal(covariance))
print(f"Error on a: {param_errors[0]:.5f}")
print(f"Error on b: {param_errors[1]:.5f}")
#plot en residu
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)

ax1.errorbar(
    pulseheight, energy, yerr=energy_error, fmt='o', 
    color='red', label='Data (met foutmarge)', capsize=5
)
ax1.plot(
    pulseheight, predicted_energy, 
    color='blue', label='Lineaire fit'
)
ax1.set_ylabel('Energie (keV)')
ax1.set_title('Lineaire fit tussen pulsehoogte en energie')
ax1.legend()

# residu
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)  # Lijn bij residu = 0
ax2.errorbar(
    pulseheight, residuals, yerr=energy_error, fmt='o', 
    color='red', capsize=5
)
ax2.set_xlabel('Pulsehoogte (mV)')
ax2.set_ylabel('Residu (keV)')
ax2.set_title('Residual plot')

plt.tight_layout()

plt.scatter(pulseheight, energy, color='red', label='Data')
plt.plot(pulseheight, fit_function(pulseheight), color='blue')
plt.xlabel('pulseheight (mV)')
plt.ylabel('energy (eV)')
plt.legend()
plt.title('Lineair fit between pulseheight and energy')

plt.show()




