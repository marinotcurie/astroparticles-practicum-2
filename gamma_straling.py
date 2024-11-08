import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

pulseheight_sodium = np.array([80, 188, 112.5])  # in mV
energy = np.array([0.511, 1.274, 0.662])  # in eV

def linear_function(P, m, b):
    return m * P + b


params, covariance = curve_fit(linear_function, pulseheight_sodium, energy)

m_fit, b_fit = params
print(f"slope: {m_fit:.5f} eV/mV")
print(f"(b): {b_fit:.5f} eV")

def fit_function(P):
    return m_fit * P + b_fit


plt.scatter(pulseheight_sodium, energy, color='red', label='Data')
plt.plot(pulseheight_sodium, fit_function(pulseheight_sodium), color='blue')
plt.xlabel('pulseheight (mV)')
plt.ylabel('energy (eV)')
plt.legend()
plt.title('Lineair fit between pulseheight and energy')
plt.show()




