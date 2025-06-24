import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots(1,1, figsize=(3.375,3),layout='constrained')

Cov = 1015 #µM

#Sticky ends
C_SE, T = np.loadtxt('../simulations/mfold_SE6_concentration.tsv', skiprows=1, usecols=[0,3], unpack=True)
plt.plot(C_SE / 1.5 / Cov, T, label=r'$T_\mathrm{SE}$')

C_NS, T = np.loadtxt('../simulations/melting_Y16SE0_concentrations.tsv', skiprows=1, usecols=[0,1], unpack=True)
plt.plot(C_NS / Cov, T, label=r'$T_\mathrm{NS}$')

ax.set_xlabel(r'$\phi_\mathrm{NS}=C_\mathrm{NS}/C_\mathrm{ov}$')
ax.set_ylabel(r'$T$ (°C)')
ax.set_xlim(0,1.2)
ax.set_ylim(0, 100)

plt.show()

