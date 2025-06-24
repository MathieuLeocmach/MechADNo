import numpy as np
from scipy import constants
from matplotlib import pyplot as plt



def plot_phase_diagram(ax, Y=16, SE=6):
    """plot the phase diagram (T, phi) on the matplotlib axis ax.
    Y is the arm length in base pairs
    SE is the sticky end length in base pairs"""
    
    #overlap concentration
    R = 1 + 0.332 * (Y+SE/2) #nm
    Cov = 1/(4/3*np.pi*(R*1e-9)**3*1e3)/constants.Avogadro*1e6 #µM

    #Sticky ends
    C_SE, T = np.loadtxt(f'../simulations/mfold_SE{SE}_concentration.tsv', skiprows=1, usecols=[0,3], unpack=True)
    ax.plot(C_SE / 1.5 / Cov, T, label=r'$T_\mathrm{SE}$')
    
    #Nanostars
    C_NS, T = np.loadtxt(f'../simulations/melting_Y{Y}SE0_concentrations.tsv', skiprows=1, usecols=[0,1], unpack=True)
    ax.plot(C_NS / Cov, T, label=r'$T_\mathrm{NS}$')

    ax.set_xlabel(r'$\phi_\mathrm{NS}=C_\mathrm{NS}/C_\mathrm{ov}$')
    ax.set_ylabel(r'$T$ (°C)')
    ax.set_xlim(0,1.2)
    ax.set_ylim(0, 100)

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1, figsize=(3.375,3),layout='constrained')
    plot_phase_diagram(ax)
    plt.show()

