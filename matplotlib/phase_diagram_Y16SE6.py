import numpy as np
from scipy import constants
from matplotlib import pyplot as plt



def plot_phase_diagram(ax, Y=16, SE=6, Cdense=894):
    """plot the phase diagram (T, phi) on the matplotlib axis ax.
    Y is the arm length in base pairs
    SE is the sticky end length in base pairs
    Cdense is the concentration of the dense phase at 20°C in µM
    """
    
    #overlap concentration
    R = 1 + 0.332 * (Y+SE/2) #nm
    Cov = 1/(4/3*np.pi*(R*1e-9)**3*1e3)/constants.Avogadro*1e6 #µM

    #Sticky ends
    C_SE, T = np.loadtxt(f'../simulations/mfold_SE{SE}_concentration.tsv', skiprows=1, usecols=[0,3], unpack=True)
    line, = ax.plot(C_SE / 1.5 / Cov, T, label=r'$T_\mathrm{SE}$')
    imax = np.where(4/3*C_SE > Cdense)[0][0]
    TC = np.interp(Cdense, 4/3*C_SE, T)
    CC = np.interp(TC, T, C_SE/1.5)
    print(TC, CC)
    ax.plot(
        np.concatenate([[Cdense] , (Cdense - (C_SE[:imax] / 1.5)), [CC]]) / Cov, 
        np.concatenate([[0], T[:imax], [TC]]), 
        '--', color=line.get_color()
    )
    
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
    for ext in ['png', 'pdf']:
        plt.savefig(f'phase_diagram_Y16SE6.{ext}')

