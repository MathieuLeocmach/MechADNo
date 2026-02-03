import numpy as np
from scipy import constants
from matplotlib import pyplot as plt

def contour2length(L, persistence=50):
    """Conversion from contour length to actual size of the random walk"""
    return L*(1+L/persistence)**(-2/5)


def plot_phase_diagram(ax, Y=16, SE=6, Cdense=894):
    """plot the phase diagram (T, phi) on the matplotlib axis ax.
    Y is the arm length in base pairs
    SE is the sticky end length in base pairs
    Cdense is the concentration of the dense phase at 20°C in µM
    """
    
    #overlap concentration
    R = 1 + 0.332 * (Y+SE/2) #nm
    R = 0.764 + 0.332 * Y#nm
    Cov = 1/(4/3*np.pi*(contour2length(R)*1e-9)**3*1e3)/constants.Avogadro*1e6 #µM
    print(f'Cov = {Cov:.0f} µM')
    R = 0.764 + 0.332 * (Y+SE/2)#nm
    CovSE = 1/(4/3*np.pi*(contour2length(R)*1e-9)**3*1e3)/constants.Avogadro*1e6 #µM
    print(f'CovSE = {CovSE:.0f} µM')

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
    
    #overlap of SE
    ax.axvline(CovSE/Cov, ls=':', color='k')
    
    #arrow at 1000µM
    ax.annotate(
        '', xy=(1000/Cov, 10), xytext=(1000/Cov, 95), 
        arrowprops=dict(facecolor='black', shrink=0.05),
    )

    ax.set_xlabel(r'$\phi_\mathrm{NS}=C_\mathrm{NS}/C_\mathrm{ov}$')
    ax.set_ylabel(r'$T$ (°C)')
    ax.set_xlim(0,0.74)
    ax.set_ylim(0, 100)

if __name__ == "__main__":
    fig, ax = plt.subplots(1,1, figsize=(3.375,3),layout='constrained')
    plot_phase_diagram(ax)
    for ext in ['png', 'pdf']:
        plt.savefig(f'phase_diagram_Y16SE6.{ext}')

