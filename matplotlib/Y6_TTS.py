# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:08:19 2025

@author: ajiye
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar, curve_fit, least_squares, leastsq
from scipy.special import gamma
from scipy import constants as const
import uncertainties
import matplotlib as mpl
from matplotlib.ticker import LogLocator
import matplotlib.gridspec as gridspec
from matplotlib import color_sequences

mpl.rcdefaults()
plt.style.use(['plt-style-2.mplstyle', 'tableau-colorblind10'])

π = np.pi
def fmmGp(ω, V, G, α, β):
    '''Elastic modulus of the Fractional Maxwell Model'''
    Go = G*ω**β
    Vo = V*ω**α
    return (
        Go**2 * Vo * np.cos(π*α/2) + Vo**2 * Go * np.cos(π*β/2)
    )/(
        Vo**2 + Go**2 + 2*Vo*Go*np.cos(π*(α-β)/2)
    )

def fmmGpp(ω, V, G, α, β):
    '''Viscous modulus of the Fractional Maxwell Model'''
    Go = G*ω**β
    Vo = V*ω**α
    return (
        Go**2 * Vo * np.sin(π*α/2) + Vo**2 * Go * np.sin(π*β/2)
    )/(
        Vo**2 + Go**2 + 2*Vo*Go*np.cos(π*(α-β)/2)
    )
def fmmtandelta(ω, V, G, α, β):
    '''Loss tangent of the Fractional Maxwell Model'''
    Go = G*ω**β
    Vo = V*ω**α
    return (
        Go * np.sin(π*α/2) + Vo * np.sin(π*β/2)
    )/(
        Go * np.cos(π*α/2) + Vo * np.cos(π*β/2)
    )
def fmmVGratio(τ, α, β):
    '''Ratio between quasi-properties V / G, given characteristic time tau and exponents alpha and beta'''
    return (np.sin(π*α/2) - np.cos(π*α/2)) / (np.cos(π*β/2) - np.sin(π*β/2)) * τ**(α-β)

def mmtandelta(ω, k, τ):
    '''Loss tangent of the Maxwell Model'''
    return 1/(τ * ω)
def mmGp(ω, k, η):
    '''Elastic modulus of the Maxwell Model'''
    τ = η/k
    return k * (
        τ**2 * ω**2
    )/(
        1 + τ**2 * ω**2
    )
def mmGpp(ω, k, η):
    '''Viscous modulus of the Maxwell Model'''
    τ = η/k
    return k * (
        τ * ω
    )/(
        1 + τ**2 * ω**2
        )
def get_moduli(basedir, fid):
    filename = os.path.join(basedir, f'Y6-FS-{fid}.tsv')
    data = np.loadtxt(filename, skiprows=2, delimiter='\t')
    Temp = data[0,6]
    freq, Gp, Gpp, tandelta = np.transpose(data[:,2:6])
    torque = np.transpose(data[:,7])
    mask = (Gp != 0) & (Gpp != 0)
    freq, Gp, Gpp, tandelta, torque = freq[mask], Gp[mask], Gpp[mask], tandelta[mask], torque[mask]
    return Temp, freq, Gp, Gpp, tandelta, torque    


basedir = '../rheometer/20250122'

# figures setup
fig, axs = plt.subplots(3,2, figsize=(3.5,3.5), sharex='col', layout='constrained')



ax2 = axs[0,1].inset_axes([0.6, 0.55, 0.35, 0.4])
ax1 = axs[1,1].inset_axes([0.6, 0.1, 0.35, 0.4])


    
axs[2,0].set_xlabel('ω (rad/s)')
axs[2,1].set_xlabel('ωτ')
axs[0,0].set_ylabel('tan δ')
axs[1,0].set_ylabel("G' (Pa)")
axs[2,0].set_ylabel("G'' (Pa)")
axs[0,1].set_ylabel(r'$\tan\delta$')
axs[1,1].set_ylabel(r'$G^\prime\tau^{\beta}/\mathbb{G}$')
axs[2,1].set_ylabel(r'$G^{\prime\prime}\tau^{\beta}/\mathbb{G}$')

ax1.set_xlabel(r'T ($^\circ$C)')
ax1.set_ylabel(r'$\mathbb{G}/\tau^\beta$ (Pa)')
ax2.set_xlabel(r'T ($^\circ$C)')
ax2.set_ylabel(r'$\tau$ (s)')

# n = [3,4,5,6,7,8,9,10] #[3,5,8,6,7] 
# n = [3,4,5,6,7]

# fit to find beta from curve T=10C
T, freq, Gp, Gpp, tandelta, torque = get_moduli(basedir, 7)
omega = 2*π*freq
tau, alpha, beta0 = curve_fit(
    lambda ω, τ, α, β: np.log(fmmtandelta(ω, fmmVGratio(τ, α, β), 1, α, β)),
    omega,
    np.log(tandelta),
    p0=[1, 1, 0],
    bounds=([0,0,0], [np.inf,1,1])
)[0]
print(rf"fit for $\beta$ on curve at T={T}$^\circ$C : $\beta$ = {beta0:.2f}")

# fit to find alpha from curve T=25C
T, freq, Gp, Gpp, tandelta, torque = get_moduli(basedir, 9)
omega = 2*π*freq
tau, alpha0 = curve_fit(
    lambda ω, τ, α: np.log(fmmtandelta(ω, fmmVGratio(τ, α, beta0), 1, α, beta0)),
    omega,
    np.log(tandelta),
    p0=[1, 1],
    bounds=([0,0], [np.inf,1])
)[0]
print(rf"fit for $\alpha$ on curve at T={T}$^\circ$C : $\alpha$ = {alpha0:.2f}")

#redo of fit of beta
T, freq, Gp, Gpp, tandelta, torque = get_moduli(basedir, 7)
omega = 2*π*freq
tau, beta0 = curve_fit(
    lambda ω, τ, β: np.log(fmmtandelta(ω, fmmVGratio(τ, alpha0, β), 1, alpha0, β)),
    omega,
    np.log(tandelta),
    p0=[1, 0],
    bounds=([0,0], [np.inf,1])
)[0]
print(rf"fit for $\beta$ on curve at T={T}$^\circ$C : $\beta$ = {beta0:.2f}")

# time-temperature superposition

n = [3, 9, 5, 6, 7]
Gs, taus, Ts = [], [], []
for i, (x,y), rot in zip(n, [(1,10), (2e-2,0.5), (5e-3,2), (1.5e-3,1.6), (2e-3,1.3e3)], [60]*4+[0]):
    T, freq, Gp, Gpp, tandelta, torque = get_moduli(basedir, i) 
    omega = 2*π*freq
    line = axs[0,0].errorbar(omega, tandelta, 0.2/torque*tandelta, marker='d', linestyle='None')[0]
    axs[1,0].errorbar(omega, Gp, 0.1/torque*Gp, marker='s', linestyle='None', color=line.get_color(), label=rf'T={T}$^\circ$C')
    axs[2,0].errorbar(omega, Gpp, 0.1/torque*Gpp, marker='v', linestyle='None', color=line.get_color())
    #fit by Maxwell
    tauM = curve_fit(
        lambda ω, τ: np.log(mmtandelta(ω, 1, τ)),
        omega,
        np.log(tandelta),
        p0=1,
        bounds=(0, np.inf)
    )[0]
    #fit by fractional Maxwell, using alpha beta values from specific curves
    tau, = curve_fit(
        lambda ω, τ: np.log(fmmtandelta(ω, fmmVGratio(τ, alpha0, beta0), 1, alpha0, beta0)),
        omega,
        np.log(tandelta),
        p0=[1],
        bounds=(0, np.inf)
    )[0]
    G, = curve_fit(
            lambda ω, G: np.log(fmmGp(ω, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0)),
            omega,
            np.log(Gp),
            p0=[1e3],
            bounds=(0,np.inf)
        )[0]
    print(f'T={T}C')
    print(f'tau={tau:.2e}, G={G:.2f}')
    
    # tau, alpha0, beta0 = curve_fit(
    #     lambda ω, τ, α, β: np.log(fmmtandelta(ω, fmmVGratio(τ, α, β), 1, α, β)),
    #     omega,
    #     np.log(tandelta),
    #     p0=[1, 1, 0],
    #     bounds=([0,0,0], [np.inf,1,1])
    # )[0]
    # G, = curve_fit(
    #         lambda ω, G: np.log(fmmGp(ω, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0)),
    #         omega,
    #         np.log(Gp),
    #         [1e3],
    #         bounds=(0,np.inf)
    #     )[0]
    # print(f'T={T}C')
    # print(f'tau={tau:.2e}, alpha={alpha0:.2f}, beta={beta0:.2f}, G={G:.2f}')
    
    Ts.append(T)
    Gs.append(G)
    taus.append(tau)
    
    #fit of the unscaled data
    axs[0,0].plot(omega, fmmtandelta(omega, fmmVGratio(tau, alpha0, beta0), 1, alpha0, beta0), color=line.get_color())
    axs[1,0].plot(omega, fmmGp(omega, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0), color=line.get_color())
    axs[2,0].plot(omega, fmmGpp(omega, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0), color=line.get_color())
    #display temperatures on panel (b)
    axs[1,0].text(x, y, f'{int(T):d}°C', color=line.get_color(), size='x-small', rotation=rot)

    # TTS
    axs[0,1].plot(omega*tau, tandelta, marker='d', linestyle='None', color=line.get_color())
    axs[1,1].plot(omega*tau, Gp/G*tau**beta0, marker='s', linestyle='None', color=line.get_color())
    axs[2,1].plot(omega*tau, Gpp/G*tau**beta0, marker='v', linestyle='None', color=line.get_color())
    
#save fit parameters
np.savetxt(
    'Y16SE6_fMM_bulkrheo_low_temp.tsv',
    np.column_stack((Ts, Gs, taus, np.full(len(Ts), alpha0), np.full(len(Ts), beta0))),
    header='T\tG\ttau\talpha\tbeta',
    delimiter='\t',
    fmt=['%.0f', '%.01f', '%.03f', '%.03f', '%.03f']
)

axs[0,1].plot(omega*tauM, mmtandelta(omega, 1, tauM), linestyle=':', color='black')
axs[1,1].plot(np.logspace(-2,4), mmGp(np.logspace(-2,4), 1, 1), linestyle=':', color='black')
axs[2,1].plot(omega*tauM, mmGpp(omega, 1, tauM), linestyle=':', color='black')
axs[0,1].plot(np.logspace(-2,4), fmmtandelta(np.logspace(-2,4), 1, 1, alpha0, beta0), linestyle='--', color='black')
axs[1,1].plot(np.logspace(-2,4), fmmGp(np.logspace(-2,4), 1, 1, alpha0, beta0), linestyle='--', color='black')
axs[2,1].plot(np.logspace(-2,4), fmmGpp(np.logspace(-2,4), 1, 1, alpha0, beta0), linestyle='--', color='black')

color = color_sequences['tab20c'][4]
ax1.plot(Ts, Gs/taus**beta0, 'o', color=color, label=r'$\mathbb{G}/\tau^\beta$')
ax2.plot(Ts, taus, 'o', color=color, label=r'$\tau$')

#Arrhenius fit
EA, A = curve_fit(
    lambda T, EA, A: A + EA/(const.R*T),
    const.convert_temperature(Ts, 'C', 'K'),
    np.log(taus),
    [2*91.5e3, 0]
)[0]
print(f'E_a = {EA/1e3} kJ/mol')
ax2.plot(Ts, np.exp(A + EA/const.convert_temperature(Ts, 'C', 'K')/const.R), '--k')

#pSE=1 prediction of G
C_NS = 1000 #µM to be converted to mol/m3
ax1.plot(Ts, 0.5*C_NS * 1e-6 *1e3* const.convert_temperature(Ts, 'C', 'K')*const.R, '--k')


axs[0,0].set_ylim(3e-2, 3e1)
axs[1,0].set_ylim(2e-1, 3e3)
axs[2,0].set_ylim(1e1, 7e2)
axs[0,1].set_ylim(3e-2, 3e1)
axs[1,1].set_ylim(2e-4, 3e0)
axs[2,1].set_ylim(1e-2, 9e-1)
axs[0,0].text(0.95, 0.95, '(a)', ha='right', va='top', transform=axs[0,0].transAxes)
for ax, label in zip(axs[1:,0], 'bc'):
    ax.text(0.95, 0.05, f'({label})', ha='right', va='bottom', transform=ax.transAxes)
axs[0,1].text(0.05, 0.05, '(d)', ha='left', va='bottom', transform=axs[0,1].transAxes)
for ax, label in zip(axs[1:,1], 'ef'):
    ax.text(0.05, 0.95, f'({label})', ha='left', va='top', transform=ax.transAxes)
# axs[1,0].legend()

# ax1.legend(fontsize=8, loc='center right')
ax2.set_yscale('log')
ax1.set_ylim(0, 1500)
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top') 
# ax2.xaxis.set_label_position('top')
for ax in [ax1, ax2]:
    ax.tick_params(axis='both', which='major', labelsize=6)   
    ax.tick_params(axis='both', which='minor', labelsize=5)   
    ax.xaxis.label.set_size(7)                             
    ax.yaxis.label.set_size(7) 

for ax in axs.ravel():
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
for label in axs[-1,1].xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)


#handles, labels = axs[1,0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='outside upper center', ncol=3, borderaxespad=0.01)
fig.get_layout_engine().set(wspace=0, w_pad=0, hspace=0, h_pad=0.01)
fig.align_ylabels()

for ext in ["png", "pdf"]:
    fig.savefig(f'Y6-TTS-22012025-colourblind.{ext}', bbox_inches='tight')
