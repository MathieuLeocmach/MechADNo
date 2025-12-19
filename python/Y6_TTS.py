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
import uncertainties
import matplotlib as mpl
from matplotlib.ticker import LogLocator
import matplotlib.gridspec as gridspec

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
def get_moduli(file):
    filename = f'Y6-FS-{file}.tsv'
    data = np.loadtxt(filename, skiprows=2, delimiter='\t')
    Temp = data[0,6]
    freq, Gp, Gpp, tandelta = np.transpose(data[:,2:6])
    torque = np.transpose(data[:,7])
    mask = (Gp != 0) & (Gpp != 0)
    freq, Gp, Gpp, tandelta, torque = freq[mask], Gp[mask], Gpp[mask], tandelta[mask], torque[mask]
    return Temp, freq, Gp, Gpp, tandelta, torque    


os.chdir(r"C:\Users\ajiye\Documents\Rheometre\20250122")

# figures setup

# fig, axs = plt.subplots(3, 2, figsize=(89.7/25.4, 89.7/25.4*1.2), sharex='col', layout='constrained')#, gridspec_kw={'bottom': 0.20})
fig = plt.figure(
    figsize=(89.7/25.4, 89.7/25.4 * 1.4),
    layout="constrained"
)

gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.3],wspace=0.001)

axs = np.empty((3,2), dtype=object)
for i in range(3):
    for j in range(2):
        axs[i,j] = fig.add_subplot(gs[i, j])

# fig1, ax1 = plt.subplots(1, 1, figsize=(3,2), layout='constrained')
# fig2, ax2 = plt.subplots(1, 1, figsize=(3,2), layout='constrained')
ax2 = axs[0,1].inset_axes([0.6, 0.5, 0.35, 0.4])
ax1 = axs[1,1].inset_axes([0.6, 0.15, 0.35, 0.4])


    
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
T, freq, Gp, Gpp, tandelta, torque = get_moduli(7)
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
T, freq, Gp, Gpp, tandelta, torque = get_moduli(9)
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
T, freq, Gp, Gpp, tandelta, torque = get_moduli(7)
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
for i in n:
    T, freq, Gp, Gpp, tandelta, torque = get_moduli(i) 
    omega = 2*π*freq
    line = axs[0,0].errorbar(omega, tandelta, 0.2/torque*tandelta, marker='o', linestyle='None')[0]
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
    
    axs[0,0].plot(omega, fmmtandelta(omega, fmmVGratio(tau, alpha0, beta0), 1, alpha0, beta0), color=line.get_color())
    axs[1,0].plot(omega, fmmGp(omega, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0), color=line.get_color())
    axs[2,0].plot(omega, fmmGpp(omega, G*fmmVGratio(tau, alpha0, beta0), G, alpha0, beta0), color=line.get_color())

    
    axs[0,1].plot(omega*tau, tandelta, marker='o', linestyle='None', color=line.get_color())
    # axs[0,1].plot(omega*tau, fmmtandelta(omega, G*fmmVGratio(tau, alpha, beta), G, alpha, beta), color=line.get_color())
    # axs[1,1].plot(omega*tau, Gp, marker='s', linestyle='None', color=line.get_color())
    axs[1,1].plot(omega*tau, Gp/G*tau**beta0, marker='s', linestyle='None', color=line.get_color())
    # axs[1,1].plot(omega*tau, fmmGp(omega, G*fmmVGratio(tau, alpha, beta), G, alpha, beta)/G, color=line.get_color())
    # axs[2,1].plot(omega*tau, fmmGpp(omega, G*fmmVGratio(tau, alpha, beta), G, alpha, beta)/G, color=line.get_color())
    axs[2,1].plot(omega*tau, Gpp/G*tau**beta0, marker='v', linestyle='None', color=line.get_color())

axs[0,1].plot(omega*tauM, mmtandelta(omega, 1, tauM), linestyle='--', color='black')
axs[1,1].plot(np.logspace(-2,4), fmmGp(np.logspace(-2,4), 1, 1, alpha0, beta0), linestyle='--', color='black')
axs[2,1].plot(np.logspace(-2,4), fmmGpp(np.logspace(-2,4), 1, 1, alpha0, beta0), linestyle='--', color='black')

ax1.plot(Ts, Gs/taus**beta0, marker='d', label=r'$\mathbb{G}/\tau^\beta$')
ax2.plot(Ts, taus, marker='d', label=r'$\tau$')


axs[0,0].set_ylim(3e-2, 3e1)
axs[1,0].set_ylim(1e-1, 3e3)
axs[2,0].set_ylim(1e1, 7e2)
axs[0,1].set_ylim(1e-2, 3e1)
axs[1,1].set_ylim(1e-4, 3e0)
axs[2,1].set_ylim(1e-2, 1e0)
axs[0,0].text(8e-5, 3e1, 'A', color='black')
axs[1,0].text(8e-5, 3e3, 'B', color='black')
axs[2,0].text(8e-5, 7e2, 'C', color='black')
axs[0,1].text(2e-5, 3e1, 'D', color='black')
axs[1,1].text(2e-5, 3e0, 'E', color='black')
axs[2,1].text(2e-5, 1e0, 'F', color='black')
# axs[1,0].legend()

# ax1.legend(fontsize=8, loc='center right')
ax2.set_yscale('log')
# ax1.set_ylim(900, 1500)
ax1.xaxis.tick_top()
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
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
    ax.tick_params(axis='x', which='minor', length=2)

# fig.set_constrained_layout(False)
fig.get_layout_engine().set(h_pad=0.05, w_pad=0.001)

legend_ax = fig.add_subplot(gs[3, :])
# legend_ax = fig.add_axes([0, -0.01, 1, 0.15])  
legend_ax.axis('off')
handles, labels = axs[1,0].get_legend_handles_labels()

legend_ax.legend(
    handles, labels,
    loc='center',
    # bbox_to_anchor=(0.5, -0.02),   # adjust spacing below
    ncol=3,              # horizontal legend
    frameon=True,
    fontsize=7,                    # adjust for journal size
)

figname = r"C:\Users\ajiye\Documents\Redaction\paper dna gels\figures\Y6-TTS-22012025-colourblind"
for ex in [".png", ".pdf"]:
    fig.savefig(figname+ex, bbox_inches='tight', pad_inches=0.2)