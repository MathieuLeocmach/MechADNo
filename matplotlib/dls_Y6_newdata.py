# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:56:20 2025

@author: ajiye
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os, re
from scipy import optimize, constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from uncertainties import ufloat, unumpy
import uncertainties
import matplotlib as mpl
from matplotlib import color_sequences
mpl.rcdefaults()
plt.style.use(['plt-style-2.mplstyle', 'tableau-colorblind10'])
# plt.style.use(['tableau-colorblind10'])

q = 23e6 #m-1
a = 0.255e-6 #m
π = np.pi

#%%

def f2msd(f,q=23e6,d=3):
    return -2*d/q**2 * np.log(f)
    
def f2J(f, T=20, q=23e6, a=0.255e-6, d=3):
    """Converts g1(t|q) to a complience J(t) using generalized Stokes-Einstein. 
    T in Celsius, wavenumber q in 1/m, radius a in m."""
    return -2*d*np.pi*a/(q**2*const.Boltzmann * const.convert_temperature(T, 'C', 'K')) * np.log(f)
    
def J2msd(J, T=20, a=0.255e-6, d=3):
    """Converts complience J(t) to msd(t) in m².
    T in Celsius, radius a in m."""
    return d*const.Boltzmann * const.convert_temperature(T, 'C', 'K') /(3*np.pi*a) * J
    
def load_and_sort(pattern):
    """Load all measurements and sort them by measurement order"""
    measurements = np.zeros(180, dtype=[
        ('file_ID', np.int64),
        ('meas_ID', np.int64),
        ('T', np.float64),
        ('count', np.float64),
        ('g1', np.float64, (36,))
        ])
    for i in range(len(measurements)):
        with open(pattern.format(i+1), encoding='shift_jis') as f:
            for line in f:
                if ".nsz" in line:
                    match = re.search(r'_(\d+)\.nsz', line)
                    meas_ID = int(match.group(1))
                if "Temperature of the Holder" in line:
                    T = float(line.split(",")[1])
                if "Count Rate" in line:
                    count = float(line.split(",")[1])
                if "Correlation g1(T)" in line:
                    data = np.loadtxt(f, delimiter=",", skiprows=0)
        measurements[i] = (i+1, meas_ID, T, count, data[:,1])
    return measurements[np.argsort(measurements['meas_ID'])]

def fit_Newtonian(Dt, J, eta_s=1e-3):
    """Fits a Newtonian model to the compliance and returns the viscosity with uncertainty"""
    popt, pcov = curve_fit(
        lambda t, eta: np.log(t / eta),
        Dt,
        np.log(J),
        p0 = eta_s,
        bounds=[[0,], [np.inf,]]
    )
    return uncertainties.correlated_values(popt, pcov)
    
def jsJ(t, Gi, η, ηs):
    '''Creep compliance of the Johnson-Segalman model'''
    return t/(η+ηs) + 1/Gi * (η/(η+ηs))**2 * (1 - np.exp(-Gi*(1/η + 1/ηs)*t))
    
def fit_JS(Dt, J, eta_s, Gi=20, eta=1e-3):
    """Fit to the compliance a Johnson Segalman model with a fixed solvent viscosity.
    Returns the Maxwell modulus at infinite frequency and the Maxwell viscosity, with uncertainties."""
    popt, pcov = curve_fit(
        lambda t, Gi, eta: np.log(jsJ(t, Gi, eta, eta_s)),
        Dt,
        np.log(J),
        p0 = [Gi, eta],
        bounds=[[0,0], [np.inf, np.inf]]
    )
    return uncertainties.correlated_values(popt, pcov)
    

# delay time vector in s
Dts = np.array([
    1.00000e+00, 1.40000e+00, 1.90000e+00, 2.60000e+00, 
    3.60000e+00, 5.00000e+00, 6.90000e+00, 9.50000e+00, 
    1.31000e+01, 1.81000e+01, 2.50000e+01, 3.45000e+01, 
    4.76000e+01, 6.57000e+01, 9.07000e+01, 1.25200e+02, 
    1.72800e+02, 2.38500e+02, 3.29200e+02, 4.54400e+02,
    6.27200e+02, 8.65700e+02, 1.19490e+03, 1.64930e+03, 
    2.27650e+03, 3.14220e+03, 4.33710e+03, 5.98640e+03, 
    8.26290e+03, 1.14051e+04, 1.57422e+04, 2.17286e+04, 
    2.99915e+04, 4.13966e+04, 5.71388e+04, 7.88674e+04,
    ])*1e-6  #s
    
if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    
    #load DLS data from Y16SE0 in buffer
    # NOSEpattern = os.path.join(dirname, '../DLS/Y16SE0/autocorr-gel6_{:02d}.csv')
    NOSEpattern = os.path.join(dirname, '..', 'DLS', 'Y16SE0', 'autocorr-gel6_{:02d}.csv')
    measSE0 = np.zeros(70, dtype=[
        ('file_ID', np.int64),
        ('meas_ID', np.int64),
        ('T', np.float64),
        ('count', np.float64),
        ('g1', np.float64, (36,)),
    ])
    for i in range(len(measSE0)):
        filepath = os.path.normpath(NOSEpattern.format(i+1))
        with open(filepath, encoding='shift_jis') as f:
        # with open(NOSEpattern.format(i+1), encoding='shift_jis') as f:
            for line in f:
                if ".nsz" in line:
                    match = re.search(r'_(\d+)\.nsz', line)
                    meas_ID = int(match.group(1))
                if "Temperature of the Holder" in line:
                    T = float(line.split(",")[1])
                if "Count Rate" in line:
                    count = float(line.split(",")[1])
                if "Correlation" in line:
                    #g2-1
                    data = np.loadtxt(f, delimiter=",", skiprows=0)
        measSE0[i] = (i+1, meas_ID, T, count, np.sqrt(data[:,1]))
    
    #load DLS data from Y16SE6 in buffer
    dirfile = os.path.join(dirname, '../DLS/Y16SE6/')
    Ncooling = 5
    Nrepeat = 5
    Ntemperature = 36
    measurements = []
    
    #load water viscosity
    #water = np.loadtxt(os.path.join(dirfile, '../water.tsv'), skiprows=1)
    #eta_w = CubicSpline(water[:,0], water[:,1]*1e-3) #Pa.s
    
    #load all measurements
    for c in range(Ncooling):
        name = f"cooling_{c+1}/Y16SE6-1mM-NP500nm-0.1pct-properprotocol2_cool{c+1}_{{:03d}}.csv"
        measurements.append(load_and_sort(os.path.join(dirfile, name)))
    measurements = np.reshape(measurements, (Ncooling, Ntemperature, Nrepeat))
    
    #average g2 across coolings and repeats, taking count rates into account, but discarding low intercepts
    good = measurements['g1'][...,0] > 1-4e-3
    meang2s = np.sum((good * measurements['count'])[...,None] * (1 + measurements['g1']**2), axis=(0,2)) / np.maximum(1, (good * measurements['count']).sum((0,2)))[...,None]
    meang1s = np.sqrt(meang2s -1)
    Ts = np.rint(measurements['T'].mean((0,2))).astype(int)
    
    #draw J(t)
    taus, Gis, etas = [], [], []
    fig = plt.figure(figsize=(3.53, 3.53), layout="constrained")
    gs = gridspec.GridSpec(6, 2, figure=fig, width_ratios=[3,2])#, hspace=0.8, wspace=0.5)
    axs = [None, None]
    axs[0] = fig.add_subplot(gs[0:3, 0])
    axs[1] = fig.add_subplot(gs[3:6, 0], sharex=axs[0])
    for T,m in zip(range(75,54,-1), '^osv.'*5): #[75,70,65,60,55]
        #Y16SE0
        i0 = np.argmin(np.abs(measSE0['T'] - T))
        g1 = measSE0['g1'][i0]
        err_f = 1-g1[0]
        J = f2J(g1, T=T, q=q, a=a) #Pa-1
        err_minus_J = np.maximum(0, J - f2J(g1 + err_f, T, q, a))
        err_plus_J = f2J(np.maximum(0, g1 - err_f), T, q, a) - J
        goodt = g1>0.1
        if T%5 == 0:
            print(f"T={T}")
            line = axs[0].errorbar(
                Dts[goodt], J[goodt], 
                yerr=(err_minus_J[goodt], err_plus_J[goodt]),
                ls='none', marker=m, mfc='none',
                label=f'{T:d}°C'
            )[0]
        #Johnson-Segalman fit of Y16SE0
        eta_s, eta, tau =  curve_fit(
            lambda t, eta_s, eta, tau: np.log(jsJ(t, eta/tau, eta, eta_s)),
            Dts[goodt],
            np.log(J[goodt]),
            [5e-4, 0.1, 0.1],
            sigma = (err_plus_J + err_minus_J)[goodt]/J[goodt],
            bounds=(0, np.inf),
        )[0]
        goodt = Dts < eta_s/eta*tau
        #plot as if Newtonian with viscosity eta_s
        if T%5 == 0:
            axs[0].plot(
                Dts[goodt], Dts[goodt]/eta_s,
                color=line.get_color()
                )
        
        #Y16SE6
        iT = np.argmin(np.abs(Ts - T))
        g1 = meang1s[iT]
        err_f = 1-g1[0]
        J = f2J(g1, T=T, q=q, a=a) #Pa-1
        err_minus_J = np.maximum(0, J - f2J(g1 + err_f, T, q, a))
        err_plus_J = f2J(np.maximum(0, g1 - err_f), T, q, a) - J
        goodt = g1>0.2
        if T%5 == 0:
            axs[1].errorbar(
                Dts[goodt], J[goodt], 
                yerr=(err_minus_J[goodt], err_plus_J[goodt]),
                ls='none', marker=m, color=line.get_color(), mfc='none',
                label=f'{T:d}°C'
            )[0]
        #Johnson-Segalman fit of Y16SE6
        Gi, tau, eta_s =  curve_fit(
            lambda t, Gi, tau, eta_s: np.log(jsJ(t, Gi, Gi*tau, eta_s)),
            Dts[goodt],
            np.log(J[goodt]),
            [0.1, 10, eta_s/10],
            #sigma = (err_plus_J + err_minus_J)[goodt]/J[goodt],
            bounds=(0, np.inf),
        )[0]
        #print(np.ptp((err_plus_J + err_minus_J)[goodt]/J[goodt]))
        if T%5 == 0:
            axs[1].plot(
                Dts[goodt], 
                jsJ(Dts[goodt], Gi, Gi*tau, eta_s), 
                color=line.get_color(),
                #label='JS fit'
            )
        print(f'T={T}°C\tGi={Gi:.2f} Pa')
        taus = np.append(taus, tau)
        Gis = np.append(Gis, Gi)
        etas = np.append(etas, eta_s)
        
    
    #axs[0].set_ylim(2e-5, 2e-2)
    axs[-1].set_xlabel(r'$\Delta t$ (s)')
    for ax, SE, label in zip(axs, [0,6], 'ab'):
        ax.set_ylabel(r'$J$ (Pa$^{-1})$')
        ax.set_ylim(3e-3,5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.text(0.05, 0.95, f'({label}) Y16SE{SE}', ha='left', va='top', transform=ax.transAxes)
    axs[0].legend()
    # fig.suptitle('Y16SE6_1mM_newdata')
    #axs[0].text(1e-7, 5e0, 'A', color='black', fontsize=10)
    plt.setp(axs[0].get_xticklabels(), visible=False)
    ax2 = [None, None, None]
    ax2[0] = fig.add_subplot(gs[0:2, 1])   
    ax2[1] = fig.add_subplot(gs[2:4, 1], sharex=ax2[0])   
    ax2[2] = fig.add_subplot(gs[4:6, 1], sharex=ax2[0])
    color = color_sequences['tab20c'][4]
    ax2[0].plot(range(75,54,-1), Gis, 'o', color=color)
    ax2[1].plot(range(75,54,-1), Gis*taus, 'o', color=color, label=r'$\eta + \eta_s$')
    ax2[1].text(56, 3e-3, r'$\eta + \eta_s$', color=color, size='x-small', ha='left')
    ax2[1].plot(range(75,54,-1), etas, marker='.', color='black', mfc='none', label=r'$\eta_s$')
    ax2[1].text(56, 3e-4, r'$\eta_s$', color='black', size='x-small', ha='left')
    
    ax2[2].plot(range(75,54,-1), taus, 'o', color=color)
    # ax2[0].set_yscale('log')
    ax2[1].set_yscale('log')
    ax2[2].set_yscale('log')
    ax2[0].set_xlim(54, 76)
    ax2[0].set_ylim(12, 29)
    ax2[-1].set_xlabel(r'T ($^\circ$C)')
    ax2[0].set_ylabel(r'$G_i$ (Pa)')
    ax2[1].set_ylabel(r'$\eta$ (Pa.s)')
    ax2[2].set_ylabel(r'$\tau$ (s)')
    #ax2[1].legend()
    
    # fig2.suptitle('Y16SE6_1mM_newdata')
    for ax, label in zip(ax2, 'cde'):
        ax.text(0.95, 0.95, f'({label})', ha='right', va='top', transform=ax.transAxes)
        ax.axvspan(65, 80, ls='none', color=[0.8]*3+[1])
    plt.setp(ax2[0].get_xticklabels(), visible=False)
    plt.setp(ax2[1].get_xticklabels(), visible=False)
    # np.savetxt('params_Y16SE6_newdata.tsv', 
    #         np.array([range(75,55,-1),Gis, taus]),
    #         delimiter=',', fmt='%.4e', header='T(C),Gi(Pa),tau(s)')
    
    fig.get_layout_engine().set(wspace=0, w_pad=0, hspace=0, h_pad=0.01)
    fig.align_ylabels()
    for ext in ['png', 'pdf']:
        fig.savefig(f'DLS_Y16SE6_newdata_colourblind.{ext}')
