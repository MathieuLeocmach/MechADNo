import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os, re
from scipy import optimize, constants as const
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from uncertainties import ufloat, unumpy
import uncertainties
from mechanical_model.linear_mech import Newtonian, JohnsonSegalman

q = 23e6 #m-1
a = 0.255e-6 #m
π = np.pi

#%%

def f2J(f, T=20, q=23e6, a=0.255e-6, d=3):
    """Converts g1(t|q) to a complience J(t) using generalized Stokes-Einstein.
    T in Celsius, wavenumber q in 1/m, radius a in m."""
    return -2*d*np.pi*a/(q**2*const.Boltzmann * const.convert_temperature(T, 'C', 'K')) * np.log(f)

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
    NOSEpattern = os.path.join(dirname, '../DLS/Y16SE0/autocorr-gel6_{:02d}.csv')
    measSE0 = np.zeros(70, dtype=[
        ('file_ID', np.int64),
        ('meas_ID', np.int64),
        ('T', np.float64),
        ('count', np.float64),
        ('g1', np.float64, (36,)),
    ])
    for i in range(len(measSE0)):
        with open(NOSEpattern.format(i+1), encoding='shift_jis') as f:
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
    fig, axs = plt.subplots(2,1, sharex=True, sharey=True, layout='constrained', figsize=(3.375,6))
    for T,m in zip([75,70,65,60,55], '^osv.'):
        #Y16SE0
        i0 = np.argmin(np.abs(measSE0['T'] - T))
        g1 = measSE0['g1'][i0]
        err_f = 1-g1[0]
        J = f2J(g1, T=T, q=q, a=a) #Pa-1
        err_minus_J = np.maximum(0, J - f2J(g1 + err_f, T, q, a))
        err_plus_J = f2J(np.maximum(0, g1 - err_f), T, q, a) - J
        goodt = g1>0.1
        line = axs[0].errorbar(
            Dts[goodt], J[goodt],
            yerr=(err_minus_J[goodt], err_plus_J[goodt]),
            ls='none', marker=m, mfc='none',
            label=f'{T:d}°C'
        )[0]
        #Johnson-Segalman fit of Y16SE0
        G, eta, eta_s =  curve_fit(
            lambda t, G, eta, eta_s: np.log(JohnsonSegalman(G=G, eta=eta, eta_s=eta_s).J(t)),
            Dts[goodt],
            np.log(J[goodt]),
            [0.1, 0.1, 5e-4],
            sigma = (err_plus_J + err_minus_J)[goodt]/J[goodt],
            bounds=(0, np.inf),
        )[0]
        goodt = Dts < eta_s/G
        #plot as if Newtonian with viscosity eta_s
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
        axs[1].errorbar(
            Dts[goodt], J[goodt],
            yerr=(err_minus_J[goodt], err_plus_J[goodt]),
            ls='none', marker=m, color=line.get_color(), mfc='none',
            label=f'{T:d}°C'
        )[0]
        #Johnson-Segalman fit of Y16SE6
        Gi, eta, eta_s =  curve_fit(
            lambda t, Gi, eta, eta_s: np.log(JohnsonSegalman(Gi, eta, eta_s).J(t)),
            Dts[goodt],
            np.log(J[goodt]),
            [0.1, eta_s*10, eta_s/10],
            #sigma = (err_plus_J + err_minus_J)[goodt]/J[goodt],
            bounds=(0, np.inf),
        )[0]
        js = JohnsonSegalman(Gi, eta, eta_s)
        #print(np.ptp((err_plus_J + err_minus_J)[goodt]/J[goodt]))
        axs[1].plot(
            Dts[goodt],
            js.J(Dts[goodt]),
            color=line.get_color(),
            #label='JS fit'
        )
        print(f'T={T}°C\tGi={Gi:.2f} Pa\ttau={1e3*eta/Gi:.2f}ms\teta_s={1e3*eta_s:.3f} mPa.s\txi={(const.Boltzmann*const.convert_temperature(T, 'C', 'K')/Gi/2)**(1/3)*1e9:.1f} nm')

    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    #axs[0].set_ylim(2e-5, 2e-2)
    axs[-1].set_xlabel(r'$\Delta t$ (s)')
    for ax, SE in zip(axs, [0,6]):
        ax.set_ylabel(r'$J(t)$ Pa$^{-1}$')
        ax.set_ylim(3e-3,5)
        ax.legend(title=f'Y16SE{SE}')

    for ext in ['png', 'pdf']:
        plt.savefig(f'J_Y16SE0_Y16SE6.{ext}')
