import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os, re
from scipy import optimize, constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
import uncertainties

q = 23e6 #m-1
a = 0.25e-6 #m
π = np.pi

#%%

def f2msd(f,q=23e6,d=3):
    return -2*d/q**2 * np.log(f)
    
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
    dirfile = os.path.join(dirname, '../DLS/Y16SE6/')
    Ncooling = 5
    Nrepeat = 5
    Ntemperature = 36
    measurements = []
    
    #load all measurements
    for c in range(Ncooling):
        name = f"cooling_{c+1}/Y16SE6-1mM-NP500nm-0.1pct-properprotocol2_cool{c+1}_{{:03d}}.csv"
        measurements.append(load_and_sort(os.path.join(dirfile, name)))
    measurements = np.reshape(measurements, (Ncooling, Ntemperature, Nrepeat))
    
    #average g2 across coolings and repeats, taking count rates into account
    meang2s = np.sum(measurements['count'][...,None] * (1 + measurements['g1']**2), axis=(0,2)) / measurements['count'].sum((0,2))[...,None]
    meang1s = np.sqrt(meang2s -1)
    Ts = np.rint(measurements['T'].mean((0,2)))
    
    fig, axs = plt.subplots(2,1, sharex=True, sharey=True, layout='constrained', figsize=(3.375,6))
    for ax, iT in zip(axs, [10,19]):
        goodt = meang1s[iT]>1e-1
        err_f = 1-meang1s[iT][0]
        msd = f2msd(meang1s[iT], q)*1e12 #µm²
        err_minus_msd = np.maximum(0, msd - f2msd(meang1s[iT] + err_f, q)*1e12)
        err_plus_msd = f2msd(np.maximum(0, meang1s[iT] - err_f), q)*1e12 - msd
        ax.errorbar(
            Dts[goodt], msd[goodt], 
            yerr=(err_minus_msd[goodt], err_plus_msd[goodt]),
            ls='none', marker='o', mfc='none', zorder=1.5,
            label=f'T={Ts[iT]}°C'
        )
        #ax.plot(Dts[goodt], f2msd(meang1s[iT][goodt], q)*1e12, marker='.', label=f'T={Ts[iT]}°C')
        #ax.plot(Dts, meang1s[iT], marker='.', label=f'T={Ts[iT]}°C')
        #ax.plot(Dts, measurements['g1'][0,iT,0], marker='.', label=f'T={Ts[iT]}°C')
        
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[-1].set_xlabel(r'$\Delta t$ (s)')
    for ax in axs:
        ax.set_ylabel(r'$\langle\Delta r^2\rangle$ µm$^2$')
        ax.legend()
    
    for ext in ['png', 'pdf']:
        plt.savefig(f'MSD_Y16SE6.{ext}')
