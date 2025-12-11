import os, re, datetime
import numpy as np
from scipy import optimize, constants as const
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d
from pandas import read_csv
import matplotlib.pyplot as plt
from matplotlib import color_sequences

q = 23e6 #m-1
a = 0.255e-6 #m
π = np.pi

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

def f2J(f, T=20, q=23e6, a=0.255e-6, d=3):
    """Converts g1(t|q) to a complience J(t) using generalized Stokes-Einstein.
    T in Celsius, wavenumber q in 1/m, radius a in m."""
    return -2*d*np.pi*a/(q**2*const.Boltzmann * const.convert_temperature(T, 'C', 'K')) * np.log(f)

def load_measurement(fname):
    with open(fname, encoding='shift_jis') as f:
        for line in f:
            if ".nsz" in line:
                match = re.search(r'_(\d+)\.nsz', line)
                meas_ID = int(match.group(1))
            if "Date" in line:
                timestamp = datetime.datetime.strptime(line.split(",")[1], '%Y年%m月%d日 %H:%M:%S').isoformat()
            if "Temperature of the Holder" in line:
                T = float(line.split(",")[1])
            if "Count Rate" in line:
                count = float(line.split(",")[1])
            if "Correlation g1(T)" in line:
                data = np.loadtxt(f, delimiter=",", skiprows=0)
            if "Correlation" in line and "g1(T)" not in line:
                #g2-1
                data = np.loadtxt(f, delimiter=",", skiprows=0)
                data[:,1] = np.sqrt(data[:,1])
    return meas_ID, timestamp, T, count, data[:,1]

def load_and_sort(pattern, i0=1):
    """Load 180 measurements and sort them by measurement order"""
    measurements = np.zeros(180, dtype=[
        #('file_ID', np.int64),
        ('meas_ID', np.int64),
        ('timestamp', 'datetime64[s]'),
        ('T', np.float64),
        ('count', np.float64),
        ('g1', np.float64, (36,))
        ])
    for i in range(len(measurements)):
        measurements[i] = load_measurement(pattern.format(i+i0))
    return measurements[np.argsort(measurements['timestamp'])]

def load_variable(directory, m):
    """Load all measurements in a directory that match regular expression m and sort them by measurement order"""
    measurements = np.array(
        [
            load_measurement(os.path.join(directory, fname))
            for fname in os.listdir(directory)
            if m.match(fname) is not None
        ],
        dtype=[
            #('file_ID', np.int64),
            ('meas_ID', np.int64),
            ('timestamp', 'datetime64[s]'),
            ('T', np.float64),
            ('count', np.float64),
            ('g1', np.float64, (36,))
        ]
    )
    return measurements[np.argsort(measurements['timestamp'])]

def load_DLS(rootdir, SE=6, Y=16, c=1000):
    """Helper function to load the right DLS data"""
    if c==1000:
        if SE in [4,6]:
            dirfile = os.path.join(rootdir, f'../DLS/Y16SE{SE}/')
            Ncooling = 5
            Nrepeat = 5
            Ntemperature = 36
            middle_name = {
                4:'1mM-NP500nm-0.1pct-cooling',
                6:'1mM-NP500nm-0.1pct-properprotocol2_cool'
            }[SE]
            measurements = []
            #load all measurements
            for c in range(Ncooling):
                name = f"cooling_{c+1}/Y16SE{SE}-{middle_name}{c+1}_{{:03d}}.csv"
                measurements.append(load_and_sort(os.path.join(dirfile, name)))
            return np.reshape(measurements, (Ncooling, Ntemperature, Nrepeat))
        elif SE==8:
            measurements = [
                load_variable(
                    os.path.join(rootdir, '../DLS/Y16SE8/sample_2/cooling_1'),
                    re.compile('Y16SE8-1mM-NP-500nm-01pct-cooling1_([0-9]*).csv')
                )
            ]
            for c in range(2,10):
                measurements.append(load_variable(
                    os.path.join(rootdir, f'../DLS/Y16SE8/sample_2/cooling_{c}'),
                    re.compile(f'Y16SE8-1mM-NP500nm-01pct-cooling{c}_([0-9]*).csv')
                ))
            return np.reshape(measurements, (9, 20, 5))
    elif Y==16:
        measurements = [
                load_variable(
                    os.path.join(rootdir, f'../DLS/Y16SE{SE}_{c}uM/'),
                    re.compile('autocorr-gel.*_([0-9]*).csv')
                )
            ]
        return np.reshape(measurements, (1, len(measurements[0]), 1))
    elif c==500 and SE==6 and Y==32:
        measurements = [
            load_variable(
                os.path.join(rootdir, f'../DLS/Y{Y}SE{SE}_{c}uM/sample_{i+1}'),
                re.compile('autocorr-gel.*_([0-9]*).csv')
            )
            for i in range(2)
        ]
        return np.reshape(measurements, (2, len(measurements[0]), 1))

def phi_rotating_assembled_only_saturated(pSE, pNS=1, SE=6, C_0=1e-3, Y=16, persistence=50):
    """Packing fraction of freely rotating doublets of assembled NS. Valid only at small pSE.
    Lengths are saturated by the persistence length.

    pSE is the probability of association of a SE-SE bond.
    pNS is the probability of association of a nanostar.
    SE is the length of the sticky end, in base pairs.
    C_0 is the nominal molar concentration of nanostars, i.e. the concentration of each strand, in mol/L.
    Y is the length of the arm in base pairs.
    persistence is the persistence length in nm."""
    R = 0.764 + Y * 0.332
    L = 2*R + SE * 0.332
    #Volume occupied by a NS in nm^3
    V_NS = np.pi/3 * min(2*R, persistence)**3
    #Volumne occupied by a freely rotating doublet of NS in nm^3
    V_rotating = np.pi/6* min(2*R+L, persistence)**3
    #At small pSE, doublet is the only relevant cluster,
    # thus the concentration of doublet is half the concentration of assembled nanostar
    # that are assembled
    return (V_rotating/2 * 3 * pSE * (1-pSE)**2 * pNS) *1e-9**3 * C_0*1e3 *const.Avogadro


fig, axs = plt.subplots(1,2, figsize=(7.3,3), sharey=True, layout='constrained')
axs[0].set_ylabel(r'$\tan\delta_\min$')
axs[0].set_xlabel(r'$p_\mathrm{SE}$')
axs[1].set_xlabel(r'$\phi_\mathrm{2,rot}$')
axs[0].set_yscale('log')
#axs[1].set_xscale('log')
for ax in axs:
    ax.axhline(1, ls='--', color='k')

marks = {16:'.', 32:'*'}
#colors = to_rgba_array(plt.rcParams['axes.prop_cycle'].by_key()['color'])
#SE2col = {4:colors[0], 6:colors[1], 8:colors[2]}

for Y, SE, C_NS, icolor in [(16, 4, 600, 2), (16, 4, 800, 1), (16, 4, 1000, 0), (16, 6, 500,6), (16, 6, 1000, 4), (16, 8, 1000,8), (32, 6, 500, 6)]:
    measurements = load_DLS(os.path.dirname('.'), Y=Y, SE=SE, c=C_NS)

    #average g2 across coolings and repeats, taking count rates into account, but discarding low intercepts
    good = measurements['g1'][...,0] > 1-4e-3
    meang2s = np.sum((good * measurements['count'])[...,None] * (1 + measurements['g1']**2), axis=(0,2)) / np.maximum(1, (good * measurements['count']).sum((0,2)))[...,None]
    meang1s = np.sqrt(meang2s -1)
    Ts = np.rint(measurements['T'].mean((0,2))).astype(int)

    #calculate tan delta from the local log-log slope of J(t) and take the minimum for each T
    mintandelta = []
    for T, g1 in zip(Ts, meang1s):
        if np.any(np.isnan(g1)): continue
        J = f2J(g1, T=T, q=q, a=a) #Pa-1
        # log-log slope
        #alpha = np.gradient(np.log(J))/np.gradient(np.log(Dts))
        # Uncomment for smoothing before taking the slope
        w = 1
        alpha = gaussian_filter1d(np.log(J), w, order=1)/gaussian_filter1d(np.log(Dts), w, order=1)
        # Restrict the range of t where to look for minimum
        goodt = (g1>0.2)
        goodt[:5] = False
        if np.any((alpha[5:]<=0) | (alpha[5:]>1)):
            goodt[np.where((alpha[5:]<=0) | (alpha[5:]>1))[0][0]+5:] = False
        M = np.argmax(alpha[goodt])
        m = np.argmin(alpha[goodt][:M])
        #if m<1: continue
        #print(m, M)
        #convert from slope to tan delta (locally a power-law fluid)
        mintandelta.append([T, np.tan(np.pi/2*alpha[goodt][m])])
    mintandelta = np.array(mintandelta)

    #remove very high temperature regime where there is no maximum
    mintandelta = mintandelta[np.argmax(mintandelta[:,1]):]

    #Load simulation results of NS melting
    meltingSE0 = read_csv(f'../simulations/melting_Y{Y}SE0/Y{Y}SE0_{C_NS:.1f}uM_complexes_concentration_melting-1.tsv', sep='\t').rename(columns={'# temperature':'T'})
    pNS = CubicSpline(meltingSE0['T'], meltingSE0['+'.join('ABC'*1)]/(C_NS*1e-6))
    #load simulation results of SE-SE duplex melting
    pSEdata = read_csv(f'../simulations/melting_SE{SE}/SE{SE}_{int(np.ceil(C_NS*3)):d}uM_complexes_concentration_melting-1.tsv', sep='\t').rename(columns={'# temperature':'T'})
    pSE = CubicSpline(pSEdata['T'], pSEdata['SE+SE']/(1.5*C_NS*1e-6))

    color = color_sequences['tab20c'][icolor]
    line = axs[0].plot(
        pSE(mintandelta[:,0]),
        mintandelta[:,1],
        ls='none', marker=marks[Y], color = color,
        label=f'Y{Y}SE{SE} {C_NS: >4d} µM'
        )[0]
    axs[1].plot(
        phi_rotating_assembled_only_saturated(
            pSE(mintandelta[:,0]),
            pNS(mintandelta[:,0]),
            SE=SE, C_0=C_NS*1e-6, Y=Y, persistence=40
            ),
        mintandelta[:,1],
        ls='none', marker=marks[Y], color=line.get_color(),
        #label=f'Y{Y}SE{SE} {C_NS: >4d} µM'
        )

axs[1].set_xlim(0,1.5)
axs[1].axvline(0.58, ls=':', color='k')
fig.legend(loc='outside right upper')
for ext in ['png', 'pdf']:
    plt.savefig(f'all_designs.{ext}')
