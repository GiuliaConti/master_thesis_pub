import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import warnings
import multiprocessing
import os, datetime
from astropy import units as u

from lisatools.diagnostic import *
from lisatools.sensitivity import SensitivityMatrix, A1TDISens, E1TDISens, LISASens
from lisatools.utils.constants import *
from lisatools.detector import ESAOrbits, EqualArmlengthOrbits
from lisatools.datacontainer import DataResidualArray 
from lisatools.analysiscontainer import AnalysisContainer

from few.waveform import GenerateEMRIWaveform
from few.utils.constants import *
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_p_at_t
from few.utils.fdutils import *

from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from scipy.signal.windows import tukey, hann, boxcar, nuttall, blackman
from eryn.backends import HDFBackend

from fastlisaresponse import pyResponseTDI, ResponseWrapper



# gpu
use_gpu = False



# metric
metric = "FastKerrEccentricEquatorialFlux"   # Kerr
traj = "KerrEccEqFlux"


# Observation parameters
Tobs = 1  # [years]
dt = 50.0  # [s]
eps = 1e-5  # mode content

emri_waveform_kwargs = dict(T=Tobs, dt=dt, eps=eps)



# Waveform parameters
M = 1e6  # central object mass
mu = 10  # secondary object mass
a = 0.5  # spin (will be ignored in Schwarzschild waveform)
p0 = 8.2  # initial semi-latus rectum
e0 = 0.5  # eccentricity
x0 = 1.0  # cosine of inclination 
dist = 1.0  # distance

qK = np.pi / 6  # polar spin angle (theta)
phiK = np.pi / 3  # azimuthal viewing angle
qS = np.pi / 6  # polar sky angle
phiS = np.pi / 3  # azimuthal viewing angle

Phi_phi0 = np.pi / 3
Phi_theta0 = np.pi / 6
Phi_r0 = np.pi / 3



emri_waveform_args = [
    M,
    mu,
    a,
    p0,
    e0,
    x0,
    dist,
    qS,
    phiS,
    qK,
    phiK,
    Phi_phi0,
    Phi_theta0,
    Phi_r0,
]



# TDI
tdi_chan="AE"
tdi_labels=["A", "E"]

orbit_file_esa = "equalarmlength-trailing-fit.h5"

orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

order = 25 # order of the langrangian interpolation

tdi_gen="1st generation"

response_kwargs = dict(
        Tobs=Tobs,
        dt=dt,
        t0 = 100000.0,  
        order = order, # order of the langrangian interpolation
        index_beta = 7,   # Sky location parameters: theta --> qS
        index_lambda = 8, #                          phi --> phiS
        tdi=tdi_gen, 
        tdi_chan=tdi_chan,
        orbit_kwargs=orbit_kwargs_esa,
    )



# Initialise FEW generator
td_gen = GenerateEMRIWaveform(
        metric,
        sum_kwargs=dict(pad_output=True, odd_len=True),
        return_list=False,
        use_gpu=use_gpu,
    )


# Initialise wrapper for TDI
lisa_response = ResponseWrapper(waveform_gen=td_gen,
                                flip_hx=True,
                                use_gpu=use_gpu,
                                remove_sky_coords=False,
                                is_ecliptic_latitude=False,
                                remove_garbage=True,
                                **response_kwargs)

def fastlisaresponse(*params, emri_waveform_kwargs=None):
    return lisa_response(*params, **(emri_waveform_kwargs or {}))




# generate FEW signal
start = time.time()
chans_few = td_gen(
    *emri_waveform_args,
    **emri_waveform_kwargs
)
print(f"Waveform generation took {time.time()-start:.2f} s")



# Generate TDI signal
chans_tdi = fastlisaresponse(
    *emri_waveform_args,
    emri_waveform_kwargs=emri_waveform_kwargs,
)




# Save signal and useful info
filename='signal_no_gpu.npz'
def save_signal_data(dt, chans_few, chans_tdi, 
                     filename=filename):
    # Save signal data to a .npz file
    
    # Convert from GPU to CPU if needed
    def to_numpy(data):
        # If it's a list or tuple, convert each element
        if isinstance(data, (list, tuple)):
            return np.array([to_numpy(item) for item in data])
        # If it's CuPy
        if hasattr(data, 'get'):
            return data.get()
        # If it's already NumPy or scalar
        else:
            return np.asarray(data)
    
    # Convert all data to NumPy
    dt_np = to_numpy(dt)
    chans_few_np = to_numpy(chans_few)
    chans_tdi_np = to_numpy(chans_tdi)
    
    # Save everything to a compressed file
    np.savez_compressed(
        filename,
        dt=dt_np,
        chans_few=chans_few_np,
        chans_tdi=chans_tdi_np
    )

    print()
    print(f"Data saved to '{filename}'")
    print(f"  - dt: {dt_np}")
    print(f"  - chans_few shape: {chans_few_np.shape}")
    print(f"  - chans_tdi shape: {chans_tdi_np.shape}")

save_signal_data(dt, chans_few, chans_tdi)










