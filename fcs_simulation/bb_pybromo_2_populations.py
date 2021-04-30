# -*- coding: utf-8 -*-
"""
FRET Simulator

Usage: python bb_pybromo_2_populations.py argument(1)
argument() = integer

Run using terminal unless troubleshooting.

By Bryan Bogin
Updated 4/27/21
"""
print('Adapted from James Losey, Moradi Lab, Arkansas')

import sys
from pathlib import Path
from textwrap import dedent, indent
import numpy as np
import tables
import matplotlib.pyplot as plt
#import seaborn as sns
import pybromo as pbm
from scipy.stats import expon
import phconvert as phc
print('PyTables version:', tables.__version__)
print('PyBroMo version:', pbm.__version__)


for x in sys.argv:
     print ("Argument: %s" %(x))

#sd = int(sys.argv[1]) FIXME
sd = 1

#def efficiency(r,r0):
#    return 1./(1. + 0.975*(r/r0)**2.65) 

# Initialize the random state
rs = np.random.RandomState(seed=sd)
print('Initial random state:', pbm.hashfunc(rs.get_state()))

# Simulation time step (seconds)
t_step = 0.5e-6

# Time duration of the simulation (seconds)
t_max = 2

# Diffusion coefficient
D1u = 30. # um^s/s
D1 = D1u*(1e-6)**2

#D2u = 6. # um^s/s
#D2 = D2u*(1e-6)**2
# Simulation box definition
box = pbm.Box(x1=-4.e-6, x2=4.e-6, y1=-4.e-6, y2=4.e-6, z1=-6e-6, z2=6e-6)

# Particles definition
n1 = 10
n2 = 0
#nn = n1+n2
P = pbm.Particles.from_specs(num_particles=(n1,)
        ,D=(D1,)
        ,box=box
        ,rs=rs)

# PSF definition
#psf = pbm.NumericPSF()
#psf = pbm.GaussianPSF(sx=0.3e-6,sy=0.3e-6,sz=0.5e-6)
psf = pbm.NumericPSF()

# Particle simulation definition
S = pbm.ParticlesSimulation(t_step=t_step, t_max=t_max, 
                            particles=P, box=box, psf=psf)

###Run simulation:
#Note that the save_pos=True seems to be very important in order to use the
#particle simulation visualizer (PSV)
print('Current random state:', pbm.hashfunc(rs.get_state()))
S.simulate_diffusion(total_emission=False, save_pos=True, verbose=True,
                     rs=rs, chunksize=2**19, chunkslice='times')
print('Current random state:', pbm.hashfunc(rs.get_state()))
print(S.compact_name())

#%% Simulate FRET:
#E1 = efficiency(40.,56.)
#E2 = efficiency(65.,56.)
params1 = dict(
    em_rates = (200e3,200e3), #em_list,    # Peak emission rates (cps) for each population (D+A)
    E_values = (0.40,0.75),     # FRET efficiency for each population
    num_particles = (n1,n2), #p_list,   # Number of particles in each population
    bg_rate_d = 1800,       # Poisson background rate (cps) Donor channel
    bg_rate_a = 1200,        # Poisson background rate (cps) Acceptor channel
    )
sim1 = pbm.TimestampSimulation(S, **params1)
sim1.summarize()
#rs = np.random.RandomState(sd)
#print('Test E to DA')
#print(em_rates_from_E_DA([1000],[0.3]))
sim1.run(rs=rs, 
            overwrite=True,      # overwite existing timstamp arrays
            skip_existing=True,  # skip simulation of existing timestamps arrays to save time
            save_pos=False,       # save particle position at emission time
           )
sim1.merge_da()
str1 = sim1.__str__()+"\nTimestep,Time,Channel(0=D,1=A),ParticleNum"
ts_1 = sim1.ts 
dt_1 = sim1.a_ch.astype('uint8')
p_1 = sim1.part
sim1.save_photon_hdf5()
print(ts_1.shape,dt_1)
np.savetxt(f"timestamp_nonlangevin_{sd}.txt"
        ,np.column_stack((ts_1,ts_1*t_step*0.1,dt_1,p_1))
        ,fmt=['%d','%f','%d','%d']
        ,header = str1.lstrip()
        )
