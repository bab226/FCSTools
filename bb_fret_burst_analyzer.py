#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FRET Burst Analyzer
Purpose: 
1) Analyze photons from donor and acceptor.
2) Determine FRET efficiency from simulated PyBroMo data.
    
Updated on Wed Apr 28, 2021

@author: Bryan Bogin
"""

import fretbursts as fb
import glob
import lmfit
import pybroom as br
import os

print('lmfit version:', lmfit.__version__)
print('FretBursts version:', fb.__version__)
#%%

filename = list(glob.glob('smFRET_*'))

#Check if present:
if os.path.isfile(filename[0]):
    print("Perfect, I found the file!")
else:
    print("Who now what whyyyy...? This file no longer exists:\n%s" % filename)

#Import hdf5 data:
d = fb.loader.photon_hdf5(str(filename[0]))

print("\nNow analyzing: \n%s" %(d))
print("Total simulation time: %s" %(d.time_max))

#Plot photons over time:
fb.dplot(d, fb.timetrace)

#Calculate background (calc_bg):
d.calc_bg(fb.bg.exp_fit, time_s=0.5, tail_min_us='auto')
fb.dplot(d, fb.timetrace_bg)
fb.dplot(d, fb.hist_bg, period = 0)

#Sliding Window Burst Search:
d.burst_search(L=10, m=10, F=6)
d.burst_search()

#Narrowing selection of data:
ds = d.select_bursts(fb.select_bursts.size, add_naa=True, th1=30)

#Fitting the FRET Histogram:
model = fb.mfit.factory_three_gaussians()
model.print_param_hints()

#Modifying paramaters:
model.set_param_hint('p1_center', value=0.1, min=-0.1, max=0.3)
model.set_param_hint('p2_center', value=0.4, min=0.3, max=0.7)
model.set_param_hint('p2_sigma', value=0.04, min=0.02, max=0.18)
model.set_param_hint('p3_center', value=0.85, min=0.7, max=1.1)
model.print_param_hints()

#Fit and plot model:
E_fitter = fb.bext.bursts_fitter(ds, 'E', binwidth=0.03)

E_fitter.fit_histogram(model=model, pdf=False, method='nelder')
fb.dplot(ds, fb.hist_fret, show_model=True, pdf=False)

E_fitter.fit_histogram(model=model, pdf=False, method='leastsq')
fb.dplot(ds, fb.hist_fret, show_model=True, pdf=False)

res = E_fitter.fit_res[0]
res.params.pretty_print()

#lmfit's report (uncomment):
#print(res.fit_report(min_correl=0.5))

#Confidence interval in params:
#ci = res.conf_interval()
#lmfit.report_ci(ci)

E_fitter.params
df = br.tidy(res)

"""
Now use pandas functions like: df.loc[df.name.str.contains('p1')]

Can help select data
"""




