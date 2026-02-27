# This script parallelizes the forward model

#####################
import numpy as np
import pylab
from scipy.integrate import *
from scipy.interpolate import interp1d
from scipy import optimize
import scipy.optimize 
from radiative_functions import * ## This contains an improved climate function
from other_functions import *
from stellar_funs import main_sun_fun #stellar evolution function
from carbon_cycle_model import * #contains WHAK and MAC weathering functions
from escape_functions import * #contains atmospheric escape functions
from outgassing_module_fast import * ##VolcGases outgassing code
from Albedo_module import *
from thermodynamic_variables import *
import time #time module used to generate random seeds
from numba import jit # standard module for speeding up python, will need to install
from numba_nelder_mead import nelder_mead #fast optimizer
import os
from Hypso_fun import hypso_fun # hypsometric function for calculating land fraction from surface water volume
from Latent_Heat_Flux import latent_heat_flux

global  init_water, Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts
from planetary_carbon_forward_MC import forward_model #import forward model function
import pdb

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab

'''
To run the script 10000 times, you need it run it ten times in your terminal, for exampl:
#for i in {0..9}; do     sed -i "s/batch_index = 9
'''

##########################################################################
### Options
# Number of forward model calls in Monte Carlo calculations 
# 1000 provides approximation distrubitions, 10000 used in manuscript 
it_num=1000

# Parallelize on/off number
# 0 - no parallelization (slow)
# number of threads (e.g. 4, 6, 8, 10, 12)
Parallelize = 48


## Save options. Will later be called by the forward model.
options_array = np.array([Parallelize])
np.save('options_array.npy',options_array)

batch_index = 9
#########################################################################

#Dissolution_change_array=np.zeros(shape=it_num)
imbalance_array=[]   # This array will contain the mass conservation imbalance for each forward model run

## Run forward model once with arbitrary parameters to define size of array that will contain outputs
[temporary_array,imb]=forward_model(1.4e21, 0.5e21, 6.77e20, 1e19, 1e21, 1.5, 0.4, 11.1, 2e5)
temporary_array = np.array(temporary_array)


all_output = np.zeros((imb.shape[0], imb.shape[1], it_num))
init_water_array = np.zeros(it_num)
solid_H2O_array = np.zeros(it_num)
ts_array = np.zeros(it_num)
Te_array = np.zeros(it_num)
mm_array = np.zeros(it_num)
n_out_array = np.zeros(it_num)
solid_CO2_array = np.zeros(it_num)
fluid_CO2_array = np.zeros(it_num)
Max_mantle_H2O_array = np.zeros(it_num)


def try_run_forward(ii):
    ij=0
    while ij<1:
        print( ii," of ",it_num)

        ## Generate random seed for sampling parameter distributions
        mtime = int(time.time()) % (ii+100000) ### use remainder of this division as the random seed
        np.random.seed(mtime)
        
        ################################################################
        ### Sample uniform distribution for unknown parameters
        ### For each parameter, the two numbers in brackets define
        ### the range of their uniform distribution.
        ################################################################

        init_water = 10**np.random.uniform(np.log10(1.4e18), np.log10(1.4e21))
        Init_solid_H2O = 10**np.random.uniform(19, 21)
        Max_mantle_H2O = 10**np.random.uniform(20,22)
        #Max_mantle_H2O = 10**np.random.uniform(18,20)# small capacity
        mm = np.random.uniform(1.0,2.0)
        n_out=np.random.uniform(0,0.73)
        Te = np.random.uniform(5, 15)
        ts = 10**np.random.uniform(3, 5)
        
        # Randomize total carbon inventory across a plausible range
        Total_Carbon = 10**np.random.uniform(20,22) #BSE composition is 10^21

        # Randomize the partitioning fraction between mantle and atmosphere
        Mantle_fraction = np.random.uniform(0.01, 0.8)  #1% to 80% in the mantle

        # Calculate solid and fluid CO2 amounts
        Init_solid_CO2 = Mantle_fraction * Total_Carbon  # CO2 in the interior
        Init_fluid_CO2 = (1 - Mantle_fraction) * Total_Carbon  # CO2 at the surface/atmosphere
        
        params = [init_water, Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts]

        
        #make array of ~30 values
        #################################################################
        
        ## Attempt to run forward model
        try:
            output = forward_model(*params)
            if output is None or len(output) != 2:
                raise ValueError("Forward model returned invalid result.")
            time_vector, y_array = output
            
            if time_vector is None or y_array is None:
                print(f" Run {ii} failed cleanly, skipping this run.")
                
            all_output[:,:,ii] = y_array
            all_output[2,:,ii] = time_vector

            return ii,all_output[:,:,ii], *params #Return iteration number, carbon cycle outputs, and various input parameters.
            ij=ij+1
        except Exception as e: # if forward model call unsuccessful, print error message and try again
            #print(f"init_water: {init_water}") # Uncomment to print more details about failed runs
            print(f"error, forward model failed - try again: {e}")


### Non-parallelized version, run all forward model calls in same thread:
if Parallelize == 0:
    kk=0
    while kk<it_num:
        try:
            [jj,all_output[:,:,kk], init_water, Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts] = try_run_forward(kk) # fill in kk-th element of output array, and record mass imbalance
            #imbalance_array.append(imbalan)
            '''
            init_water_array[kk] = init_water
            ts_array[kk] = ts
            Te_array[kk] = Te
            mm_array[kk] = mm
            n_out_array[kk] = n_out
            solid_CO2_array[kk] = Init_solid_CO2
            fluid_CO2_array[kk] = Init_fluid_CO2
            Max_mantle_H2O_array[kk] = Max_mantle_H2O
            '''
            kk=kk+1
        except Exception as e:
            #print("Try again")
            print(f"Try again: {e}")

### Parallelized version, distribute forward model calls among 'Parallelize' number of threads
          
else:   
    items=range(it_num)
    import concurrent.futures
    with concurrent.futures.ProcessPoolExecutor(Parallelize) as executor: 
        for row,result, init_water, Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts in executor.map(try_run_forward, items):
            print(np.shape(result))
            print(np.shape(all_output[:,:,row]))
            all_output[:,:,row] = result
            init_water_array[row] = init_water  # Save init_water and other parameters for plotting
            solid_H2O_array[row] = Init_solid_H2O
            ts_array[row] = ts
            Te_array[row] = Te
            mm_array[row] = mm
            n_out_array[row] = n_out
            solid_CO2_array[row] = Init_solid_CO2
            fluid_CO2_array[row] = Init_fluid_CO2
            Max_mantle_H2O_array[row] = Max_mantle_H2O


            
##################################
###### Plotting ##########
##################################

import numpy as np
import pylab

# Set time axis from outputs
time_axis = all_output[2,:,0]

# Define plotting limits
strt_lim = 0.0  # Lower limit on x-axis (0 Ga)
fin_lim = 4.0   # Upper limit on x-axis (4 Ga)

####### Summary Plot - Figure 1 #######
pylab.figure(figsize=(13, 8))  

# Loop through each individual model run and plot separately
num_runs = all_output.shape[2]  # all_output is (variables, time, runs)

# Plot each individual model run for Surface Water
pylab.subplot(2, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[1, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Surface water (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# Surface Temperature
pylab.subplot(2, 2, 2)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[8, :, i], color='black', alpha=0.2, linewidth=1)
pylab.ylabel('Surface Temperature (K)')
pylab.legend(['Individual Runs'], loc="upper left")
pylab.xlim(1e0, 1e10)

# Precipitation
pylab.subplot(2, 2, 3)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[3, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Precipitation (m/yr)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# Atmosphere + Ocean CO2 Reservoir
pylab.subplot(2, 2, 4)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[12, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Atmosphere + Ocean CO$_2$ Reservoir mass (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

time_axis = all_output[2, :, 0]  # time data

######################## Water Plots (Figure 2) ########################
pylab.figure(figsize=(13, 8))

# Mantle Water
pylab.subplot(2, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[0, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Mantle H$_2$O (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# Atmospheric Water
pylab.subplot(2, 2, 2)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[18, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Atmospheric H$_2$O (kg)')
pylab.xlim(1e0, 1e10)

# Ocean Water
pylab.subplot(2, 2, 3)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[20, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean H$_2$O (kg)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# Total Water
pylab.subplot(2, 2, 4)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[0, :, i] + all_output[1, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Total H$_2$O (kg)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)


######################## CO2 Flux Plots (Figure 3) ########################
pylab.figure(figsize=(13, 8))

# CO2 Weathering
pylab.subplot(3, 2, 1)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('CO$_2$ Weathering Flux (Tmol CO$_2$/yr)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# CO2 Outgassing
pylab.subplot(3, 2, 2)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[15, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('CO$_2$ Outgassing Flux (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

# Net CO2
pylab.subplot(3, 2, 3)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[15, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Net CO$_2$ Flux (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

# Subaerial CO2
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[11, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Subaerial CO$_2$ (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

# Submarine CO2
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[16, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Submarine CO$_2$ (Tmol CO$_2$/yr)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# Mantle CO2
pylab.subplot(3, 2, 6)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[13, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Mantle CO$_2$ (kg)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)


######################## Atmospheric Pressure (Figure 4) ########################
pylab.figure(figsize=(13, 8))

# fH2O
pylab.subplot(3, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[21, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('fH$_2$O Pressure (bar)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# fCO2
pylab.subplot(3, 2, 2)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[23, :, i] / 1e5, color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('fCO$_2$ Pressure (bar)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# Outgassing total
pylab.subplot(3, 2, 3)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[26, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Total Outgassing Flux (kg/s)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# CO2 Ocean
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[17, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean CO$_2$ Mass (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# CO2 Atmosphere
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[19, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Atmosphere CO$_2$ Mass (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)


# CO2 Atmosphere + Ocean Check
pylab.subplot(3, 2, 6)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[17, :, i] + all_output[19, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Atmo+Ocean CO$_2$ Mass Check (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)



######################## Mantle Water and Land Fraction (Figure 5) ########################
pylab.figure(figsize=(13, 8))

# H2O Degassing
pylab.subplot(3, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[28, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('H$_2$O Degassing (kg/s)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# H2O Mantle Return
pylab.subplot(3, 2, 2)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[27, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('H$_2$O Mantle Return (kg/s)')
pylab.xlim(1e0, 1e10)

# net H2O Mantle Flux
pylab.subplot(3, 2, 3)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[27, :, i] - all_output[28, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Net H$_2$O Mantle Flux (kg/s)')
pylab.xlim(1e0, 1e10)

# ocean depth
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[5, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean Depth (m)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# ocean fraction
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[25, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean Fraction')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# land fraction
pylab.subplot(3, 2, 6)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[6, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Land Fraction')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

######################## Final surface temperature plots (Figure 6) #######################

pylab.figure(figsize=(13, 10))

# Scatter plot of initial water vs. final temperature
pylab.subplot(2, 2, 1) 
for i in range(it_num):
    pylab.scatter(init_water_array[i]/1.4e21, all_output[8, -1, i], color='darkcyan', alpha=0.6, edgecolor='k')
pylab.title('Initial Water vs. Final Surface Temperature')
pylab.xscale('log') 
min_earth_oceans = 7e17 / 1.4e21  # Convert xlim values
max_earth_oceans = 2.4e21 / 1.4e21
pylab.xlim(min_earth_oceans, max_earth_oceans)
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)
pylab.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])


# initial carbon vs. final temp
pylab.subplot(2, 2, 2)
for i in range(it_num):
    pylab.scatter(all_output[12, 0, i], all_output[8, -1, i], color='mediumpurple', alpha=0.6, edgecolor='k')
pylab.xlabel('Initial Carbon Mass Atmo+Ocean (kg)')
pylab.ylabel('Final Surface Temperature (K)')
pylab.title('Initial Surface Carbon vs. Final Surface Temperature')
pylab.xscale('log')  
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)


# initial interior water vs. final temp
pylab.subplot(2, 2, 3)
for i in range(it_num):
    pylab.scatter(all_output[0, 0, i], all_output[8, -1, i], color='darkcyan', alpha=0.6, edgecolor='k')
pylab.xlabel('Initial Interior Water Mass (kg)')
pylab.ylabel('Final Surface Temperature (K)')
pylab.title('Initial Interior Water vs. Final Surface Temperature')
pylab.xscale('log')  
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)


### Probability that the final temperature is > 400 K
pylab.subplot(2,2,4)

initial_water = init_water_array # Initial surface water
final_temperature = all_output[8, -1, :]  # Final surface temperature

# define log bins
num_bins = 10

iw_min = 1e18   # 0.01 Earth oceans
iw_max = 2e21   # 1.00 Earth oceans

water_bins = np.logspace(np.log10(iw_min), np.log10(iw_max), num_bins + 1)
bin_centers = np.sqrt(water_bins[:-1] * water_bins[1:]) #geometric mean


# probability: T > 400 K
probability_above_400K = []
probability_water = []
bin_counts = []

for i in range(num_bins):

    if i == num_bins - 1:
        idx = (initial_water >= water_bins[i]) & (initial_water <= water_bins[i + 1])  
    else:
        idx = (initial_water >= water_bins[i]) & (initial_water < water_bins[i + 1])

    
    # filter out NaNs only within the bin
    idx = idx & (~np.isnan(final_temperature)) & (initial_water > 0)
    n_bin = np.sum(idx)
    bin_counts.append(n_bin)
    
    if np.sum(idx) > 0:
        p_temp = np.sum(final_temperature[idx] > 400) / n_bin
    else:
        p_temp = 0

    probability_above_400K.append(p_temp)

# Plot: Final Temp > 400 K
pylab.plot(bin_centers / 1.4e21, probability_above_400K, marker='o', linestyle='-', color='darkred')
pylab.xscale('log')
pylab.xlabel('Initial Surface Water (Earth oceans)')
pylab.ylabel('Probability (Final T > 400 K)')
pylab.title('Probability of High Final Temperature vs. Initial Water')
pylab.grid(True)
pylab.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])



#############  Time to runaway warming (figure 7)    ###########
pylab.figure(figsize=(13, 10))

#### How long it takes the model to reach Runaway Warming (excluding models already in runaway) ####
runaway_threshold = 400  # K

# initialize with NaNs
runaway_time = np.full(num_runs, np.nan)

for i in range(num_runs):
    temps = all_output[8, :, i]

    # exclude runs that start already in runaway
    if temps[0] > runaway_threshold:
        continue

    # search for first irreversible runaway point
    for j in range(len(temps)):
        if temps[j] > runaway_threshold and np.all(temps[j:] > runaway_threshold):
            runaway_time[i] = time_axis[j]
            break  # Stop once found

# filter valid cases   
valid_mask = ~np.isnan(runaway_time) & (runaway_time > 0)
valid_runaway_times = runaway_time[valid_mask]

# summary stats   
num_runaway = np.sum(valid_mask)
num_non_runaway = num_runs - num_runaway

### plotting the duration of habitability until runaway warming for each run
pylab.subplot(2, 2, 1)

if len(valid_runaway_times) == 0:
    # Plot an empty histogram with placeholder message
    pylab.hist([], bins=50, color='salmon', edgecolor='black', alpha=0.7)
    pylab.title("No Irreversible Runaway Greenhouse Events")
    pylab.ylabel("Probability Density")
    pylab.grid(True)

else:
    median_runaway = np.median(valid_runaway_times)
    mean_runaway = np.mean(valid_runaway_times)
    bins = np.linspace(0, np.max(valid_runaway_times) * 1.05, 50)

    pylab.hist(valid_runaway_times, bins=bins, color='#56B4E9', edgecolor='black', alpha=0.7, density=True)
    pylab.axvline(median_runaway, color='#E69F00', linestyle='-', linewidth=2, label=f'Median = {median_runaway:.2e} Gyr')
    pylab.axvline(mean_runaway, color='#F0E442', linestyle='-', linewidth=2, label=f'Mean = {mean_runaway:.2e} Gyr')

    pylab.xlabel("Time to Runaway Warming (Gyr)")
    pylab.ylabel("Probability Density")
    pylab.legend()
    pylab.grid(True)


## Histogram of initial water sampling - this makes sure we are sampling the initial surface water evenly
initial_water_EO = all_output[1, 0, :] / 1.4e21  # Convert to Earth oceans

pylab.subplot(2, 2, 1)
pylab.hist(initial_water_EO, bins=np.logspace(np.log10(min_earth_oceans), np.log10(max_earth_oceans), 50),
           color='skyblue', edgecolor='black')
pylab.xscale('log')
pylab.xlabel("Initial Surface Water Mass (Earth oceans)")
pylab.ylabel("Frequency")
pylab.title("Sampling of Initial Surface Water")



####### figure 8 other scatter plots
total_water = all_output[0, 0, :]+all_output[1, 0, :]
mantle_water = all_output[0, 0, :]


# parameters and labels 
params = [
    (mantle_water / 1.4e21, "Initial Mantle Water (Earth oceans)", True),
    (ts_array, "ts (s)", True),
    (Te_array, "Te (K)", False),
    (mm_array, "mm", False),
    (n_out_array, "n_out", False),
    (solid_CO2_array, "Solid CO$_2$ (kg)", True),
    (fluid_CO2_array, "Fluid CO$_2$ (kg)", True),
    (Max_mantle_H2O_array/1.4e21, "Max Mantle H$_2$O (Earth oceans)", True),
    (total_water/ 1.4e21, "Total Water (Earth oceans)", True)
]


fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()


for i, (param_array, label, use_log) in enumerate(params):
    param = np.array(param_array)
    temp = np.array(final_temperature)

    # make sure param and temp have the same shape
    if param.shape != temp.shape:
        min_len = min(len(param), len(temp))
        param = param[:min_len]
        temp = temp[:min_len]

    # filter out invalid entries
    valid = (~np.isnan(param)) & (~np.isnan(temp)) & np.isfinite(param) & np.isfinite(temp)
    param = param[valid]
    temp = temp[valid]

    # if all data was filtered out, skip the plot
    if len(param) == 0:
        axs[i].text(0.5, 0.5, "No valid data", transform=axs[i].transAxes,
                    ha='center', va='center')
        axs[i].axis("off")
        continue

    axs[i].scatter(param, temp, s=10, alpha=0.6, edgecolor='k')
    axs[i].set_xlabel(label)
    axs[i].set_ylabel("Final Surface Temp (K)")
    if use_log:
        axs[i].set_xscale('log')
        plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])
    axs[i].grid(True)



######## figure 9 probability plots for other Monte Carlo parameters #########
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()

### probability calculation loop
for i, (param_array, label, use_log) in enumerate(params):
    param = np.array(param_array)
    temp = np.array(final_temperature)

    if param.shape != temp.shape:
        min_len = min(len(param), len(temp))
        param = param[:min_len]
        temp = temp[:min_len]

    valid = (~np.isnan(param)) & (~np.isnan(temp)) & np.isfinite(param) & np.isfinite(temp) & (param > 0)
    param = param[valid]
    temp = temp[valid]


    # define bins
    num_bins = 10
    if use_log:
        bins = np.logspace(np.log10(np.min(param)), np.log10(np.max(param)), num_bins+1)
        bin_centers_params = np.sqrt(bins[:-1]* bins[1:]) #geometric mean
    else:
        bins = np.linspace(np.min(param), np.max(param), num_bins +1)
        bin_centers_param = 0.5 * (bins[:-1] + bins[1:]) # arithmetic mean

    # calculate probabilities
    prob_temp = []
    for j in range(num_bins):
        in_bin = (param >= bins[j]) & (param < bins[j+1]) if j < num_bins - 1 else (param >= bins[j]) & (param <= bins[j+1])
        if np.sum(in_bin) > 0:
            prob = np.sum(temp[in_bin] > 400)/np.sum(in_bin)
        else:
            prob = np.nan
        prob_temp.append(prob)

    # plotting
    axs[i].plot(bin_centers_params, prob_temp, marker = 'o', linestyle='-', color = 'palevioletred')
    axs[i].set_xlabel(label)
    axs[i].set_ylabel("P(Final T > 400K)")
    axs[i].grid(True)
    if use_log:
        axs[i].set_xscale('log')
        plt.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])
    axs[i].set_ylim(-0.1,1.1)

# hide unused subplots
for j in range(len(params), len(axs)):
    axs[j].axis("off")




pylab.tight_layout()




#    Define the main output directory    
output_directory = "Figures/folder_1/folder_2/"
os.makedirs(output_directory, exist_ok=True)
print("Saving to:", os.path.join(output_directory))

  
sheet_switch = "y"
if sheet_switch == "y":
    ## probability data
    directory_prob = "ProbabilityFigure"
    filename_base = output_directory.replace("/", "_")
    filename = f"{filename_base}.csv"
    data_prob = np.column_stack((bin_centers / 1.4e21, probability_above_400K))
    df = pd.DataFrame(data_prob, columns=["Initial_Water_Mass", "Probability"])
    df.to_csv(os.path.join(directory_prob, filename), index=False)


# plotting and saving Figures   
plot_switch = "y"
if plot_switch == "y":
    fig_size = (13, 8)
    figure_names = [
        "figure1.png", "figure2.png", "figure3.png", "figure4.png",
        "figure5.png", "figure6.png", "figure7.png", "figure8.png", "figure9"
    ]

    for i, figure_name in enumerate(figure_names):
        pylab.figure(i + 1, figsize=fig_size)
        save_path = os.path.join(output_directory, figure_name) 
        pylab.savefig(save_path, dpi=300, bbox_inches='tight')

# save outputs  
np.save(os.path.join(output_directory, f"batch_output_{batch_index}.npy"), all_output)
if batch_index == 0:
    np.save(os.path.join(output_directory, 'time_axis.npy'), time_axis)
else:
    np.save(os.path.join(output_directory, f"time_axis_{batch_index}.npy"), time_axis)

np.savez(os.path.join(output_directory, f"inputs_{batch_index}.npz"),
         ts=ts_array,
         Te=Te_array,
         mm=mm_array,
         n_out=n_out_array,
         solid_CO2=solid_CO2_array,
         fluid_CO2=fluid_CO2_array,
         Max_mantle_H2O=Max_mantle_H2O_array,
         init_water=init_water_array,
         solid_H2O = solid_H2O_array)



input_params = {
    'it_num': it_num,
    'Parallelize': Parallelize,
    'init_water': ( np.log10(1.4e18), np.log10(1.4e21)),  # initial water (log scale)
    'Init_solid_H2O': (19, 21), # initial solid H2O (log scale)
    'Max_mantle_H2O': (20,22), # for max mantle H2O (log scale)
    'mm': (1.0, 2.0),           # Range for mm
    'n_out': (0.1, 0.73),        # Range for n_out
    'Te': (5, 15),              # Range for Te
    'ts': (3, 5),              # Range for ts (log scale)
    'Melt_volume_past': (950, 1580), #Range for melt volume past
    'Total_Carbon': (20, 22),   # Range for total carbon (log scale)
    'Mantle_fraction': (0.01, 0.8), #Range for mantle fraction
}

#pylab.show()

