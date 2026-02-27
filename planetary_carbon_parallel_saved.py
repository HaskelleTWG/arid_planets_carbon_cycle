# This script reproduces plots from CSV files. It is needed to stitch together the full 10000 model plots. 
# each batch has 1000 runs, this script puts all 10 batches together
# You can also remake plots without rerunning the code.

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
from outgassing_module_fast import * ##Fast version of Nick's outgassing code
from Albedo_module import *
from thermodynamic_variables import *
import time #time module used to generate random seeds
from numba import jit # standard module for speeding up python, will need to install
from numba_nelder_mead import nelder_mead #fast optimizer, hopefully just works once numba is installed
import os
from Hypso_fun import hypso_fun ### hypsometric function for calculating land fraction from surface water volume
from Latent_Heat_Flux import latent_heat_flux

global  Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts, Melt_volume_past
from planetary_carbon_forward_MC import forward_model #import forward model function
import pdb

from astropy.io import ascii
from astropy.table import Table, Column, MaskedColumn
import pandas as pd

import numpy as np
import pylab
import os
import glob
import matplotlib.pyplot as plt
import pylab

it_num = 10000

# load the data from your file directory 
output_directory = "Figures/folder_1/folder_2/"  


try:
    #load and sort all batch files
    batch_files = sorted(glob.glob(os.path.join(output_directory, 'batch_output_*.npy')))
    if not batch_files:
        raise FileNotFoundError

    all_batches = [np.load(f) for f in batch_files]
    all_output = np.concatenate(all_batches, axis=2)  # Concatenate over the run axis


    #load time axis 
    time_axis = np.load(os.path.join(output_directory, 'time_axis.npy'))

except FileNotFoundError:
    print(f"Error: Data files not found in {output_directory}. Did you run the Monte Carlo simulation first?")
    exit()

'''
#### uncomment to test only one output, not the entire folder
try:
    all_output = np.load(os.path.join(output_directory, 'batch_output_4.npy'))
    time_axis = np.load(os.path.join(output_directory, 'time_axis_4.npy'))
    
except FileNotFoundError:
    print(f"Error: Data files not found in {output_directory}.  Did you run the Monte Carlo simulation first?")
    exit()  # Or handle the error in another way

'''

#Load and sort all input .npz files
input_files = sorted(glob.glob(os.path.join(output_directory, "inputs_*.npz")))
if not input_files:
    raise FileNotFoundError("No input .npz files found.")

# initialize lists to collect arrays
ts_list = []
Te_list = []
mm_list = []
n_out_list = []
solid_CO2_list = []
fluid_CO2_list = []
Max_mantle_H2O_list = []
init_water_list = []


# loop over each file and append arrays
for f in input_files:
    data = np.load(f)
    ts_list.append(data['ts'])
    Te_list.append(data['Te'])
    mm_list.append(data['mm'])
    n_out_list.append(data['n_out'])
    solid_CO2_list.append(data['solid_CO2'])
    fluid_CO2_list.append(data['fluid_CO2'])
    Max_mantle_H2O_list.append(data['Max_mantle_H2O'])
    init_water_list.append(data['init_water'])

    

#concatenate arrays
ts_array = np.concatenate(ts_list)
Te_array = np.concatenate(Te_list)
mm_array = np.concatenate(mm_list)
n_out_array = np.concatenate(n_out_list)
solid_CO2_array = np.concatenate(solid_CO2_list)
fluid_CO2_array = np.concatenate(fluid_CO2_list)
Max_mantle_H2O_array = np.concatenate(Max_mantle_H2O_list)
init_water_array = np.concatenate(init_water_list)


##################################
### Plotting 
##################################


# set time axis from outputs
time_axis = all_output[2,:,0]

# Define plotting limits
strt_lim = 0.0  # lower limit on x-axis (0 Ga)
fin_lim = 4.0   # upper limit on x-axis (4 Ga)


####### Figure 1 #######
pylab.figure(figsize=(13, 8))  

# Loop through each individual model run and plot separately
num_runs = all_output.shape[2]  # all_output is (variables, time, runs)

# Plot each individual model run for Surface Water
pylab.subplot(2, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[1, :, i], color='darkcyan', alpha=0.2)
pylab.ylabel('Surface water (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# individual model runs for Surface Temperature
pylab.subplot(2, 2, 2)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[8, :, i], color='black', alpha=0.2, linewidth=1)
pylab.ylabel('Surface Temperature (K)')
pylab.legend(['Individual Runs'], loc="upper left")
pylab.xlim(1e0, 1e10)

# individual model runs for Precipitation
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


######################## Water Plots (Figure 2) ########################
pylab.figure(figsize=(13, 8))

#mantle water
pylab.subplot(2, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[0, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Interior H$_2$O (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# atmospheric water
pylab.subplot(2, 2, 2)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[18, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Atmospheric H$_2$O (kg)')
pylab.xlim(1e0, 1e10)

# ocean water
pylab.subplot(2, 2, 3)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[20, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean H$_2$O (kg)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# Total water
pylab.subplot(2, 2, 4)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[0, :, i] + all_output[1, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Total H$_2$O (kg)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)


######################## CO2 Flux Plots (Figure 3) ########################
pylab.figure(figsize=(13, 8))

#CO2 Weathering
pylab.subplot(3, 2, 1)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('CO$_2$ Weathering Flux (Tmol CO$_2$/yr)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

#CO2 Outgassing
pylab.subplot(3, 2, 2)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[15, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('CO$_2$ Outgassing Flux (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

#net CO2
pylab.subplot(3, 2, 3)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[15, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Net CO$_2$ Flux (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

# subaerial CO2
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[11, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Subaerial CO$_2$ (Tmol CO$_2$/yr)')
pylab.xlim(1e0, 1e10)

# submarine CO2
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[16, :, i] - all_output[14, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Submarine CO$_2$ (Tmol CO$_2$/yr)')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

# mantle CO2
pylab.subplot(3, 2, 6)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[13, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Interior CO$_2$ (kg)')
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

#CO2 Ocean
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[17, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean CO$_2$ Mass (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

#CO2 Atmosphere
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[19, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Atmosphere CO$_2$ Mass (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)


#CO2 Atmosphere + Ocean Check
pylab.subplot(3, 2, 6)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[17, :, i] + all_output[19, :, i], color='mediumpurple', alpha=0.2, linewidth=1)
pylab.ylabel('Atmo+Ocean CO$_2$ Mass Check (kg)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)



######################## Mantle Water and Land Fraction (Figure 5) ########################
pylab.figure(figsize=(13, 8))

# H2O degassing
pylab.subplot(3, 2, 1)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[28, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('H$_2$O Degassing (kg/s)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

# H2O mantle return
pylab.subplot(3, 2, 2)
for i in range(num_runs):
    pylab.loglog(time_axis, all_output[27, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('H$_2$O Interior Return (kg/s)')
pylab.xlim(1e0, 1e10)

# net H2O mantle flux
pylab.subplot(3, 2, 3)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[27, :, i] - all_output[28, :, i], color='darkcyan', alpha=0.2, linewidth=1)
pylab.ylabel('Net H$_2$O Flux (kg/s)')
pylab.xlim(1e0, 1e10)

#ocean depth
pylab.subplot(3, 2, 4)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[5, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean Depth (m)')
pylab.legend(['Individual Runs'], loc="lower left")
pylab.xlim(1e0, 1e10)

#ocean fraction
pylab.subplot(3, 2, 5)
for i in range(num_runs):
    pylab.semilogx(time_axis, all_output[25, :, i], color='green', alpha=0.2, linewidth=1)
pylab.ylabel('Ocean Fraction')
pylab.xlabel('Time (yrs)')
pylab.xlim(1e0, 1e10)

#land fraction
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
    pylab.scatter(all_output[1, 0, i]/1.4e21, all_output[8, -1, i], color='darkcyan', alpha=0.6, edgecolor='k')
pylab.xlabel('Initial Surface Water Mass (Earth oceans)')
pylab.ylabel('Final Surface Temperature (K)')
pylab.title('Initial Water vs. Final Surface Temperature')
pylab.xscale('log')  # Log scale for water mass
min_earth_oceans = 1.2e18 / 1.4e21  # Convert to original xlim values
max_earth_oceans = 1.6e21 / 1.4e21
pylab.xlim(min_earth_oceans, max_earth_oceans)
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)


# Scatter plot of initial carbon vs. final temp
pylab.subplot(2, 2, 2)
for i in range(it_num):
    pylab.scatter(all_output[12, 0, i], all_output[8, -1, i], color='mediumpurple', alpha=0.6, edgecolor='k')
pylab.xlabel('Initial Carbon Mass Atmo+Ocean (kg)')
pylab.ylabel('Final Surface Temperature (K)')
pylab.title('Initial Surface Carbon vs. Final Surface Temperature')
pylab.xscale('log')  # Log scale for water mass
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)


# Scatter plot of initial mantle water vs. final temp
pylab.subplot(2, 2, 3)
for i in range(it_num):
    pylab.scatter(all_output[0, 0, i], all_output[8, -1, i], color='darkcyan', alpha=0.6, edgecolor='k')
pylab.xlabel('Initial Interior Water Mass (kg)')
pylab.ylabel('Final Surface Temperature (K)')
pylab.title('Initial Interior Water vs. Final Surface Temperature')
pylab.xscale('log')  # Log scale for water mass
pylab.grid(True, which="both", linestyle="--", linewidth=0.5)

initial_water = all_output[1, 0, :]
final_temperature = all_output[8, -1, :]
final_water = all_output[1, -1, :]

##### Probability that the final temperature is > 400 K
pylab.subplot(2, 2, 4)

# Define log bins
num_bins = 10

iw_min = 1e18   # 0.01 Earth oceans
iw_max = 2e21   # 1.00 Earth oceans

water_bins = np.logspace(np.log10(iw_min), np.log10(iw_max), num_bins + 1)
bin_centers = np.sqrt(water_bins[:-1] * water_bins[1:]) #geometric mean


# Probability that final temperature > 400 K
probability_above_400K = []
probability_water = []
bin_counts = []

for i in range(num_bins):

    if i == num_bins - 1:
        idx = (initial_water >= water_bins[i]) & (initial_water <= water_bins[i + 1])  # include upper edge
    else:
        idx = (initial_water >= water_bins[i]) & (initial_water < water_bins[i + 1])

    
    #filter out NaNs only within the bin
    idx = idx & (~np.isnan(final_temperature)) & (initial_water > 0)
    n_bin = np.sum(idx)
    bin_counts.append(n_bin)
    
    if np.sum(idx) > 0:
        p_temp = np.sum(final_temperature[idx] > 400) / n_bin
        p_water = np.sum(final_water[idx] > 1e10) / n_bin
    else:
        p_temp = 0
        p_water = 0

    probability_above_400K.append(p_temp)
    probability_water.append(p_water)

# Plot of Final Temp > 400 K
pylab.plot(bin_centers / 1.4e21, probability_above_400K, marker='o', linestyle='-', color='darkred')
pylab.xscale('log')
pylab.xlabel('Initial Surface Water (Earth oceans)')
pylab.ylabel('Probability (Final T > 400 K)')
pylab.title('Probability of High Final Temperature vs. Initial Water')
pylab.grid(True)
pylab.xticks([0.001, 0.01, 0.1, 1.0], ["0.1%", "1%", "10%", "100%"])




###################### Figure 7 ######################
pylab.figure(figsize=(13, 10))

#### How long it takes the model to reach Runaway Warming (excluding instant runaways) ####
runaway_threshold = 400  # K

# Initialize with NaNs
runaway_time = np.full(num_runs, np.nan)

sustained_duration = 1e8  # 100 Myr
dt = time_axis[1] - time_axis[0]  # Time step in years
sustained_steps = int(sustained_duration / dt)


for i in range(num_runs):
    temps = all_output[8, :, i]
    if temps[0] > runaway_threshold:
        continue
    found_runaway = False

    for j in range(len(temps) - sustained_steps):
        window = temps[j:j + sustained_steps]
        if np.all(window > runaway_threshold):
            runaway_time[i] = time_axis[j]
            found_runaway = True
            break
        
    if not found_runaway:
        for j in range(len(temps)):
            if temps[j] > runaway_threshold and np.all(temps[j:] > runaway_threshold):
                runaway_time[i] = time_axis[j]
                break  # Stop once found

# filter valid cases 
valid_mask = ~np.isnan(runaway_time) & (runaway_time > 0)
valid_runaway_times = runaway_time[valid_mask]

# summary stats
runaway_data = valid_runaway_times/1e9
num_runaway = np.sum(valid_mask)
num_non_runaway = num_runs - num_runaway
median_runaway = np.median(runaway_data)
mean_runaway = np.mean(runaway_data)

### plotting the duration of habitability
pylab.subplot(2, 2, 1)
bins = np.linspace(0, np.max(runaway_data) * 1.05, 50)

pylab.hist(runaway_data, bins=bins, color='#56B4E9', edgecolor='black', alpha=0.6, density=True)
pylab.axvline(median_runaway, color='#E69F00', linestyle='--', linewidth=2, label=f'Median = {median_runaway:.2e} Gyr')
pylab.axvline(mean_runaway, color='#F0E442', linestyle='--', linewidth=2, label=f'Mean = {mean_runaway:.2e} Gyr')

pylab.xlabel("Time to Runaway Warming (Gyr)")
pylab.ylabel("Probability Density")
pylab.legend()
pylab.grid(True)

print(f"Number of runaway runs plotted in histogram: {num_runaway}")
print(f"Number of runaway runs not plotted in histogram: {num_non_runaway}")


# Histogram of initial water sampling - this makes sure we are sampling the initial surface water evenly
# make a mask to find valid data, and normalize initial water values
valid = (~np.isnan(initial_water)) & (initial_water > 0)
initial_water_clean = initial_water[valid] / 1.4e21  # convert to Earth oceans

# compute bin edges from clean data
min_ew = np.min(initial_water_clean)
max_ew = np.max(initial_water_clean)
num_bins = 50

# Plot histogram
pylab.subplot(2, 2, 2)
pylab.hist(initial_water_clean, bins=np.logspace(np.log10(min_ew), np.log10(max_ew), num_bins),
           color='skyblue', edgecolor='black')
pylab.xscale('log')
pylab.xlabel("Initial Surface Water Mass (Earth oceans)")
pylab.ylabel("Frequency")




########## Figure 8 scatter plots for other Monte Carlo parameters #############

import matplotlib.pyplot as plt
final_Tsurf = all_output[8, -1, :]
total_water = all_output[0, 0, :]+all_output[1, 0, :]
mantle_water = all_output[0, 0, :]


# parameters and their labels 
params = [
    (mantle_water / 1.4e21, "Initial Interior Water (Earth oceans)", True),
    (ts_array, "Soil Age $ts$ (yr)", True),
    (Te_array, "Kinetic Weathering Temp. Dependence $Te$ (K)", False),
    (mm_array, " Outgassing exponent $m$", False),
    (n_out_array, "Internal heat flow exponent $n_{out}$", False),
    (solid_CO2_array, "Initial Interior CO$_2$ (kg)", True),
    (fluid_CO2_array, "Initial Surface CO$_2$ (kg)", True),
    (Max_mantle_H2O_array/1.4e21, "Max Interior H$_2$O (Earth oceans)", True),
    (total_water/ 1.4e21, "Total Water (Earth oceans)", True)
]


fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()


for i, (param_array, label, use_log) in enumerate(params):
    param = np.array(param_array)
    temp = np.array(final_Tsurf)

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
    axs[i].grid(True)


######## figure 9 probability plots for other Monte Carlo parameters #########
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
axs = axs.flatten()

### Probability calculation loop
for i, (param_array, label, use_log) in enumerate(params):
    param = np.array(param_array)
    temp = np.array(final_Tsurf)

    if param.shape != temp.shape:
        min_len = min(len(param), len(temp))
        param = param[:min_len]
        temp = temp[:min_len]

    valid = (~np.isnan(param)) & (~np.isnan(temp)) & np.isfinite(param) & np.isfinite(temp) & (param > 0)
    param = param[valid]
    temp = temp[valid]


    #Define bins
    num_bins = 10
    if use_log:
        bins = np.logspace(np.log10(np.min(param)), np.log10(np.max(param)), num_bins+1)
        bin_centers_param = np.sqrt(bins[:-1]* bins[1:]) #geometric mean
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

    # Plotting
    axs[i].plot(bin_centers_param, prob_temp, marker = 'o', linestyle='-', color = 'palevioletred')
    axs[i].set_xlabel(label)
    axs[i].set_ylabel("P(Final T > 400K)")
    axs[i].grid(True)
    if use_log:
        axs[i].set_xscale('log')
        axs[i].set_ylim(-0.2,1.2)
    axs[i].set_ylim(-0.2,1.2)

# hide unused subplots
for j in range(len(params), len(axs)):
    axs[j].axis("off")

plt.suptitle("Probability of High Final Surface Temperature vs. Monte Carlo Parameters", fontsize = 16, y=1.02)

pylab.tight_layout()



# sefine the main output directory  

sheet_switch = "y"

if sheet_switch == "y": 
    directory_prob = "ProbabilityFigure"
    filename_base = output_directory.replace("/", "_")
    filename = f"{filename_base}.csv"
    data_prob = np.column_stack((bin_centers / 1.4e21, probability_above_400K))
    df = pd.DataFrame(data_prob, columns=["Initial_Water_Mass", "Probability"])
    df.to_csv(os.path.join(directory_prob, filename), index=False)


# plotting and saving figures 
plot_switch = "y"
if plot_switch == "y":
    fig_size = (13, 8)
    figure_names = [
        "figure1.png", "figure2.png", "figure3.png", "figure4.png",
        "figure5.png", "figure6.png", "figure7.png", "figure8.png", "figure9.png"
    ]

    for i, figure_name in enumerate(figure_names):
        pylab.figure(i + 1, figsize=fig_size)
        save_path = os.path.join(output_directory, figure_name) 
        pylab.savefig(save_path, dpi=300, bbox_inches='tight')


#pylab.show()


