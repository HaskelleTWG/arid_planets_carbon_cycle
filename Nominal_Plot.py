# This script contains the forward model parallelizes the forward model
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

global init_water
from planetary_carbon_forward_NominalPlot import forward_model
import pdb
import matplotlib.pyplot as plt
import pylab
from matplotlib.transforms import Bbox
import pandas as pd



import numpy as np
import matplotlib.pyplot as plt

run_switch = "n"
if run_switch == "y":
    # Initial water test cases
    test_cases = [1.4e18, 1.4e19, 1.4e20, 1.4e21] #kg

    # Run the forward model for each test case and store the results
    results = []
    for init_water in test_cases:
        print(init_water)
        total_time, total_y = forward_model(init_water)
        results.append((total_time, total_y))


##################### Rerunning saved outputs (use if you don't need to recalculate anything)#####################

load_switch = "y"
if load_switch == "y":    
    # Load from saved CSV 
    folder_path = 'Figures/folder_1/folder_2/'  # Folder with saved CSV
    folder_name = os.path.basename(os.path.normpath(folder_path))
    csv_file = os.path.join(folder_path, f"{folder_name}.csv")

    # Load data 
    df = pd.read_csv(csv_file)
    time = df['time'].values

    # calculate the number of runs 
    n_cases = len([col for col in df.columns if col.startswith("var0_")])
    n_vars = len([col for col in df.columns if col.startswith("var")]) // n_cases

    # reconstruct results 
    results = []
    for i in range(n_cases):
        y = []
        for j in range(n_vars):
            colname = f"var{j}_case{i+1}"
            y.append(df[colname].values)
        results.append((time, y))  # Append as (total_time, total_y)


    

# Plotting
# test case labels for legend
test_case_labels = ['0.1% Earth Ocean', '1% Earth Ocean','10% Earth Ocean', '100% Earth Ocean'] #initial surface water, some low inventories


# Generate gradient colors
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(test_case_labels)))
colors1 = plt.cm.Greens(np.linspace(0.3, 0.9, len(test_case_labels)))
colors2 = plt.cm.Purples(np.linspace(0.3, 0.9, len(test_case_labels)))

#################################
### Plotting figure #############
#################################

plt.figure(figsize=(13, 20))  # create multipanel figure

#######################Figure 1####################
# Plot Surface Water
plt.subplot(4, 2, 1)
for idx, (total_time, total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[1], label=test_case_labels[idx], color=colors[idx])
plt.ylabel(r"Surface water, $R^{\mathrm{surf}}_{\mathrm{H_2O}}$ (kg)")
plt.xlim(0, 4.5e9)
plt.legend(loc="lower left")
#plt.xlabel('Time (years)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

# Plot Surface Temperature for each test case
plt.subplot(4, 2, 2)
for idx, (total_time, total_y) in enumerate(results):
    plt.plot(total_time, total_y[8], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Surface Temperature (K)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


# Plot Precipitation 
plt.subplot(4, 2, 3)
for idx, (total_time, total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[3], label=test_case_labels[idx], color=colors[idx])
    plt.semilogy(total_time, total_y[4], label=test_case_labels[idx], color=colors1[idx])
plt.ylabel(r"Precipitation, $p$ (m/yr)")
plt.xlim(0, 4.5e9)
plt.ylim(1e-5, 1e1)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


# Plot Atmosphere + Ocean carbon reservoir
plt.subplot(4, 2, 4)
for idx, (total_time, total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[12], label=test_case_labels[idx], color=colors[idx])
plt.ylabel(r"Surface carbon, $R^{\mathrm{surf}}_{\mathrm{CO_2}}$ (kg)")
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


#### Figure 1.5 - summary of outgassing and ingassing 

plt.subplot(4, 2, 5)

# Containers for legend handles and labels
ingassing_lines = []
ingassing_labels = []
outgassing_lines = []
outgassing_labels = []

conversion_factor = 1395.24  # Tmol/yr → kg/s for CO2

for idx, (total_time, total_y) in enumerate(results):

    # CO2 weathering flux (index 14) in purple — ingassing
    ing_line, = plt.plot(total_time, total_y[14]*conversion_factor, color=colors2[idx], linestyle='-',label = fr'{test_case_labels[idx]} - $F^{{\mathrm{{ingas}}}}_{{\mathrm{{CO_2}}}}$')
    ingassing_lines.append(ing_line)
    ingassing_labels.append(ing_line.get_label())
    
    # CO2 outgassing flux (index 15) in green — outgassing
    out_line, = plt.plot(total_time, total_y[15]*conversion_factor, color=colors1[idx], linestyle='--',  label = fr'{test_case_labels[idx]} - $F^{{\mathrm{{volc}}}}_{{\mathrm{{CO_2}}}}$')
    outgassing_lines.append(out_line)
    outgassing_labels.append(out_line.get_label())


plt.ylabel('CO$_2$ Flux (kg/s)')
plt.xlim(0, 4.5e9)
plt.ylim(-500,7500)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

# Combined legend order
plt.legend(handles=ingassing_lines + outgassing_lines,
           labels=ingassing_labels + outgassing_labels,
           loc='upper center', fontsize='small')



plt.subplot(4, 2, 6)
# H2O ingassing and outgassing
# Containers for legend handles
outgassing_lines = []
outgassing_labels = []
ingassing_lines = []
ingassing_labels = []

for idx, (total_time, total_y) in enumerate(results):
    # H2O degassing (index 28) 
    out_line, = plt.semilogy(total_time, total_y[28], color=colors1[idx], linestyle='--', label = fr'{test_case_labels[idx]} - $F^{{\mathrm{{volc}}}}_{{\mathrm{{H_2O}}}}$')
    outgassing_lines.append(out_line)
    outgassing_labels.append(out_line.get_label())

    # H2O mantle return (index 27) 
    in_line, = plt.semilogy(total_time, total_y[27], color=colors2[idx], linestyle='-', label = fr'{test_case_labels[idx]} - $F^{{\mathrm{{ingas}}}}_{{\mathrm{{H_2O}}}}$')
    ingassing_lines.append(in_line)
    ingassing_labels.append(in_line.get_label())

plt.ylabel('H$_2$O Flux (kg/s)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


#Combine legend entries: first all ingassing, then all outgassing
plt.legend(handles=ingassing_lines + outgassing_lines,labels=ingassing_labels + outgassing_labels,loc='lower center', fontsize='small')


## Net C flux
plt.subplot(4, 2, 7)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, (total_y[14] - total_y[15])*conversion_factor, label=test_case_labels[idx], color=colors[idx])
plt.ylabel(r"Net CO$_2$ Flux, $F^{{\mathrm{{ingas}}}}_{{\mathrm{{CO_2}}}} - F^{{\mathrm{{volc}}}}_{{\mathrm{{CO_2}}}}$  (kg/s)")
plt.xlim(0, 4.5e9)
plt.ylim(-3000,1500)
plt.xlabel('Time (Gyr)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


# Net H2O flux
plt.subplot(4, 2, 8)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, (total_y[27]-total_y[28]), label=test_case_labels[idx], color=colors[idx])
plt.ylabel(r"Net H$_2$O Flux, $F^{{\mathrm{{ingas}}}}_{{\mathrm{{H_2O}}}} - F^{{\mathrm{{volc}}}}_{{\mathrm{{H_2O}}}}$  (kg/s)")
plt.xlim(0, 4.5e9)
plt.xlabel('Time (Gyr)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.yscale('symlog')


######################## water plots (Figure 2)########################
plt.figure(figsize=(13, 12))

plt.subplot(3, 2, 1)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[0], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Interior H$_2$O (kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.legend()

plt.subplot(3, 2, 2)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[1], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Atmo+ocean H$_2$O (kg)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlim(0, 4.5e9)

plt.subplot(3, 2, 3)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, (total_y[1]+total_y[0]), label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Total H$_2$O (kg)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlim(0, 4.5e9)

plt.subplot(3, 2, 4)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time,total_y[18], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Atmospheric H$_2$O(kg)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlim(0, 4.5e9)

plt.subplot(3, 2, 5)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time,total_y[20] , label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Ocean H$_2$O(kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')

plt.subplot(3, 2, 6)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time,total_y[20]+total_y[18]+total_y[0] , label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Test Total H$_2$O(kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')



####################### CO2 flux plots (Figure 3)##############################
plt.figure(figsize=(13, 12))
plt.subplot(3, 2, 1)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[14], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('CO$_2$ Weathering Flux (Tmol CO$_2$/yr)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.legend()

plt.subplot(3, 2, 2)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[15], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('CO$_2$ Outgassing Flux (Tmol CO$_2$/yr)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

plt.subplot(3, 2, 3)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[15] - total_y[14], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Net CO$_2$ Flux (Tmol CO$_2$/yr)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

#fco2
plt.subplot(3, 2, 4)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[23] / 1e5, label=test_case_labels[idx], color=colors[idx])
plt.ylabel('fCO$_2$ Pressure (bar)')
plt.xlim(0, 4.5e9)
plt.xlabel('Time (Gyr)')
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])



# Mantle CO2 
plt.subplot(3, 2, 5)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[13], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Interior CO$_2$ (kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')

#fH2O
plt.subplot(3, 2, 6)
for idx, (total_time, total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[21], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('fH$_2$O Pressure (bar)')
plt.xlim(0, 4.5e9)
plt.xlabel('Time (yrs)')



####################### More CO2 flux plots (Figure 4)########################
plt.figure(figsize=(13, 12))
plt.subplot(3, 2, 1)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[11], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Subaerial CO$_2$(Tmol CO$_2$/yr)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.legend()

plt.subplot(3, 2, 2)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[16], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Submarine CO$_2$ (Tmol CO$_2$/yr)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

plt.subplot(3, 2, 3)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[26], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Total Outgassing Flux (kg/s)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


plt.subplot(3, 2, 4)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[17], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Oceanic CO$_2$ (kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

plt.subplot(3, 2, 5)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[19], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Atmospheric CO$_2$ (kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')


plt.subplot(3, 2, 6)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[17]+ total_y[19], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Atmo+Ocean CO$_2$ Check(kg)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')


######################### Mantle H2O Fluxes (Figure 5)##########################

plt.figure(figsize=(13, 12))
plt.subplot(3, 2, 1)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[28], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('H$_2$O Degassing (kg/s)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.legend()

plt.subplot(3, 2, 2)
for idx, (total_time,total_y) in enumerate(results):
    plt.semilogy(total_time, total_y[27], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('H$_2$O Interior Return (kg/s)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

plt.subplot(3, 2, 3)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, (total_y[27]-total_y[28]), label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Net H$_2$O Flux (kg/s)')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])


## Ocean and Land Fraction

plt.subplot(3, 2, 4)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[5], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Ocean Depth')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

plt.subplot(3, 2, 5)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[25], color=colors[idx])
plt.plot(total_time,total_time*0 + (2.5/11.4)**0.25,'k-',label='Modern Earth')
plt.ylabel('Ocean Fraction')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.legend()
plt.xlabel('Time (yrs)')

plt.subplot(3, 2, 6)
for idx, (total_time,total_y) in enumerate(results):
    plt.plot(total_time, total_y[6], label=test_case_labels[idx], color=colors[idx])
plt.ylabel('Land Fraction')
plt.xlim(0, 4.5e9)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])
plt.xlabel('Time (yrs)')

plt.tight_layout()



sheet_switch = "n"
if sheet_switch == "y":
    # create a dictionary to store the data
    data_dict = {'time': results[0][0]}  # Use time from the first run
    for idx, (_, total_y) in enumerate(results):
        data_dict[f'temp_case_{idx+1}'] = total_y[8]  # Store temperature for each case

    # convert to DataFrame
    data = pd.DataFrame(data_dict)

    # ensure folder exists
    folder_path = 'Figures/Earth/Escape_off/'  
    os.makedirs(folder_path, exist_ok=True)

    # save the file to the specified folder
    file_name = 'Area_MAC_v4.csv'
    data.to_csv(os.path.join(folder_path, file_name), index=False)

    print(f"Data saved to {os.path.join(folder_path, file_name)}")


save_switch = "y"
if save_switch == "y":
    # set save folder (change this for each run)
    folder_path = 'Figures/test/'
    os.makedirs(folder_path, exist_ok=True)

    # get last folder name for file naming
    folder_name = os.path.basename(os.path.normpath(folder_path))
    file_name = f'{folder_name}.csv'

    # save all output variables to CSV
    time = results[0][0]  # Assume same time array across runs
    data_dict = {'time': time}

    for var_idx in range(len(results[0][1])):  # Loop over output variables
        for case_idx, (_, total_y) in enumerate(results):
            key = f'var{var_idx}_case{case_idx+1}'  # e.g., var0_case1
            data_dict[key] = total_y[var_idx]

    data = pd.DataFrame(data_dict)
    data.to_csv(os.path.join(folder_path, file_name), index=False)
    print(f"Data saved to {os.path.join(folder_path, file_name)}")

    # save plots
    fig_size = (13, 8)
    figure_names = [
        "figure1.png", "figure2.png", "figure3.png", "figure4.png",
        "figure5.png"
    ]

    for i, figure_name in enumerate(figure_names):
        plt.figure(i + 1, figsize=fig_size)  # Assumes these figures already exist
        save_path = os.path.join(folder_path, figure_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Figures saved to {folder_path}")


plt.show()



