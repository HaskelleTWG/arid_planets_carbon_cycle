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
from planetary_carbon_forward_NominalPlot_ingas import forward_model
import pdb
import matplotlib.pyplot as plt
import pylab
from matplotlib.transforms import Bbox
import pandas as pd



import numpy as np
import matplotlib.pyplot as plt


run_switch = "y"
if run_switch == "y":
    # Initial water test cases
    #test_cases = [1.4e18, 1.4e19, 1.4e20, 1.4e21]
    #test_cases = [270, 330, 390, 450] #surface temperature
    test_cases = [550, 675, 800, 925] #serpentine stability



    # Run the forward model for each test case and store the results
    results = []
    for T_dehyd in test_cases:
        print(T_dehyd)
        total_time, total_y = forward_model(T_dehyd)
        results.append((total_time, total_y))



##################### Rerunning saved outputs (use if you don't need to recalculate anything)#####################

load_switch = "n"
if load_switch == "y":    
    # Load from saved CSV 
    folder_path = 'Figures/test/'  # Folder with saved CSV
    folder_name = os.path.basename(os.path.normpath(folder_path))
    csv_file = os.path.join(folder_path, f"{folder_name}.csv")

    # Load data
    df = pd.read_csv(csv_file)
    time = df['time'].values

    # detect the number of runs
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
# Test case labels for legend
#test_case_labels = ['0.1% Earth Ocean', '1% Earth Ocean','10% Earth Ocean', '100% Earth Ocean'] #initial surface water, some low inventories
#test_case_labels = ['270 K', '330 K', '390 K', '450 K'] # surface temp
test_case_labels = ['550 K', '700 K', '850 K', '1000 K'] #max mantle storage


# Generate gradient colors
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(test_case_labels)))
colors1 = plt.cm.Greens(np.linspace(0.3, 0.9, len(test_case_labels)))
colors2 = plt.cm.Purples(np.linspace(0.3, 0.9, len(test_case_labels)))

#################################
### Plotting figure #############
#################################

plt.figure(figsize=(11, 8))  # create multipanel figure

# Containers for legend handles and labels
ingassing_lines = []
ingassing_labels = []
outgassing_lines = []
outgassing_labels = []


for idx, (total_time, total_y) in enumerate(results):

    # CO2 weathering flux (index 14) in purple
    ing_line, = plt.semilogy(total_time, total_y[4], color=colors2[idx], linestyle='-',label = fr'$T_{{dehyd}}$ = {test_case_labels[idx]}, $T_{{surf}}$ = 525 K')
    ingassing_lines.append(ing_line)
    ingassing_labels.append(ing_line.get_label())
    
    # CO2 outgassing flux (index 15) in green 
    out_line, = plt.semilogy(total_time, total_y[5], color=colors1[idx], linestyle='-',  label =fr'$T_{{dehyd}}$ = {test_case_labels[idx]}, $T_{{surf}}$ = 290 K')
    outgassing_lines.append(out_line)
    outgassing_labels.append(out_line.get_label())


plt.ylabel(r"Maximum Crustal Hydration, $\Omega^{\mathrm{crust}}_{\mathrm{H_2O}}$ (kg)")
plt.xlabel('Time (Gyr)')
plt.xlim(0, 4.5e9)
plt.ylim(1e18, 2e20)
plt.xticks( ticks=[0, 1e9, 2e9, 3e9, 4e9, 4.5e9], labels=['0', '1', '2', '3', '4', '4.5'])

# Combined custom legend order
plt.legend(handles=ingassing_lines + outgassing_lines,
           labels=ingassing_labels + outgassing_labels,
           loc='lower right', fontsize='large')


save_switch = "y"
if save_switch == "y":
    # set save folder (adjust this for each run)
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
        "figure_serp_2temp.png"
    ]

    for i, figure_name in enumerate(figure_names):
        plt.figure(i + 1, figsize=fig_size)  # Assumes these figures already exist
        save_path = os.path.join(folder_path, figure_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    print(f"Figures saved to {folder_path}")

plt.show()
