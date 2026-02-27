# arid_planets_carbon_cycle
Version 1.0

This set of python scripts runs our geological carbon cycle model for arid terrestrial planets. The model is described in  H. White-Gianella and J. Krissansen-Totton (2026) "Carbon Cycle Imbalances on Arid Terrestrial Planets with Implications for Venus", The Planetary Science Journal. With the appropriate choice of parameters, this code can reproduce Figures 3-8 in the main text and Figures in the Supplemental Materials. 

If you are using this code, please cite White-Gianella and Krissansen-Totton et al. (2026).

REQUIREMENTS: Python, including numpy, pylab, and scipy modules.

HOW TO RUN THE NOMINAL CODE (Figures 3, S1, and S2) :
(1) Put all the python scripts in the same directory, and ensure python is working in this directory.
(2) Open planetary_carbon_forward_NominalPlot.py. Select desired parameter ranges, e.g., planetary mass, instellation, initial carbon inventory, and all parameters listed in Table 2.
(3) Run Nominal_Plot.py. Code will output model results of four different initial water inventories (0.1, 1, 10, and 100% Earth Oceans). Users can modify the initial water inventory, the number of test cases, and the overall figure design. 

Repeat these steps with planetary_carbon_forward_NominalPlot_ingas.py and Nominal_Plot_ingas.py to recreate Figure S3 (maximum crustal water inventory calculations).

HOW TO RUN THE MONTE CARLO CODE (Figures 5-8, S4-10) :
(1) Put all the python scripts in the same directory, and ensure python is working in this directory.
(2) Open planetary_carbon_forward_MC.py. Select desired parameter ranges, and choose between Earth-like or Venus-like planetary parameters. You also must set Earth or Venus planetary hypsometry in Hypso_fun.py.
(3) Run planetary_carbon_forward_batch.py. Select the desired MC parameters (refer to Table 2), and the number of runs. The code is set to run ten times, each with 1000 runs to total 10000 runs. 
(4) Run planetary_carbon_forward_saved.py. This will stitch together all the batch outputs into plots with 10000 MC runs. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EXPLANATION OF CODE STRUCTURE:

%% planetary_carbon_forward_batch.py
This script provides the shell to repeatedly call the forward model and plot the output distributions. Several initial parameters can be modified for Monte Carlo sampling

- it_num: number of forward model calls used to build output distributions. The larger this is, the longer the code will take to run.
- Parallelize: determines whether multiple threads should be used (faster) or whether all iterations should be computer using the same thread (slower).
- batch_index: saves outputs into separate .npy files. 

Once parameter ranges have been defined, the script calls the forward model ('forward_model') once and uses the dimensions of the outputs to define an output array to store all future outputs ('all_output'). The forward model is then called many times (equal to 'it_num') and parameter ranges are randomly sampled for each forward model call. Forward model calls resulting in errors or non-physical outputs are discarded (the code may still print error messages from these discarded outputs). 

%% planetary_carbon_forward_MC.py
This script contains the following functions which, taken together, define and solve the forward model:

% forward_model - Given parameter inputs, forward_model calculates the initial conditions for the carbon cycle 

% system_of_equations - Contains the ODEs that describe the time evolution of the carbon cycle (equations 1-4 in manuscript). The model couples outgassing/weathering, stellar evolution, planetary hypsometry, and the deep-water cycle to calculate the transfer of H2O and C between the interior and the surface. 

% land_fraction - Calculates continental land fraction using a hypsometric curve for Earth or Venus. Calls Hypso_fun.py to calculate how much of the planet would be submerged by that volume of water.

%% carbon_cycle_model - Contains all tested weathering paramterizations described in Section 2.2 of the manuscript. Users can choose which paramterization to use in the forward_model. 'weathering_flux' is the WHAK model, 'MAC_weathering_flux' is the MAC model, and 'MAC_LH_wx_flux' is our MAC formulation that includes wind-driven evaporation (Section 2.2.3). 'MAC_LH_wx_flux' should be used with planetary_carbon_forward_NominalPlot.py, and 'MAC_LH_wx_flux_MC" should be used with planetary_carbon_forward_MC.py.  

%% escape_functions.py
Calculates atmospheric escape of water assuming diffusion-limited escape of hydrogen through a background atmosphere composed of CO2 and N2. 

%% Latent_Heat_Flux.py
Determines the saturation vapor pressure for water using the Clausius-Claperyron relation. 'latent_heat_flux' uses the SVP and other atmospheric parameters (wind speed, drag coeffficient, etc.) to calculate latent heat as a function of temperature. 

%% outgassing_module_fast.py
Contains the outgassing code described in Wogan et al., (2020). Requires VolcGases installation. 

%% radiative_functions.py
This contains the the parameterized climate models used in this study. 'my_water_frac' calculates the atmosphere-ocean partitioning of water and uses the surface temperature (Tsurf), amount of H2O (PH2O) and amount of CO2 (PCO2) determinded by the 'forward_model.'

%% stellar_funs.py
Takes inputs for stellar type and returns total luminosity, FUV, and XUV as a function of time. 

%% thermodynamic_variables.py
Contains the function Sol_prod that calculates carbonate solubility product as a function of temperature from Pilson, M. E. (1998) "An Introduction to the Chemistry of the Sea", Prentice-Hall, Inc. This function reproduces table G.1 in Pilson for salinity=35. thermodynamic_variables.py also contains the temperature-dependent functions for carbon equilibrium constants, equil_cont.

%% ProbabilityFigure.py
Produces all probability plots in the Manuscript and Supplemental Materials. The probability of Final Temp > 400 K calculated from the forward_model are stored in .csv files, which are plotted together in this script. 

%% WHAKvsMAC_Plot.py
This script uses saved .csv files from Nominal_Plot.py to create Figure 4 in the manuscript. 


%% Other files included in the code:
- VolcGases-master is the outgassing moduls required to run outgassing_module_fast.py
- options_array.npy is an array created by the code for storing user choices about parallelization, iteration number, and carbon cycle parameters. 

END EXPLANATION OF CODE STRUCTURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-------------------------
Contact e-mail: hasktw@uw.edu

