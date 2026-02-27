# This script contains the forward model

#####################
import numpy as np
import pylab
import matplotlib.pyplot as plt
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
import time
from numba import jit # standard module for speeding up python, will need to install
from numba_nelder_mead import nelder_mead #fast optimizer, hopefully just works once numba is installed
import os
from Hypso_fun import hypso_fun ### hypsometric function for calculating land fraction from surface water volume
from Latent_Heat_Flux import latent_heat_flux
import pdb




#####################
ppn2 = 1e5 #fixed background nitrogen - changing this won't do anything because N2 is fixed in climate grid calculations



def forward_model(init_water, Init_solid_H2O, Max_mantle_H2O, Init_fluid_CO2, Init_solid_CO2, mm, n_out, Te, ts):

    '''
    ## Earth   
    RE = 1.0 #Planet_inputs.RE #Planet radius relative Earth 
    ME = 1.0 #Planet_inputs.ME #Planet mass relative Earth 
    Planet_sep = 1 #Planet_inputs.Planet_sep #planet-star separation (AU)
    albedoC = 0.3 #Planet_inputs.albedoC    #cold state albedo,
    albedoH = 0.2#Planet_inputs.albedoH    #hot state albedo, not used currently

    '''
    ## Venus
    RE = 0.95  #Planet_inputs.RE #Planet radius relative Earth - Venus
    ME = 0.815 #Planet_inputs.ME #Planet mass relative Earth
    Planet_sep = 0.72 #Planet_inputs.Planet_sep #planet-star separation (AU) - Venus
    albedoC = 0.65#Planet_inputs.albedoC    #cold state albedo. High = 0.65, Higher = 0.7 
    albedoH = 0.2#Planet_inputs.albedoH    #hot state albedo, not used currently
    
    
    pm = 4000.0#Planet_inputs.pm #Average mantle density
    rc = 3600000#Planet_inputs.rc #Metallic core radius (m)
    Total_Fe_mol_fraction = 0.06#Planet_inputs.Total_Fe_mol_fraction # iron mol fraction in mantle
   
    #Stellar parameters
    Omega_sun_ar = 1.8
    tsat_XUV = (2.9*Omega_sun_ar**1.14)/1000 #XUV saturation time
    Stellar_Mass = 1.0#Stellar_inputs.Stellar_Mass #stellar mass (relative sun)
    fsat = 10**(-3.13)#Stellar_inputs.fsat 
    beta0 = 1.0/(0.35*np.log10(Omega_sun_ar) - 0.98)#Stellar_inputs.beta0
    epsilon = 0.1#Stellar_inputs.epsilon
    
    #generate random seed for this forward model call
    np.random.seed(int(time.time()))
    seed_save = np.random.randint(1,1e9)

    ## Initial volatlie and redox conditions:
    #Init_solid_H2O = 0.5e20#Init_conditions.Init_solid_H2O - initial water in the mantle (kg)
    Init_fluid_H2O = init_water##Init_conditions.Init_fluid_H2O - initial water at the surface (kg)
    #Init_fluid_H2O = 1.4e19##Earth initial water at the surface (kg)
    Init_solid_O= 0.0#Init_conditions.Init_solid_O, not used here
    Init_fluid_O = 0.0#Init_conditions.Init_fluid_O, not used here
    Init_solid_FeO1_5 = 0.0#Init_conditions.Init_solid_FeO1_5, not used here
    Init_solid_FeO = 0.0#Init_conditions.Init_solid_FeO, not used here
    #Init_fluid_CO2 =1e19# Init_conditions.Init_fluid_CO2 - initial CO2 at the surface (kg)
    #Init_solid_CO2= 1e21#Init_conditions.Init_solid_CO2  - initial CO2 in the mantle (kg)

    #Oxidation parameters - see PACMAN paper
    wet_oxid_eff = 1e-1#MC_inputs.interiord
    MFrac_hydrated = 1e-2#3e-2#MC_inputs.interior, 1%
    dry_oxid_frac = 1e-2#MC_inputs.interiorc 


    #ocean chemistry and weathering parameters - again see PACMAN paper, many of these not important or not used
    ocean_Ca = 1e-2 #MC_inputs.ocean_a 
    omega_ocean = 5.0 #MC_inputs.ocean_b 
    efold_weath = 15.0 #MC_inputs.ccycle_a
    alpha_exp = 0.2 #MC_inputs.ccycle_b
    supp_lim = 1e6 #MC_inputs.supp_lim    

    #Escape parameters - see PACMAN paper
    mult = 0.0 #MC_inputs.esc_c ## for when transition from diffusion to XUV
    mix_epsilon = 0.5 #MC_inputs.esc_d # fraction energy goes into escape above O-drag
    Te_input_escape_mod = 0.0#-15#MC_inputs.Tstrat
    Thermosphere_temp = 1000.0

    #some constants and conversions
    MEarth = 5.972e24 #Mass of Earth (kg)
    kCO2 = 2e-3 #Crystal-melt partition coefficent for CO2
    #kCO2 = 0.99 #sensitivity test reduced mantle (CO2 retained in interior)
    G = 6.67e-11 #gravitational constant
    rp = RE * 6.371e6 #Planet radius (m)
    Mp = ME * MEarth #Planet mass (kg)
    g = G*Mp/(rp**2) # gravity (m/s2)
    Tsolidus = sol_liq(rp,g,pm,rp,0.0,0.0) #Solidus for magma ocean evolution - only used in volcanic calculations
    Tliquid = Tsolidus + 600 #Liquidus for magma ocean evolution - not used

    F_Earth = 0.28*(10**12)/(365*24*60*60) #Modern flux into the interior, kg/s
    density_water = 1000  # kg/m^3


    min_Te = 150.0 ## Minimum Te for purposes of OLR/ASR calculations and escape calculations
    min_ASR = 5.67e-8 * (min_Te/(0.5**0.25))**4.0 ## Minimum Absorbed Shortwave Radiation (ASR)
    min_Te = 207.14285714 # Threshold to prevent skin temperature from getting too low where OLR grid contains errors. Note this lower limit does not apply to stratosphere temperatures used for escape calculations.

    #Max_mantle_H2O = 0.5*1.4e21 * 1.0 * (rp**3 - rc**3) / ((6.371e6)**3 - (3.4e6)**3) ## Max mantle water content (kg) - this is the most water than can be held in mantle minerals, important for deep water cycle parameterization 
    #Max_mantle_H2O = 1e22
    
    Start_time = 0.5e9 #Switch_Inputs.Start_time #Model start time (relative to stellar evolution track), in yrs
    Max_time=5e9 #finish time, in yrs
    #Max_time = 1.5425716462258995e+17 /(60*60*24*365)
    test_time = np.linspace(Start_time*365*24*60*60,Max_time*365*24*60*60,10000)
    new_t = np.linspace(Start_time/1e9,Max_time/1e9,100000)

    [Relative_total_Lum,Relative_XUV_lum,Absolute_total_Lum,Absolute_XUV_Lum] = main_sun_fun(new_t,Stellar_Mass,tsat_XUV,beta0,fsat) #Calculate stellar evolution
    Absolute_XUV_Lum = 1.1*Absolute_XUV_Lum
    Absolute_total_Lum = 1.1* Absolute_total_Lum
    ASR_new = (Absolute_total_Lum/(16*3.14159*(Planet_sep*1.496e11)**2) ) #ASR flux through time (not accounting for bond albedo)
    
    ## All of the following is for bounding the ASR and corresponding upper atmosphere temperature for escape calcaultions - shouldn't need to change
    for ij in range(0,len(ASR_new)): # do not permit ASR outside of interpolation grid
        if (ASR_new[ij] < min_ASR):
            ASR_new[ij] = min_ASR
    Te_ar = (ASR_new/5.67e-8)**0.25
    Tskin_ar = Te_ar*(0.5**0.25) ## Skin temperature through time
    for ij in range(0,len(Tskin_ar)): #Don't permit skin temperature to exceed range min_Te - 350 due to errors in grid (does not apply to stratospheric temperature used to calculae escape fluxes)
        if Tskin_ar[ij] > 350:
            Tskin_ar[ij] = 350.0
        if Tskin_ar[ij] < min_Te:
            Tskin_ar[ij] = min_Te
    Te_fun = interp1d(new_t*1e9*365*24*60*60,Tskin_ar) #Skin temperature function, used in OLR calculations
    ASR_new_fun = interp1d(new_t*1e9*365*24*60*60, ASR_new) #ASR function, used to calculate shortwave radiation fluxes through time
    AbsXUV = interp1d(new_t*1e9*365*24*60*60 , Absolute_XUV_Lum/(4*np.pi*(Planet_sep*1.496e11)**2)) #XUV function, used to calculate XUV-driven escape

    
    @jit(nopython=True) # function for finding surface temperature that balances ASR and interior heatflow
    def funTs_general2(Ts,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 
        Ts_in= Ts[0]
        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,ppn2,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return -(heat_atm)**2  
    
    '''
    def funTs_general(Ts,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,MMW,PO2_surf): 

        [OLR_new,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(Ts_in,Te_input,H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,ppn2,MMW,PO2_surf) 
        heat_atm = OLR_new - ASR_input 
        return (qm - heat_atm)**2          
    '''

    def FeO_mass_frac(Total_Fe): # Convert total iron mole fraction to mass fraction
        XAl2O3 = 0.022423
        XCaO = 0.0335
        XNa2O = 0.0024 
        XK2O = 0.0001077 
        XMgO = 0.478144 
        XSiO2 =  0.4034    
        m_sil = XMgO * (16.+25.) + XSiO2 * (28.+32.) + XAl2O3 * (27.*2.+3.*16.) + XCaO * (40.+16.) +  (Total_Fe) * (56.0+16.0)
        return (Total_Fe) * (56.0+16.0)/m_sil 


    def system_of_equations(t0,y):

        print(t0)
        ocean_CO3 = float(omega_ocean * Sol_prod(y[8]) / ocean_Ca)
        Mantle_mass = (4./3. * np.pi * pm * (rp**3 - rc**3))
        
        #print (t0/(365*24*60*60))

        #################################################################################

    
        Psur = float(y[22]+y[23]+y[31]+ppn2)

        [FH2O,H2O_Pressure_surface] = H2O_partition_function2( y[1],0.0,0.0,rp,g,0.01,y[24])            
        [FCO2,CO2_Pressure_surface] = CO2_partition_function( y[12],0.0,0.0,rp,g,kCO2,y[24]) 


        AB = AB_fun(float(y[8]),H2O_Pressure_surface,float(y[1]+y[12]),albedoC,albedoH) ## Can calculation albedo to transition between cold and hot states
        AB = albedoC # but for illustrative purposes (and numerical efficiency) might be better to just fix albedo - we don't care about albedo in runaway greenhouse anyway

        ASR_input = float((1-AB)*ASR_new_fun(t0))
        if (ASR_input < min_ASR):
            ASR_input = min_ASR

        Te_ar = (ASR_input/5.67e-8)**0.25
        Te_input = Te_ar*(0.5**0.25)
        if Te_input > 350:
            Te_input = 350.0
        if Te_input < min_Te:
            Te_input = min_Te 

        
        initialize_fast = np.array(300.0)
        ace1 =  nelder_mead(funTs_general2, x0=initialize_fast, bounds=np.array([[10.0], [4000.0]]).T, args = (Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])), tol_f=0.0001,tol_x=0.0001, max_iter=1000)
        SurfT = ace1.x[0]
        '''
        ace1= optimize.minimize(funTs_general,x0=initialize_fast,args = (float(y[7]),ll,visc,beta,Te_input,ASR_input,H2O_Pressure_surface,CO2_Pressure_surface,ocean_CO3,float(y[24]),float(y[22])),method='COBYLA',bounds = ((100,4500),),options={'maxiter':1000})
                #SurfT = float(ace1.x)  # COBYA 
        '''


        y[8] = SurfT


        [OLR,newpCO2,ocean_pH,ALK,Mass_oceans_crude] = correction(float(SurfT),float(Te_input),H2O_Pressure_surface,CO2_Pressure_surface,rp,g,ocean_CO3,ppn2,float(y[24]),float(y[22]))
        CO2_Pressure_surface = newpCO2 
            
        ASR = ASR_input


       
        ####################################################################################  
        ## end magma ocean/solid mantle portion. The rest of the code applies to both phases
        ####################################################################################  
        #print ('FH2O,H2O_Pressure_surface',FH2O,H2O_Pressure_surface)
        y[9] = OLR
        y[10] = ASR



        if y[1]< 0:
            H2O_Pressure_surface = 0.0
            FH2O = 0.0

        ## updated strat temp adjusting for insolation
        Te_ar = (ASR_input/5.67e-8)**0.25
        Te_input_escape = Te_ar*(0.5**0.25) + Te_input_escape_mod
        if Te_input_escape > 350:
            Te_input_escape = 350.0
        if Te_input_escape < 150.0:
            Te_input_escape = 150.0

        water_frac = my_water_frac(float(y[8]),Te_input_escape,H2O_Pressure_surface,CO2_Pressure_surface)
        atmo_H2O = np.max([H2O_Pressure_surface*water_frac,0.0])
        y[18] = water_frac*H2O_Pressure_surface *4 *(0.018/float(y[24]))* np.pi * (rp**2/g) #Atmospheric H2O
        y[20] = (1 - water_frac)*H2O_Pressure_surface *4 *(0.018/float(y[24]))* np.pi * (rp**2/g) #Ocean H2O
        y[21] = water_frac*H2O_Pressure_surface/1e5
        
     
        fO2  =  y[22]
        fO2_pos = np.max([0,fO2])
        Pressure_surface = fO2_pos +atmo_H2O + CO2_Pressure_surface + ppn2 ## atmospheric pressure only (not including weight of the liquid water ocean)
        #Pressure_surface = 101325 #Pa

        
        #print ('float(SurfT),Te_input_escape,H2O_Pressure_surface,CO2_Pressure_surface')
        frac_h2o_upper = my_fH2O(float(SurfT),Te_input_escape,H2O_Pressure_surface,CO2_Pressure_surface)
        #print ('frac_h2o_upper',frac_h2o_upper)
        frac_h2o_upper = np.min([atmo_H2O / Pressure_surface, frac_h2o_upper])
        #print ('atmo_H2O,Pressure_surface,frac_h2o_upper')
        if (H2O_Pressure_surface<1e-5)and(frac_h2o_upper<1e-9): # lower threshold zero out NEW TRAPPIST THING
            frac_h2o_upper = 0.0
            atmo_H2O = 0.0
            H2O_Pressure_surface = 0.0

        #print ('Te_input_escape',Te_input_escape,'water_frac',water_frac,'frac_h2o_upper',frac_h2o_upper)
        
        #######################
        ## Atmospheric escape calculations
        ## diffusion limited escape:
        fCO2_p = (1- frac_h2o_upper)*CO2_Pressure_surface / (CO2_Pressure_surface+fO2_pos+ppn2)
        fO2_p = (1- frac_h2o_upper)*fO2_pos / (CO2_Pressure_surface+fO2_pos+ppn2)
        fN2_p = (1- frac_h2o_upper)*ppn2 / (CO2_Pressure_surface+fO2_pos+ppn2)
        mol_diff_H2O_flux = better_diffusion(frac_h2o_upper,Te_input_escape,g,fCO2_p,fO2_p,fN2_p) #mol H2O/m2/s
        
        #XUV-driven escape
        XH_upper = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p) ## assumes CO2 and N2 don't dissociate
        XH = 2*frac_h2o_upper / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XO = (2*fO2_p+frac_h2o_upper) / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        XC = fCO2_p / (3*frac_h2o_upper + 2*fO2_p + fCO2_p + fN2_p)
        #true_epsilon = find_epsilon(Te_input_escape,RE,ME,float(AbsXUV(t0)), XO, XH, XC,epsilon,mix_epsilon)
        true_epsilon = find_epsilon(Thermosphere_temp,RE,ME,float(AbsXUV(t0)), XO, XH, XC,epsilon,mix_epsilon)

        if (XC < 0)or(y[12]<1e10):
            XC = 0.0
        #[mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Te_input_escape,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XC) #kg/m2/s
        [mH_Odert3,mO_Odert3,mOdert_build3,mC_Odert3] = Odert_three(Thermosphere_temp,RE,ME,true_epsilon,float(AbsXUV(t0)), XO, XH, XC) #kg/m2/s
        numH = ( mH_Odert3 / 0.001 ) # mol H / m2/ s
        numO = ( mO_Odert3 / 0.016 ) # mol O / m2/ s
        numC = ( mC_Odert3 / 0.044 ) # mol CO2 / m2/ s
        
        if 2*mol_diff_H2O_flux> numH: ## if diffusion limit exceeds XUV-driven, shift diffusion-limit downward
             mol_diff_H2O_flux = 0.5*np.copy(numH)

        ## Combined escape flux, weighted by H abundance:
        w1 = 1.0#mult*(2./3 -  XH_upper)**4
        w2 = 0.0#XH_upper**4
        Mixer_H = (w1*2*mol_diff_H2O_flux + w2 * numH ) / (w1+w2) # mol H / m2 /s
        Mixer_O = (w1*0.0 + w2 * numO ) / (w1+w2)
        Mixer_C = (w1*0.0 + w2 * numC ) / (w1+w2)
        Mixer_Build = 0.5* Mixer_H - Mixer_O  ## CO2 drag doesn't affect redox (CO2 drag negligible anyway)

        escape = 4*np.pi*rp**2 * Mixer_H*0.018/2 ## kg H2O /s
        net_escape = 4*np.pi*rp**2 * Mixer_Build*0.016 ## kg O2/s
        CO2_loss =  4*np.pi*rp**2 * Mixer_C*0.044 ## kg CO2/s


        ## add some nonthermal escape


        # done with escape calculations
        #######################
        #print ('escape',escape,'net_escape',net_escape,'CO2_loss',CO2_loss)
        ## Find ocean depth and land fraction:
        Ocean_depth = (0.018/y[24]) * (1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
        if Ocean_depth > Max_depth:
            Linear_Ocean_fraction = 1.0
            Ocean_fraction = 1.0
        else:
            Linear_Ocean_fraction = (Ocean_depth/Max_depth) ## Crude approximation to Earth hypsometric curve
            Ocean_fraction = (Ocean_depth/Max_depth)**1.0#0.25
            water_volume = (0.018/y[24])*(1-water_frac) * H2O_Pressure_surface *4 *np.pi * 6371000**2/(g*1000) #m3 ?
            mass_surface_water = water_volume * density_water #kg
            Land_fraction = hypso_fun(water_volume)
            Ocean_fraction = 1 - Land_fraction
            
    
        ## Melt and crustal oxidation variables:
        
        F_CO2 = 0.0
        F_H2O = 0.0
        O2_consumption = 0.0
        OG_O2_consumption =0.0
        Plate_velocity = 0.05/(365*24*60*60)
        crustal_depth = 5000.0
        #Melt_volume = 0.0
        Poverburd = fO2_pos +H2O_Pressure_surface + CO2_Pressure_surface + ppn2 ##total pressure of volatlies at the surface
        
        
        #mm=1.5 # Exponent determing relationhip between crustal production and outgassing (see equation S9)
        #n_out=0.4#np.random.uniform(0.1,0.73) # Exponent determing relationship between internal heat flow and outgassing (see equation S8)
        # Calculate internal heatflow, outgassing, and spreading rate using equations S8-10:
        t_Gyr = (4.5e9-(t0/(365*24*60*60)))
        Q=(1-t_Gyr/(4.6e9))**-n_out
        #Q =1

        Melt_volume = 20*1000.0**3/(365*24*60*60) ## Modern Earth melt production, 20 km3/yr, converted to m3/s
        Melt_volume_past = 40*1000.0**3/(365*24*60*60) #40 km3/yr
        Total_Fe_mass_fraction = FeO_mass_frac(Total_Fe_mol_fraction) 
        ## Fresh crust production
        dmelt_dt = (pm * Melt_volume * Q**mm) #kg/s melt production over whole planet - not constant, includes Q
        #dmelt_dt = (pm * Melt_volume_past)#constant outgassing
        Melt_production_Earth = (20*1000.0**3/(365*24*60*60)) * pm #Modern melt production rate, 20 km3/yr. Li et al 2016, m3/s, now in kg/s

        iron_ratio_mantle = 0.5*0.1 ## assume fixed oxidation state for the purposes of outgassing calculations
        actual_phi_surf_melt = 0.5 ## assume constant melt fraction for the purposes of outgassing calculations
        ## Given melt production and land fraction, calculate outgassing from seafloor and continents
        #Outgassing from subaerial volcanism:
        [F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L] = outgas_flux_cal_fast(Tsolidus,Pressure_surface,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)
        #Outgassing from submarine volcanism:
        [F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O] = outgas_flux_cal_fast(Tsolidus,Poverburd,iron_ratio_mantle,Mantle_mass,y[13],y[0],dmelt_dt*1000.0,Total_Fe_mol_fraction,actual_phi_surf_melt)

        ##Take weighted average for total outgassing flux
        [F_H2O,F_CO2,F_H2,F_CO,F_CH4,OG_O2_consumption] = np.array([F_H2O_O,F_CO2_O,F_H2_O,F_CO_O,F_CH4_O,OG_O2_consumption_O])*Ocean_fraction+np.array([F_H2O_L,F_CO2_L,F_H2_L,F_CO_L,F_CH4_L,OG_O2_consumption_L])*(1-Ocean_fraction)

        
        # Calculate max surface emplacement available for oxidation
        # 1e7 ~ 80 km3/yr emplacement, 1e8 ~ 800 km3/yr emplacement = 1mm/yr oxidized over whole surface
        surface_emplacement = dmelt_dt

        #molecular weights, not really used for much
        mu_O = 16.0
        mu_FeO_1_5 = 56.0 + 1.5*16.0
        mu_FeO = 56.0 + 16.0

        ### for failed runs error
        total_water_gain_interior = 0.0
        
        ## IMPORTANT:
        if y[8]< 973: #If surface temperature below serpentine stability, calculate crustal hydration - this is the main way that water is removed from the surface, so important part of deep water cycle
            water_c = F_Earth*(Ocean_fraction/0.7)*(dmelt_dt/Melt_production_Earth) #surface-dependent ingassing case
            #water_c = F_Earth*(((1-water_frac)*float(y[1]))/(1.4*(10**21)))*(dmelt_dt/Melt_production_Earth) # mass-dependent case, kg, includes current mass of hydrosphere.
            #water_c = 0 #no ingassing
            
            water_crust = water_c * np.max([0.0,1 - y[0]/Max_mantle_H2O])
            wet_oxidation = 0*wet_oxid_eff *(water_c/MFrac_hydrated) * Total_Fe_mass_fraction * iron_ratio_mantle # kg FeO/s of the hydrated crust, what fraction iron oxidized
            # so this means, of the crust that is hydrated, how much iron oxidized
            ## additionally, restrict water to be less than surface inventory
            total_water_loss_surf = water_crust + wet_oxidation*18.0/(3.0* mu_FeO) #kg H2O/s
            total_water_gain_interior = water_crust

        #print((t0/(365*24*60*60)), "Ocean fraction:", Ocean_fraction, "Surface Water:", y[1], "Precip:", y[3])

 
 
        #Save some useful outputs:
        y[27] = total_water_gain_interior 

        y[5] = Ocean_depth
        y[6] = 1-Ocean_fraction
        y[25] = Ocean_fraction

        y[22] = fO2
        y[23] = CO2_Pressure_surface
        
        y[24] = (fO2_pos*0.032 + atmo_H2O*0.018 + CO2_Pressure_surface*0.044 + ppn2*0.028)/Pressure_surface #Mean molecular weight

        ###########################################################################################

        # Calculate derivates for change in water and carbon reservoirs
        dy0_dt =   + total_water_gain_interior - F_H2O*0.018 #Change in mantle water with time
        dy1_dt = - 0*escape  - total_water_loss_surf + F_H2O*0.018   #Change in surface water with time
        
        dy13_dt = 0.0 ## Change in mantle CO2 with time (terms added below)
        dy12_dt = -dy13_dt - CO2_loss ## Change in surface CO2 with time, CO2 escape basically negligible for our purposes
        
        #Weather = 0.0
        #Outgas = 0.0
              
        Outgas = F_CO2*0.044 ## We've already calculate CO2 outgasing from outgassing calculation above
        Outgas_land = F_CO2_L*0.044 #subaerial volcanism
        Outgas_ocean = F_CO2_O*0.044 #submarine volcanism
        
        relative_S = ASR_new_fun(t0)/ASR_new_fun(4.5e9*365*24*60*60) #insolation relative to modern solar flux
   
        ## Here, comment in/out appropriate weathering function (MAC or WHAK)
        #[Weather,prec,prec_plim] = MAC_weathering_flux(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,float(y[24]),relative_S)
        #[Weather,prec,prec_plim] = weathering_flux(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,float(y[24]))
        #[Weather,prec] = MAC_LH_wx_flux(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,float(y[24]),relative_S)
        [Weather,prec] = MAC_LH_wx_flux_MC(t0,CO2_Pressure_surface,float(SurfT),y[7],water_frac,efold_weath,H2O_Pressure_surface,g,y[12],alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,float(y[24]),relative_S, Te, ts)

        #print("Weather:", Weather, "Precip:", prec)

        #print (Weather,Outgas)
        ## Now add weathering and outgassing fluxes to surface and mantle reservoir time-evolution equations
        dy12_dt = dy12_dt - Weather + Outgas
        dy13_dt = dy13_dt + Weather - Outgas

        ##Store some more useful outputs
        #y[3]= prec
        y[3] = prec if prec != 0 else 1e-10
        #y[4] = prec_plim
        #comment out y[4] when using MAC_LH_wx_flux, but if using WHAK or MAC without wind-driven evaporation, uncomment this line

        y[15] = 1e-12 * Outgas*365*24*60*60/0.044 #convert outgassing flux (Kg/s) to Tmol CO2/yr
        y[14] = 1e-12 * Weather*365*24*60*60/0.044 #convert weathering flux (Kg/s) to Tmol CO2/yr

       #some new diagnostics
        y[11] = 1e-12 * Outgas_land*365*24*60*60/0.044 #convert outgassing flux (Kg/s) to Tmol CO2/yr
        y[16] = 1e-12 * Outgas_ocean*365*24*60*60/0.044 #convert outgassing flux (Kg/s) to Tmol CO2/yr
        y[26] = F_H2O + F_CO2 + F_H2 + F_CO + F_CH4 + OG_O2_consumption

        
        MCO2 = 0.044  # Molar mass of CO2 (kg/mol)
        MH2O = 0.018  # Molar mass of water (kg/mol)
        R = 8.314     # Gas constant (J/mol/K)
        
        # Convert EM_ppCO2_o from bar to Pa
        PCO2_atm = newpCO2 * 1e5  # Partial pressure of CO2 in the atmosphere (Pa)

        # Atmospheric CO2 mass
        CO2_atm_mass = PCO2_atm * (4 * np.pi * rp**2 / g) * (MCO2 / (R * SurfT))

        # Oceanic CO2 mass (by subtraction)
        CO2_ocean_mass = y[12] - CO2_atm_mass

        y[17] = CO2_ocean_mass
        y[19] = CO2_atm_mass
        
        y[28] = F_H2O*0.018
        y[29] = dy0_dt
        y[30] = dy1_dt
        
       ###########################################################################################
        return [dy0_dt,dy1_dt,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,dy12_dt,dy13_dt,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    ########################################################################################### end forward model function
    ######################################################################################################################
    ######################################################################################################################

    # Initial conditions - only ICs that matter are solid and fluid water and CO2:
    ICs = [Init_solid_H2O,Init_fluid_H2O, rc, Init_solid_O,Init_fluid_O,Init_solid_FeO1_5,Init_solid_FeO,4000.0,300.0,0.0,0.0,0.0,Init_fluid_CO2,Init_solid_CO2,0.0,0.0,300.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.044,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    ### Various numerical inputs 
    #Actually solve the system of equations above:

    try:
        sol = solve_ivp(
        system_of_equations, 
        [Start_time * 365 * 24 * 60 * 60, Max_time * 365 * 24 * 60 * 60], 
        ICs, dense_output=True, 
        method='RK23', 
        t_eval=np.linspace(Start_time * 365 * 24 * 60 * 60, Max_time * 365 * 24 * 60 * 60, 2000), 
        max_step=1e5 * 365 * 24 * 60 * 60,  # Maximum step size
        rtol = 1e-2
    
        )

        if not sol.success:
            raise RuntimeError(f"solve_ivp failed: {sol.message}")
        
        return sol.t / (365 * 24 * 60 * 60) - Start_time, sol.y

    except Exception as e:
        print("\n Forward model failed")
        print(f"  Error: {e}")
        print(f"  Inputs that caused failure:")
        print(f"    init_water        = {init_water:.12e} kg")
        print(f"    Init_solid_H2O    = {Init_solid_H2O:.12e} kg")
        print(f"    Max_mantle_H2O    = {Max_mantle_H2O:.12e} kg")
        print(f"    Init_fluid_CO2    = {Init_fluid_CO2:.12e} kg")
        print(f"    Init_solid_CO2    = {Init_solid_CO2:.12e} kg")
        print(f"    mm                = {mm:.12f}")
        print(f"    n_out             = {n_out:.12f}")
        print(f"    Te                = {Te:.12f} K")
        print(f"    ts                = {ts:.12e} yrs")
        #raise

        failed_data = np.array([[init_water, Init_solid_H2O, Max_mantle_H2O,
                                 Init_fluid_CO2, Init_solid_CO2,
                                 mm, n_out, Te, ts]])

        #create directory if needed
        os.makedirs("debug_outputs", exist_ok=True)

        # append to CSV
        with open("debug_outputs/failed_forward_runs.csv", "a") as f:
            np.savetxt(f, failed_data, delimiter=",", fmt="%.12e", 
                       header="init_water,Init_solid_H2O,Max_mantle_H2O,Init_fluid_CO2,Init_solid_CO2,mm,n_out,Te,ts",
                       comments='' if os.stat(f.name).st_size == 0 else '')
        return None, None  # re-raise the error so the parallel code can log it
            
    
    total_time = sol.t
    total_y = sol.y

    total_time = total_time/(365*24*60*60)-Start_time ## set start time to zero


    #plotting
    plot_new_switch = "n"
    if plot_new_switch == "y":

        plt.figure(figsize=(10, 8))  # create multipanel figure

        #######################Figure 1####################
        # Plot Surface Water
        plt.subplot(2, 2, 1)
        plt.semilogy(total_time, total_y[1],  color="cornflowerblue")
        plt.ylabel('Surface water (kg)')
        plt.xlim(0, 5e9)
        plt.xlabel('Time (yrs)')

        # Plot Surface Temperature 
        plt.subplot(2, 2, 2)
        plt.plot(total_time, total_y[8],  color="cornflowerblue")
        plt.ylabel('Surface Temperature (K)')
        plt.xlim(0, 5e9)
        #plt.legend(loc="upper left")
        plt.xlabel('Time (yrs)')

        # Plot Precipitation
        plt.subplot(2, 2, 3)
        plt.semilogy(total_time, total_y[3],  color="cornflowerblue")
        plt.ylabel('Precipitation (m/yr)')
        plt.xlim(0, 5e9)
        plt.ylim(1e-5, 1e1)
        #plt.legend(loc="lower left")
        plt.xlabel('Time (yrs)')

        #Atmosphere + Ocean CO2 Reservoir
        plt.subplot(2, 2, 4)
        plt.semilogy(total_time, total_y[12], color="cornflowerblue")
        plt.ylabel('fCO$_2$ Pressure (bar)')
        plt.xlim(0, 5e9)
        plt.xlabel('Time (yrs)')


        plt.tight_layout()
    
    return [total_time,total_y]
#[total_time,total_y]=forward_model(1e18)

# Show the final plot
#pylab.show()



