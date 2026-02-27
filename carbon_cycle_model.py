import numpy as np
from numba import jit
from Hypso_fun import hypso_fun ## load in hypsometric curves
from Latent_Heat_Flux import saturation_vapor_pressure_liquid
from Latent_Heat_Flux import latent_heat_flux #loads in new precipitation parameterization described in Section 2.2.3

## Calculate silicate weathering flux, in kg/s
#@jit(nopython=True) 

## WHAK Model
def weathering_flux(t0,CO2_Pressure_surface,SurfT,Tp,water_frac,T_efold,H2O_Pressure_surface,g,y12,alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,MMW): # most of these inputs not used   
    if (CO2_Pressure_surface < 0.0): # no CO2 left to be weathered.
        return 0.0,0.0,0.0
    if (SurfT<647):  #weathering only occurs if below critical point
        infront= 4000  #coefficient chosen to reproduce approximate Earth fluxes
        Supply_limit = supp_lim
        Te = T_efold 
        alpha = alpha_exp 
        Ocean_depth = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
      
        if Ocean_depth > Max_depth:
            Ocean_fraction = 1.0
            Land_fraction = 0.0
            LF = 0.0
        else:
            water_volume = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface *4 *np.pi * 6371000**2/(g*1000)
            Land_fraction = hypso_fun(water_volume)
            Ocean_fraction = 1 - Land_fraction
            LF = 1 
    
        Tdeep = SurfT + Ocean_depth * 5e-4 * g * SurfT / 4000.0 # adiabatic lapse rate in ocean
        Crustal_production = Plate_velocity/(1e-9) 
     
        Seafloor_weathering =(infront/4.0) * 10**(-0.3*(ocean_pH-7.727)) * Crustal_production * np.exp(-(90000.0/8.314)*(1.0/Tdeep-1.0/285))
        Continental_weathering =infront * LF * ((CO2_Pressure_surface/1e5)/(350e-6) )**alpha * np.exp((SurfT-285)/Te)

        Weathering = (Continental_weathering + 0*Seafloor_weathering) #to stop weathering as dry out
        
        if Weathering>Supply_limit: #Weathering flux never exceeds supply limit
            return Supply_limit,0.0,0.0
    else:
        Weathering = 0.0
    return Weathering,0.0,0.0
    

### MAC silicate weathering and seafloor weathering here, not used for any plots in Manuscript but provided for comparison to Graham and Pierrehubert (2020)
def MAC_weathering_flux(t0,CO2_Pressure_surface,SurfT,Tp,water_frac,T_efold,H2O_Pressure_surface,g,y12,alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,MMW,solar):


    if (CO2_Pressure_surface < 0.0): # no CO2 left to be weathered.
        return 0.0,0.0,0.0
    if (water_frac<0.9999999999)and(SurfT<647)and(H2O_Pressure_surface>0): #weathering only occurs if below critical point
        infront= 4000  #coefficient chosen to reproduce approximate Earth fluxes
        Supply_limit = supp_lim
        Te = T_efold 
        alpha = alpha_exp 
        Ocean_depth = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
      
        if Ocean_depth > Max_depth:
            Ocean_fraction = 1.0
            Land_fraction = 0.0
            #LF = 0.0
        else:
            water_volume = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface *4 *np.pi * 6371000**2/(g*1000)
            Land_fraction = hypso_fun(water_volume)
            Ocean_fraction = 1 - Land_fraction

        gamma = Land_fraction
        pCO2 = CO2_Pressure_surface/1e5
        T = SurfT
        mu = np.exp(1)**2
        LAMBDA = 0.0014 #weak T dep ignored here, 
        Tref = 288.0
        Te = 11.1
        pCO2ref = 280e-6
        n = 0.316
        beta = 0.2
        L = 1.0
        phi = 0.1
        psf = 12728.0
        A = 100.0
        Xr = 0.36
        keffref = 8.7e-6
        m = 0.27
        ts = 1e4 #1e5 originally
        BGAMMA = 0.2
        epsilon = 0.03
        albedo = 0.3
 
        Ceq = 1e3*LAMBDA * (pCO2)**n #ignores T dep on equilibrium, need to convert mol/L to mol /m3
        pref = 0.99
        p = pref * (1 + epsilon * (T - Tref)) #m/yr
        Latent = 1.918e9 * (T/(T - 33.91))**2 #J/m^3, latent heat flux already accounts for density
        plim = 365*24*60*60*(1 - gamma ) * solar*1368.0/4.0 * (1- albedo)/ Latent #m/yr
        if p > plim:
            p = plim
        q = BGAMMA * p 
    
        kinetic = keffref*np.exp((T-Tref)/Te)*(pCO2/pCO2ref)**beta
        thermo = q*Ceq
        alpha = L * phi * psf * A * Xr * mu
        w = alpha/(1./kinetic + m*A*ts + alpha / thermo) #mol/m2/yr
        Wmac = 0.044*w*4*np.pi*6371000**2/(365*24.0*60*60) #kg/s
        return gamma*Wmac,p,plim 
    else:
        return 0.0,0.0,0.0


### MAC silicate weathering with latent heat flux for Nominal Plots
def MAC_LH_wx_flux(t0,CO2_Pressure_surface,SurfT,Tp,water_frac,T_efold,H2O_Pressure_surface,g,y12,alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,MMW,solar):

    if (CO2_Pressure_surface < 0.0): # no CO2 left to be weathered.
        return 0.0,0.0,0.0
    if (water_frac<0.9999999999)and(SurfT<647)and(H2O_Pressure_surface>0): #weathering only occurs if below critical point
        infront= 4000  #coefficient chosen to reproduce approximate Earth fluxes
        Supply_limit = supp_lim
        Te = T_efold 
        alpha = alpha_exp 
        Ocean_depth = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
      
        if Ocean_depth > Max_depth:
            Ocean_fraction = 1.0
            Land_fraction = 0.0
        else:
            water_volume = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface *4 *np.pi * 6371000**2/(g*1000)
            Land_fraction = hypso_fun(water_volume)
            Ocean_fraction = 1 - Land_fraction

        gamma = Land_fraction
        pCO2 = CO2_Pressure_surface/1e5
        T = SurfT
        mu = np.exp(1)**2
        LAMBDA = 0.0014 #weak T dep ignored here, 
        Tref = 288.0
        Te = 11.1
        ts = 1e4 
        pCO2ref = 280e-6
        n = 0.316
        beta = 0.2
        L = 1.0
        phi = 0.1
        psf = 12728.0
        A = 100.0
        Xr = 0.36
        keffref = 8.7e-6
        m = 0.27
        BGAMMA = 0.2
        epsilon = 0.03
        albedo = 0.3
 
        Ceq = 1e3*LAMBDA * (pCO2)**n #ignores T dep on equilibrium, need to convert mol/L to mol /m3
        p = ((latent_heat_flux(T) / 2200000) * (1-gamma) / 1000) * 365 * 24 * 60 * 60  # Conversion to years, m/yr

        q = BGAMMA * p
    
        kinetic = keffref*np.exp((T-Tref)/Te)*(pCO2/pCO2ref)**beta
        thermo = q*Ceq
        alpha = L * phi * psf * A * Xr * mu
        w = alpha/(1./kinetic + m*A*ts + alpha / thermo) #mol/m2/yr
        Wmac = 0.044*w*4*np.pi*6371000**2/(365*24.0*60*60) #kg/s

        #update 
        min_weathering_flux = 1e-10  # Some small positive value
        Wmac = max(Wmac, min_weathering_flux)
        
        return (gamma*Wmac),p #[w,kin,thermo]
    else:
        return 0.0,0.0

### MAC silicate weathering with latent heat flux for Monte Carlo runs
def MAC_LH_wx_flux_MC(t0,CO2_Pressure_surface,SurfT,Tp,water_frac,T_efold,H2O_Pressure_surface,g,y12,alpha_exp,supp_lim,ocean_pH,omega_ocean,Plate_velocity,MMW,solar, Te, ts):

    if (CO2_Pressure_surface < 0.0): # no CO2 left to be weathered.
        return 0.0,0.0 #,0.0
    if (water_frac<0.9999999999)and(SurfT<647)and(H2O_Pressure_surface>0): #weathering only occurs if below critical point
        infront= 4000  #coefficient chosen to reproduce approximate Earth fluxes
        Supply_limit = supp_lim
        Te = T_efold 
        alpha = alpha_exp 
        Ocean_depth = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface / (g*1000) ## max ocean depth continents 11.4 * gEarth/gplanet (Cowan and Abbot 2014)
        Max_depth = 11400 * (9.8 / g) 
      
        if Ocean_depth > Max_depth:
            Ocean_fraction = 1.0
            Land_fraction = 0.0
            
        else:
            water_volume = (0.018/MMW)*(1-water_frac) * H2O_Pressure_surface *4 *np.pi * 6371000**2/(g*1000)
            Land_fraction = hypso_fun(water_volume)
            Ocean_fraction = 1 - Land_fraction
            

        gamma = Land_fraction
        pCO2 = CO2_Pressure_surface/1e5
        T = SurfT
        mu = np.exp(1)**2
        LAMBDA = 0.0014 #weak T dep ignored here, 
        Tref = 288.0
        #Te = 11.1 # tested in Monte Carlo
        #ts = 1e4, # tested in Monte Carlo
        pCO2ref = 280e-6
        n = 0.316
        beta = 0.2
        L = 1.0
        phi = 0.1
        psf = 12728.0
        A = 100.0
        Xr = 0.36
        keffref = 8.7e-6
        m = 0.27
        BGAMMA = 0.2
        epsilon = 0.03
        albedo = 0.3
 
        Ceq = 1e3*LAMBDA * (pCO2)**n #ignores T dep on equilibrium, need to convert mol/L to mol /m3
        p = ((latent_heat_flux(T) / 2200000) * (1-gamma) / 1000) * 365 * 24 * 60 * 60  # Conversion to years, m/yr

        q = BGAMMA * p
    
        kinetic = keffref*np.exp((T-Tref)/Te)*(pCO2/pCO2ref)**beta
        thermo = q*Ceq
        alpha = L * phi * psf * A * Xr * mu

        ##update to w 
        denom_w = (1./kinetic + m*A*ts + alpha / thermo)
        w = alpha / (denom_w + 1e-20)
        #w = alpha/(1./kinetic + m*A*ts + alpha / thermo) #mol/m2/yr

        Wmac = 0.044*w*4*np.pi*6371000**2/(365*24.0*60*60) #kg/s
        min_weathering_flux = 1e-10  # some small positive value
        Wmac = max(Wmac, min_weathering_flux)
        
        return (gamma*Wmac),p 
    else:
        return 0.0,0.0

