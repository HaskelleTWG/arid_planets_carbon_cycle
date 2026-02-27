
import numpy as np
import matplotlib.pyplot as plt

LH_switch = "y"
if LH_switch == "y":
    temperatures = np.linspace(200, 450, 100)

    def saturation_vapor_pressure_liquid(T):
        """
        Calculate saturation vapor pressure for water using the constant-L form
        of the Clausius-Clapeyron relation.

        Parameters:
        - T: Temperature in Kelvin

        Returns:
        - P: Saturation vapor pressure in Pascals at temperature T
        """

        # constants
        P0 = 611.2  # saturation vapor pressure at reference temperature (T0 = 273.15 K) in Pa
        L = 2.5e6   # latent heat of vaporization in J/kg
        Ra = 461.5  # gas constant for the substance which is condensing J/(kg*K)
        T0 = 273.15  # reference temperature in Kelvin

        # calculate saturation vapor pressure
        P = P0 * np.exp(-(L / Ra) * ((1 / T) - (1 / T0)))

        return P

    def saturation_vapor_pressure_solid(T):

        # constants
        P0 = 611.2  # saturation vapor pressure at reference temperature (T0 = 273.15 K) in Pa
        L = 2.8e6   # latent heat of sublimation in J/kg
        Ra = 461.5  # gas constant for the substance which is condensing J/(kg*K)
        T0 = 273.15  # reference temperature in Kelvin

        # calculate saturation vapor pressure
        P = P0 * np.exp(-(L / Ra) * ((1 / T) - (1 / T0)))

        return P


    psat_sol = np.piecewise(temperatures, [temperatures < 273.15, temperatures >= 273.15],
                 [saturation_vapor_pressure_solid, saturation_vapor_pressure_liquid])


    # calculate saturation vapor pressure for each temperature
    psat_liq = saturation_vapor_pressure_liquid(temperatures)
    #psat_sol = saturation_vapor_pressure_solid(temperatures)

    '''
    # plotting the saturation vapor pressure (uncomment to check the values)
    plt.plot(temperatures, psat_liq, color = 'cornflowerblue', label = 'Vapor pressure over liquid')
    plt.plot(temperatures, psat_sol, color = 'orange', linestyle = 'dashed', label = 'Vapor pressure over ice')
    plt.title('Saturation Vapor Pressure for Water')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Saturation Vapor Pressure (Pa)')
    plt.yscale('log')
    plt.legend(loc = 'lower right')
    plt.ylim(0.1, 1e5)
    plt.xlim(200, 320)
    plt.grid(True)
    plt.show()
    '''

    # In[2]:


    # Constants
    L = 2.5e6  # latent heat of vaporization (J/kg)
    Rw = 461.5  # specific gas constant (J/(kg*K))
    U = 10  # mean horizontal wind speed at the top of the surface layer (m/s)
    CD = 0.001  # drag coefficient
    hsa = 0.7  # relative humidity at the outer edge of the surface layer

    # temperature range from 200 to 450K
    Tsa_range = np.linspace(200, 450, 250)

    # calculate saturation vapor pressure for each temperature in Tsa_range
    psat_liq = saturation_vapor_pressure_liquid(Tsa_range)
    psat_sol = np.piecewise(Tsa_range, [Tsa_range < 273.15, Tsa_range >= 273.15],
                 [saturation_vapor_pressure_solid, saturation_vapor_pressure_liquid])

    # function to calculate latent heat flux
    def latent_heat_flux(T):
        FL = CD * U * saturation_vapor_pressure_liquid(T)
        return (1 - hsa) * (L / (Rw * T)) * FL

    # calculate latent heat flux for each temperature
    LH_flux = latent_heat_flux(Tsa_range)

    '''
    # plotting the latent heat flux
    plt.plot(Tsa_range, LH_flux, label='Latent Heat Flux')
    plt.xlabel('Temperature at top of surface layer, Tsa (K)')
    plt.ylabel('Latent Heat Flux, Eo (W/m$^2$)')
    plt.title('Latent Heat Flux')
    plt.legend()
    plt.yscale('log')
    #plt.ylim(0,600)
    plt.grid(True)
    plt.show()


    # pair temperature with corresponding LH value
    Tsa_LH = list(zip(Tsa_range, LH_flux))

    print("Latent Heat Flux and Temperature:")
    for pair in Tsa_LH:
        print(f"Temperature: {pair[0]} K, LHF: {pair[1]} W/m^2")

    '''

