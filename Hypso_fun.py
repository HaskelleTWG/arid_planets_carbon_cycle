import numpy as np
import pylab

Area_Earth = 4.0*np.pi*6371000.0**2

from scipy import interpolate

#load hypsometric data for either Earth or Venus
ace = np.loadtxt('Venus_hypso.txt')
#ace = np.loadtxt('Earth_hypso.txt')

inter_fun = interpolate.interp1d(ace[:,0],ace[:,1])

actual_x = np.linspace(0,100,2000)
actual_y = inter_fun(actual_x)


def hypso_fun(water_volume): ## Function takes volume of surface water and converts to a land fraction based on hypsometry
    Volume = 0.0
    i = len(actual_y)-1
    final_index = 0
    while i > 0:
        New_Area = Area_Earth*0.01*(-actual_x[i]+actual_x[-1])
        Delta_h = 1000*(actual_y[i-1]-actual_y[i])
        New_volume = New_Area*Delta_h 
        Volume = Volume + New_volume
        if Volume>water_volume:
            final_index = np.copy(i)    
            i = 0
        else:
            i = i-1
    return actual_x[final_index+1]/100.0

'''
def hypso_fun(water_volume):
    New_height = water_volume/(Area_Earth) #in m
    Max_depth = 11400.0
    return 1-New_height/Max_depth
'''
