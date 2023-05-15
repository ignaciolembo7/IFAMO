# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 11:34:19 2022

@author: ignacio
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


#Parametros a ajustar
Ah = 3.4858412775383063 #constante hiperfina N2, están en G
An = 16.06462305089914 #constante hiperfina H2
G = 3370.4215361058336
I = 1
y0 = 0.1
a = 0.5 #ancho de cada gaussiana

#definimos el conjunto de gaussianas

def gaussian(x, An, Ah, G, I, a, y0):
    #x son los datos, el resto son los parámetros a ajustar
    #definimos los centros de cada gaussiana según los parámetros a ajustar

    # los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
    # el centro de la gaussiana i-ésima está dado por xi

    #definimos los centros según los parámetros a ajustar
    x1 = (G-Ah/2-An)
    x2 = (G-Ah/2)
    x3 = (G-Ah/2+An)
    x4 = (G+Ah/2-An)
    x5 = (G+Ah/2)
    x6 = (G+Ah/2+An)

    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)

    y1 = pref1*np.exp(-0.5*((x-x1)/a)**2)
    y2 = pref2*np.exp(-0.5*((x-x2)/a)**2)
    y3 = pref3*np.exp(-0.5*((x-x3)/a)**2)
    y4 = pref4*np.exp(-0.5*((x-x4)/a)**2)
    y5 = pref5*np.exp(-0.5*((x-x5)/a)**2)
    y6 = pref6*np.exp(-0.5*((x-x6)/a)**2)

    y = I*(y1+y2+y3+y4+y5+y6)+y0
    
    return y


data = np.loadtxt("berlina_27Oct_t16.txt")

# Obtain xdata and ydata

x = data[:,1]
y = data[:,3]


# Initial guess of the parameters 
An_guess = An
Ah_guess = Ah
G_guess = G
I_guess = I
y0_guess = y0
a_guess = a 

pguess = [An_guess, Ah_guess, G_guess, I_guess, a_guess, y0_guess]

# Fit the data
popt, pcov = curve_fit(gaussian, x, y, p0 = pguess)

# Results
An, Ah, G, I, a, y0 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5]

fit = gaussian(x, An, Ah, G,  I, a, y0)

plt.plot(x,y)
plt.xlim(3325,3425)
plt.plot(x,fit)
print("I = ", 2*I*a*np.exp(-0.5), "An = ", An, "Ah = ", Ah, "G = ", G, "a = ", a, "y0 = ", y0)
plt.show()