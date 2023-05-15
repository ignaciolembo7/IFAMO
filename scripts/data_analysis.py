import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

"""
#Centros
x1 = 3360
x2 = 3345
x3 = 3375
x4 = 3330
x5 = 3307
x6 = 3394
x7 = 3337
x8 = 3352
x9 = 3367
"""

#Guess de constantes hiperfinas
A3 = 15 
A2 = 45
A1 = 87
A4 = 15
#Guess del centro del espectro
G = 3352
#Guess de la intensidad
I1 = 0.1
I2 = 0.2
I3 = 0.3
I4 = 0.02
#Guess del 0 de la funcion
y0 = 0.1
#Guess del ancho de la gaussiana
a1 = 4
a2 = 0.2
a3 = 0.3

#definimos el conjunto de gaussianas

def gaussian(x, I1, I2, I3, I4, y0, a1, a2, a3):
    #x son los datos, el resto son los parámetros a ajustar
    #definimos los centros de cada gaussiana según los parámetros a ajustar

    # los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
    # el centro de la gaussiana i-ésima está dado por xi

    x1 = (G-A1/2)
    x2 = (G-A2/2)
    x3 = (G-A3/2)
    x4 = (G+A3/2)
    x5 = (G+A2/2)
    x6 = (G+A1/2)

    x7 = (G-A4)
    x8 = G
    x9 = (G+A4)

    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)
    pref7 = -(x-x7)
    pref8 = -(x-x8)
    pref9 = -(x-x9)

    y1 = I1*pref1*np.exp(-0.5*((x-x1)/a1)**2)
    y2 = I2*pref2*np.exp(-0.5*((x-x2)/a2)**2)
    y3 = I3*pref3*np.exp(-0.5*((x-x3)/a2)**2)
    y4 = I3*pref4*np.exp(-0.5*((x-x4)/a2)**2)
    y5 = I2*pref5*np.exp(-0.5*((x-x5)/a2)**2)
    y6 = I1*pref6*np.exp(-0.5*((x-x6)/a1)**2)

    y7 = I4*pref7*np.exp(-0.5*((x-x7)/a3)**2)
    y8 = I4*pref8*np.exp(-0.5*((x-x8)/a3)**2)
    y9 = I4*pref9*np.exp(-0.5*((x-x9)/a3)**2)

    y = y1+y2+y3+y4+y5+y6+y7+y8+y9+y0

    return y

# Obtain xdata and ydata
data = np.loadtxt("AM02_pH5_RT_10min.dat", delimiter="	") #Tenes el campo en la columna 1 y la señal en la columna 2 (la 3 ignorala)
                                                           #Cambiar , a . para que ande.

x = data[:,0]
y = data[:,1]


# Initial guess of the parameters 
I1_guess = I1
I2_guess = I2
I3_guess = I3
I4_guess = I4
y0_guess = y0
G_guess = G
A1_guess = A1
A2_guess = A2
A3_guess = A3
A4_guess = A4
a1_guess = a1
a2_guess = a2
a3_guess = a3

#pguess = [A1_guess, A2_guess, A3_guess, A4_guess,  G_guess, I1_guess, I2_guess, I3_guess, I4_guess, y0_guess, a_guess]
pguess = [I1_guess, I2_guess, I3_guess, I4_guess, y0_guess, a1_guess, a2_guess, a3_guess]

# Fit the data
popt, pcov = curve_fit(gaussian, x, y, p0 = pguess)

# Results
#A1, A2, A3, A4, G, I1, I2, I3, I4, y0, a = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7], popt[8], popt[9], popt[10]
I1, I2, I3, I4, y0, a1, a2, a3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7],

#fit = gaussian(x, G, A1, A2, A3, A4, I1, I2, I3, I4, y0, a)
fit = gaussian(x, I1, I2, I3, I4, y0, a1, a2, a3)

plt.plot(x,y)
#plt.xlim(3325,3425)
plt.plot(x,fit)
##perr = np.sqrt(np.diag(pcov))
#print(perr)
print("A1 = ", A1, "A2 = ", A2, "A2 = ", A3, "A3 = ", A3, "A4 = ", A4, "G = ",  G, "I1 = ",  I1, "I2 = ",  I2, "I3 = ",  I3, "I4 = ",  I4, "y0 = ", y0, "a1 = ", a1, "a2 = ", a2, "a3 = ", a3 )

#hpp = 2*a*I*np.exp(-0.5) #altura
#Hpp = 2*a #ancho

#print("hpp = ", hpp, "Hpp = ", Hpp)

plt.show()