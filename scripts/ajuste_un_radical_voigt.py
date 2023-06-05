import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

ini = 600
fin = 1450

def pseudovoigt_derivative(x, A, sigma, fraction, center):
    """
    Calculate the derivative of a pseudo-Voigt function at x.
    """
    gaussian_derivative = -(x - center) / (sigma ** 2) * A * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    lorentzian_derivative = -2 * A * (x - center) / (np.pi * (sigma**2 + (x - center)**2))
    return fraction * gaussian_derivative + (1 - fraction) * lorentzian_derivative

#Guess de constantes hiperfinas
AOHh = 15
AOHn = 14.6
ADMPO = 14.76

#Guess del centro del espectro
G = 3353.2

#Guess de la intensidad
I1 = 0.4
I2 = 0.02

#Guess del 0 de la funcion
y0 = -9.158014352716684e-05

#Guess del ancho de la gaussiana
a1 = 0.7
a2 = 0.7

#Guess del porcentaje de la gaussiana
frac1 = 1
frac2 = 1

ini = 600
fin = 1450

#definimos el conjunto de gaussianas. x son los datos, el resto son los parámetros a ajustar
#definimos los centros de cada gaussiana según los parámetros a ajustar
# los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
# el centro de la gaussiana i-ésima está dado por xi

def peaks(x, AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2, frac1):
#def gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0):
#def gaussian(x, I1, I2, y0, a1, a2):

    x1 = (G-AOHh/2-AOHn)
    x2 = (G-AOHh/2)
    x3 = (G-AOHh/2+AOHn)
    x4 = (G+AOHh/2-AOHn)
    x5 = (G+AOHh/2)
    x6 = (G+AOHh/2+AOHn)

    x7 = (G+ADMPO)
    x8 = (G)
    x9 = (G-ADMPO)
    
    y1 = pseudovoigt_derivative(x, I1, a1, frac1, x1)
    y2 = pseudovoigt_derivative(x, I1, a1, frac1, x2)
    y3 = pseudovoigt_derivative(x, I1, a1, frac1, x3)
    y4 = pseudovoigt_derivative(x, I1, a1, frac1, x4)
    y5 = pseudovoigt_derivative(x, I1, a1, frac1, x5)
    y6 = pseudovoigt_derivative(x, I1, a1, frac1, x6)

    y7 = pseudovoigt_derivative(x, I2, a2, 1, x7)
    y8 = pseudovoigt_derivative(x, I2, a2, 1, x8)
    y9 = pseudovoigt_derivative(x, I2, a2, 1, x9)


    y = y1+y2+y3+y4+y5+y6+y7+y8+y9+y0

    return y

# Obtain xdata and ydata
data = np.loadtxt("AM02_pH5_RT_40min_baselineA.dat", delimiter="	") #Tenes el campo en la columna 1 y la señal en la columna 2 (la 3 ignorala)
                                                           #Cambiar , a . para que ande.

x = data[:,0]
y = data[:,1]


# Initial guess of the parameters 
I1_guess = I1
I2_guess = I2
y0_guess = y0
G_guess = G
AOHh_guess = AOHh
AOHn_guess = AOHn
ADMPO_guess = ADMPO
a1_guess = a1
a2_guess = a2

frac1_guess = frac1
frac2_guess = frac2

pguess = [AOHn, AOHh, ADMPO_guess, G_guess, I1_guess, I2_guess,y0_guess, a1_guess, a2_guess, frac1_guess]

# Fit the data
popt, pcov = curve_fit(peaks, x[ini:fin], y[ini:fin], p0 = pguess)

# Results

AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2, frac1= popt[0], popt[1], popt[2], popt[3], popt[4],  popt[5],  popt[6], popt[7], popt[8],  popt[9]

fit = peaks(x[ini:fin], AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2, frac1)


perr = np.sqrt(np.diag(pcov))

print("Aducto 1:")
print("An = ", popt[0], " +- ", perr[0])
print("Ah = ", popt[1], " +- ", perr[1])
print("Intensidad = ", popt[4], " +- ", perr[4])
print("Ancho = ", popt[7], " +- ", perr[7])
print("frac = ", popt[9], " +- ", perr[9])
print("----------------------------------------")
print("Aducto 2:")
print("An = ", popt[2], " +- ", perr[2])
print("Intensidad = ", popt[5], " +- ", perr[5])
print("Ancho = ", popt[8], " +- ", perr[8])
print("frac2 = 1")
print("----------------------------------------")
print("G = ", popt[3],  " +- ", perr[3])
print("y0 = ", popt[6],  " +- ", perr[6])

fig, ax = plt.subplots(figsize=(9,6)) 
ax.plot(x[ini:fin], y[ini:fin], "o", label="Medicion", color = "black", markersize=2)
ax.plot(x[ini:fin], fit, label="Ajuste", color = "r", linewidth=2)

ax.set_xlabel(r"Campo Magnético (G)", fontsize=16)
ax.set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=16)
ax.tick_params(axis='both', labelsize=15, direction='in', length = 7)
ax.legend(fontsize=14)

plt.savefig(r"../ajuste_un_radical_voigt.pdf")
plt.savefig(r"../ajuste_un_radical_voigt.png",dpi=600)
table = np.vstack((x[ini:fin], y[ini:fin],fit)) #sirve para guardar tablas de graficas 
np.savetxt(r"../ajuste_un_radical_voigt.txt", table.T, delimiter=' ', newline='\n')

plt.show()