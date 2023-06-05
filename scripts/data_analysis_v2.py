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
AOH = 15 
ADMPO = 15

#Guess del centro del espectro
G = 3352.1136018756933
#Guess de la intensidad
I1 = 0.13530184050568167
I2 = 0.11196552426208073
I3 = 2*I1
#Guess del 0 de la funcion
y0 = -9.158014352716684e-05
#Guess del ancho de la gaussiana
a1 = 0.7237514000972604
a2 = 0.9193855743534524
a3 = 0.7237514000972604

ini = 600
fin = 1450

#definimos el conjunto de gaussianas. x son los datos, el resto son los parámetros a ajustar
#definimos los centros de cada gaussiana según los parámetros a ajustar
# los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
# el centro de la gaussiana i-ésima está dado por xi

def gaussian(x, AOH, ADMPO, G, I1, I2, I3, y0, a1, a2, a3):
#def gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0):
#def gaussian(x, I1, I2, y0, a1, a2):

    x1 = (G-AOH/2)
    x2 = (G-AOH*3/2)
    x3 = (G+AOH/2)
    x4 = (G+AOH*3/2)

    x5 = (G+ADMPO)
    x6 = (G)
    x7 = (G-ADMPO)
    
    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)
    pref7 = -(x-x7)

    # Radical OH
    y1 = I3*pref1*np.exp(-0.5*((x-x1)/a3)**2)
    y2 = I1*pref2*np.exp(-0.5*((x-x2)/a1)**2)
    y3 = I3*pref3*np.exp(-0.5*((x-x3)/a3)**2)
    y4 = I1*pref4*np.exp(-0.5*((x-x4)/a1)**2)
    
    # Interacción DMPO
    y5 = I2*pref5*np.exp(-0.5*((x-x5)/a2)**2)
    y6 = I2*pref6*np.exp(-0.5*((x-x6)/a2)**2)
    y7 = I2*pref7*np.exp(-0.5*((x-x7)/a2)**2)
    
    y = y1+y2+y3+y4+y5+y6+y7+y0

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
y0_guess = y0
G_guess = G
AOH_guess = AOH
ADMPO_guess = ADMPO

#A4_guess = A4
a1_guess = a1
a2_guess = a2
a3_guess = a3


pguess = [AOH_guess, ADMPO_guess, G_guess, I1_guess, I2_guess, I3_guess,y0_guess, a1_guess, a2_guess, a3_guess]
#pguess = [AOH_guess, ADMPO_guess, AMgO_guess, G_guess, I1_guess, I2_guess, I3_guess, y0_guess]
#pguess = [I1_guess, I2_guess, y0_guess, a1_guess, a2_guess]

# Fit the data
popt, pcov = curve_fit(gaussian, x[ini:fin], y[ini:fin], p0 = pguess)

# Results
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7],  popt[8],  popt[9],  popt[10]
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]
AOH, ADMPO, G, I1, I2, I3, y0, a1, a2, a3 = popt[0], popt[1], popt[2], popt[3], popt[4],  popt[5],  popt[6], popt[7], popt[8], popt[9]

#fit = gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3)
#fit = gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0)
fit = gaussian(x[ini:fin], AOH, ADMPO, G, I1, I2, I3, y0, a1, a2,a3)



plt.plot(x[ini:fin], y[ini:fin], "o", label="Fibra " + i, color = color, linewidth=2)
#plt.xlim(3325,3425)
plt.plot(x[ini:fin], fit, label="$\\tau_c$ = " + str(round(t_c_fit,4)) + " - $M_0 = $" + str(round(M0_fit,4)), color = color, linewidth=2)
##perr = np.sqrt(np.diag(pcov))
#print(perr)
print("AOH = ", AOH, "ADMPO = ", ADMPO, "G = ",  G, "I1 = ",  I1, "I2 = ",  I2, "I3 = ", I3, "y0 = ", y0, "a1 = ", a1, "a2 = ", a2, "a3 = ", a3)



ax.legend(fontsize=14)
ax.set_xlabel(r"x [ms]", fontsize=16)
ax.set_ylabel(r"Señal $\mathrm{NOGSE}$ [u.a.]", fontsize=16)
ax.tick_params(axis='both', labelsize=15)
title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || $G$ = {} mT/m".format(T_nogse[0], int(n[0]), g[0]), fontsize=18)
plt.savefig(r"../results_" + str (file_name) + "/NOGSE_vs_x/T_NOGSE=" + str (T_NOGSE) + "/NOGSE_vs_x_rest_" + str (T_NOGSE) +"_"+ str(round(g[0])) + ".pdf")
plt.savefig(r"../results_" + str (file_name) + "/NOGSE_vs_x/T_NOGSE=" + str (T_NOGSE) + "/NOGSE_vs_x_rest_" + str (T_NOGSE) +"_"+ str(round(g[0])) + ".png")

#hpp = 2*a*I*np.exp(-0.5) #altura
#Hpp = 2*a #ancho

#print("hpp = ", hpp, "Hpp = ", Hpp)

plt.show()