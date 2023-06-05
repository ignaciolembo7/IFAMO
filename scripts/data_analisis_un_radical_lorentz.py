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
AOHh = 15
AOHn = 14.6
ADMPO = 14.76


#Guess del centro del espectro
G = 3352
#Guess de la intensidad
I1 = 0.4
I2 = 0.02

#Guess del 0 de la funcion
y0 = -9.158014352716684e-05

#Guess del ancho de la gaussiana
a1 = 0.7
a2 = 0.7

ini = 600
fin = 1450

#definimos el conjunto de gaussianas. x son los datos, el resto son los parámetros a ajustar
#definimos los centros de cada gaussiana según los parámetros a ajustar
# los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
# el centro de la gaussiana i-ésima está dado por xi

def lorentzian(x, AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2):
#def gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0):
#def gaussian(x, I1, I2, y0, a1, a2):

    x1 = (G-AOHh/2-AOHn) 
    x2 = (G-AOHh/2)
    x3 = (G-AOHh/2+AOHn)
    x4 = (G+AOHh/2-AOHn)
    x5 = (G+AOHh/2)
    x6 = (G+AOHh/2+AOHn)

    x13 = (G+ADMPO)
    x14 = (G)
    x15 = (G-ADMPO)
    
    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)
    pref13 = -(x-x13)
    pref14 = -(x-x14)
    pref15 = -(x-x15)

    # Radical OH
    y1 = I1*a1*pref1*(4*(x-x1)**2 + a1**2)**(-2)
    y2 = I1*a1*pref2*(4*(x-x2)**2 + a1**2)**(-2)
    y3 = I1*a1*pref3*(4*(x-x3)**2 + a1**2)**(-2)
    y4 = I1*a1*pref4*(4*(x-x4)**2 + a1**2)**(-2)
    y5 = I1*a1*pref5*(4*(x-x5)**2 + a1**2)**(-2)
    y6 = I1*a1*pref6*(4*(x-x6)**2 + a1**2)**(-2)
        

    # Interacción DMPO
    y13 = I2*a2*pref13*(4*(x-x13)**2 + a2**2)**(-2)
    y14 = I2*a2*pref14*(4*(x-x14)**2 + a2**2)**(-2)
    y15 = I2*a2*pref15*(4*(x-x15)**2 + a2**2)**(-2)

    y = y1+y2+y3+y4+y5+y6+y13+y14+y15+y0

    return y

# Obtain xdata and ydata
data = np.loadtxt("AM02_pH5_RT_10min.dat", delimiter="	") #Tenes el campo en la columna 1 y la señal en la columna 2 (la 3 ignorala)
                                                        
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


pguess = [AOHn, AOHh, ADMPO_guess, G_guess, I1_guess, I2_guess,y0_guess, a1_guess, a2_guess]
#pguess = [AOH_guess, ADMPO_guess, AMgO_guess, G_guess, I1_guess, I2_guess, I3_guess, y0_guess]
#pguess = [I1_guess, I2_guess, y0_guess, a1_guess, a2_guess]

# Fit the data
popt, pcov = curve_fit(lorentzian, x[ini:fin], y[ini:fin], p0 = pguess)

# Results
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7],  popt[8],  popt[9],  popt[10]
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]
AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2 = popt[0], popt[1], popt[2], popt[3], popt[4],  popt[5],  popt[6], popt[7], popt[8]

#fit = lorentzian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3)
#fit = lorentzian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0)
fit = lorentzian(x[ini:fin], AOHn, AOHh, ADMPO, G, I1, I2, y0, a1, a2)


perr = np.sqrt(np.diag(pcov))

print("Aducto 1:")
print("An = ", popt[0], " +- ", perr[0])
print("Ah = ", popt[1], " +- ", perr[1])
print("Intensidad = ", popt[4], " +- ", perr[4])
print("Ancho = ", popt[7], " +- ", perr[7])
print("----------------------------------------")
print("Aducto 2:")
print("An = ", popt[2], " +- ", perr[2])
print("Intensidad = ", popt[5], " +- ", perr[5])
print("Ancho = ", popt[8], " +- ", perr[8])
print("----------------------------------------")
print("G = ", popt[3],  " +- ", perr[3])
print("y0 = ", popt[6],  " +- ", perr[6])

fig, ax = plt.subplots(figsize=(9,6)) 

# Add labels to peaks
peak_labels = ['+', '*', '+', '*', '+', '*', '+']  # Specify the labels for each peak

x1 = (G-AOHh/2-AOHn) 
x2 = (G-AOHh/2)
x5 = (G+AOHh/2)
x6 = (G+AOHh/2+AOHn)
x13 = (G+ADMPO)
x14 = (G)
x15 = (G-ADMPO)

peak_positions = [x1,x13,x2,x14,x5,x15,x6]  # Calculate the peak positions based on parameters
peak_heights = [0.1, 0.05, 0.2, 0.05, 0.2, 0.05, 0.1]
for label, position, height in zip(peak_labels, peak_positions, peak_heights):
    ax.text(position, height, label, ha='center', va='bottom', fontsize=15, fontweight='bold')


ax.axhline(0, color='gray')

plt.text(3322.5, 0.225, r'A',fontsize=24, verticalalignment='top', fontweight='bold', color="black")

ax.plot(x[ini:fin], y[ini:fin], "o", label="Medicion", color = "black", markersize=2)
ax.plot(x[ini:fin], fit, label="Ajuste", color = "r", linewidth=2)

ax.set_xlabel(r"Campo Magnético (G)", fontsize=16)
ax.set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=16)
ax.tick_params(axis='both', labelsize=15, direction='in', length = 7)
ax.legend(fontsize=14)

x_min, x_max = 3320, 3380  # Specify the desired x-axis limits
y_min, y_max = -0.25, 0.25  # Specify the desired y-axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.savefig(r"../results/ajuste_un_radical_lorentzian_10min.pdf")
plt.savefig(r"../results/ajuste_un_radical_lorentzian_10min.png",dpi=600)
table = np.vstack((x[ini:fin], y[ini:fin],fit)) #sirve para guardar tablas de graficas 
np.savetxt(r"../ajuste_un_radical_lorentzian_10min.txt", table.T, delimiter=' ', newline='\n')

plt.show()