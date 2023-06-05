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
AOHh = 15.0
AOHn = 14.6
ADMPO = 14.6
Ah = 25
An = 15.5

#Guess del centro del espectro
G1 = 3353
G2 = 3353
G3 = 3353
#Guess de la intensidad
I1 = 6
I2 = 0.95
I3 = 3.7

#Guess del 0 de la funcion
y0 = -9.158014352716684e-05

#Guess del ancho de la gaussiana
a1 = 2
a2 = 3
a3 = 6

ini = 575
fin = 1525

#definimos el conjunto de gaussianas. x son los datos, el resto son los parámetros a ajustar
#definimos los centros de cada gaussiana según los parámetros a ajustar
# los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
# el centro de la gaussiana i-ésima está dado por xi

def gaussian(x, AOHn, AOHh, ADMPO, An, Ah, G1, G2, G3, I1, I2, I3, y0, a1, a2, a3):
#def gaussian(x, AOH, ADMPO, AMgO, G1, G2, G3, I1, I2, I3, y0):
#def gaussian(x, I1, I2, y0, a1, a2):

    x1 = (G1-AOHh/2-AOHn)
    x2 = (G1-AOHh/2)
    x3 = (G1-AOHh/2+AOHn)
    x4 = (G1+AOHh/2-AOHn)
    x5 = (G1+AOHh/2)
    x6 = (G1+AOHh/2+AOHn)

    x7 = (G2-Ah/2-An)
    x8 = (G2-Ah/2)
    x9 = (G2-Ah/2+An)
    x10 = (G2+Ah/2-An)
    x11 = (G2+Ah/2)
    x12 = (G2+Ah/2+An)

    x13 = (G3+ADMPO)
    x14 = (G3)
    x15 = (G3-ADMPO)
    
    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)
    pref7 = -(x-x7)
    pref8 = -(x-x8)
    pref9 = -(x-x9)
    pref10 = -(x-x10)
    pref11 = -(x-x11)
    pref12 = -(x-x12)
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
        
    #Interaccion x 

    y7 = I2*a2*pref7*(4*(x-x7)**2 + a2**2)**(-2)
    y8 = I2*a2*pref8*(4*(x-x8)**2 + a2**2)**(-2)
    y9 = I2*a2*pref9*(4*(x-x9)**2 + a2**2)**(-2)
    y10 = I2*a2*pref10*(4*(x-x10)**2 + a2**2)**(-2)
    y11 = I2*a2*pref11*(4*(x-x11)**2 + a2**2)**(-2)
    y12 = I2*a2*pref12*(4*(x-x12)**2 + a2**2)**(-2)

    # Interacción DMPO
    y13 = I3*a3*pref13*(4*(x-x13)**2 + a3**2)**(-2)
    y14 = I3*a3*pref14*(4*(x-x14)**2 + a3**2)**(-2)
    y15 = I3*a3*pref15*(4*(x-x15)**2 + a3**2)**(-2)

    y = y1+y2+y3+y4+y5+y6+y7+y8+y9+y10+y11+y12+y13+y14+y15+y0

    return y

# Obtain xdata and ydata
data = np.loadtxt("AM02_pH5_RT_40min_baselineA.dat", delimiter="	") #Tenes el campo en la columna 1 y la señal en la columna 2 (la 3 ignorala)
                                                           #Cambiar , a . para que ande.

x = data[:,0]
y = data[:,1]


# Initial guess of the parameters 
I1_guess = I1
I2_guess = I2
I3_guess = I3
y0_guess = y0
G1_guess = G1
G2_guess = G2
G3_guess = G3
AOHh_guess = AOHh
AOHn_guess = AOHn
ADMPO_guess = ADMPO
An_guess = An
Ah_guess = Ah

#A4_guess = A4
a1_guess = a1
a2_guess = a2
a3_guess = a3


pguess = [AOHn_guess, AOHh_guess, ADMPO_guess, An_guess, Ah_guess, G1_guess, G2_guess, G3_guess, I1_guess, I2_guess, I3_guess,y0_guess, a1_guess, a2_guess, a3_guess]
#pguess = [AOH_guess, ADMPO_guess, AMgO_guess, G_guess, I1_guess, I2_guess, I3_guess, y0_guess]
#pguess = [I1_guess, I2_guess, y0_guess, a1_guess, a2_guess]

# Fit the data
popt, pcov = curve_fit(gaussian, x[ini:fin], y[ini:fin], p0 = pguess)

# Results
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7],  popt[8],  popt[9],  popt[10]
#AOH, ADMPO, AMgO, G, I1, I2, I3, y0 = popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]
AOHn, AOHh, ADMPO, An, Ah, G1, G2, G3, I1, I2, I3, y0, a1, a2, a3  = popt[0], popt[1], popt[2], popt[3], popt[4],  popt[5],  popt[6], popt[7], popt[8], popt[9], popt[10], popt[11], popt[12], popt[13], popt[14]

#fit = gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0, a1, a2, a3)
#fit = gaussian(x, AOH, ADMPO, AMgO, G, I1, I2, I3, y0)
fit = gaussian(x[ini:fin], AOHn, AOHh, ADMPO, An, Ah, G1, G2, G3, I1, I2, I3, y0, a1, a2, a3)

perr = np.sqrt(np.diag(pcov))

print("Aducto 1:")
print("An = ", popt[0], " +- ", perr[0])
print("Ah = ", popt[1], " +- ", perr[1])
print("Intensidad = ", popt[8], " +- ", perr[8])
print("Ancho = ", popt[12], " +- ", perr[12])
print("G1 = ", popt[5],  " +- ", perr[5])
print("----------------------------------------")
print("Aducto 2:")
print("An = ", popt[3], " +- ", perr[3])
print("Ah = ", popt[4], " +- ", perr[4])
print("Ancho = ", popt[13], " +- ", perr[13])
print("Intensidad = ", popt[9], " +- ", perr[9])
print("G2 = ", popt[6],  " +- ", perr[6])
print("----------------------------------------")
print("Aducto 3:")
print("An = ", popt[2], " +- ", perr[2])
print("Ancho = ", popt[14], " +- ", perr[14])
print("Intensidad = ", popt[10], " +- ", perr[10])
print("G3 = ", popt[7],  " +- ", perr[7])
print("----------------------------------------")
print("y0 = ", popt[11],  " +- ", perr[11])

fig, ax = plt.subplots(figsize=(9,6)) 

# Add labels to peaks
peak_labels = ['^','+', '*','^', '+','^', '*', '^','+','^', '*','+','^',] 

x1 = (G1-AOHh/2-AOHn)
x2 = (G1-AOHh/2)
x5 = (G1+AOHh/2)
x6 = (G1+AOHh/2+AOHn)

x7 = (G2-Ah/2-An)
x8 = (G2-Ah/2)
x9 = (G2-Ah/2+An)
x10 = (G2+Ah/2-An)
x11 = (G2+Ah/2)
x12 = (G2+Ah/2+An)

x13 = (G3+ADMPO)
x14 = (G3)
x15 = (G3-ADMPO)

peak_positions = [x7,x1,x13,x8,x2,x9,x14,x10,x5,x11,x15,x6,x12]  # Calculate the peak positions based on parameters
peak_heights = [ 0.05, 0.25, 0.03, 0.05,  0.45, 0.05,  0.03, 0.05,  0.45, 0.05,  0.03, 0.25, 0.05, ]
for label, position, height in zip(peak_labels, peak_positions, peak_heights):
    ax.text(position, height, label, ha='center', va='bottom', fontsize=15, fontweight='bold')


ax.axhline(0, color='gray')


x_min, x_max = 3320, 3385  # Specify the desired x-axis limits
y_min, y_max = -0.5, 0.5  # Specify the desired y-axis limits
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.text(3322.5, 0.45, r'B',fontsize=24, verticalalignment='top', fontweight='bold', color="black")

ax.plot(x[ini:fin], y[ini:fin], "o", label="Medicion", color = "black", markersize=2)
ax.plot(x[ini:fin], fit, label="Ajuste", color = "r", linewidth=2)

ax.set_xlabel(r"Campo Magnético (G)", fontsize=16)
ax.set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=16)
ax.tick_params(axis='both', labelsize=15, direction='in', length = 7)
ax.legend(fontsize=14)


plt.savefig(r"../results/ajuste_dos_radicales_lorentzian_40min.pdf")
plt.savefig(r"../results/ajuste_dos_radicales_lorentzian_40min.png",dpi=300)
table = np.vstack((x[ini:fin], y[ini:fin],fit)) #sirve para guardar tablas de graficas 
np.savetxt(r"../ajuste_dos_radicales_40min.txt", table.T, delimiter=' ', newline='\n')

plt.show()




#title = ax.set_title("$T_\mathrm{{NOGSE}}$ = {} ms  ||  $N$ = {} || $G$ = {} mT/m".format(T_nogse[0], int(n[0]), g[0]), fontsize=18)
#hpp = 2*a*I*np.exp(-0.5) #altura
#Hpp = 2*a #ancho

#print("hpp = ", hpp, "Hpp = ", Hpp)

