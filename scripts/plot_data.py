import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


data = np.loadtxt("AM02_pH5_RT_10min.dat", delimiter="	") #Tenes el campo en la columna 1 y la señal en la columna 2 (la 3 ignorala)
                                                           #Cambiar , a . para que ande.

# Obtain xdata and ydata

x = data[:,0]
y = data[:,1]

plt.plot(x,y)

plt.ylabel(r"Señal EPR [u.a.]", fontsize=14)
plt.xlabel(r"Campo magnético [G]", fontsize=14)

plt.show()

plt.savefig(r"../data_plot.pdf")
plt.savefig(r"../data_plot.png")

