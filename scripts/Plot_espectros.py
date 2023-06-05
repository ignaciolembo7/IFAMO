import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Obtain xdata and ydata
data1 = np.loadtxt("ajuste_un_radical_gaussian.txt") # col 1 = x, col 2 = y, col 3 = fit
data2 = np.loadtxt("ajuste_dos_radicales.txt") # col 1 = x, col 2 = y, col 3 = fit
data3 = np.loadtxt("ajuste_un_radical_lorentzian.txt") # col 1 = x, col 2 = y, col 3 = fit
data4 = np.loadtxt("ajuste_un_radical_voigt.txt") # col 1 = x, col 2 = y, col 3 = fit

x = data1[:,0]
y = data1[:,1]
fit1 = data1[:,2]
fit2 = data2[:,2]
fit3 = data3[:,2]
fit4 = data4[:,2]

fig, ax = plt.subplots(figsize=(9,6)) 

ax.plot(x, y, "o", label="Medicion", color = "black", markersize=1)
ax.plot(x, fit1, label="Ajuste 1 radical gaussiano", color = "r", linewidth=1.25)
ax.plot(x, fit3, label="Ajuste 1 radical lorentizano", color = "orange", linewidth=1.25)
ax.plot(x, fit4, label="Ajuste 1 radical voigt", color = "green", linewidth=1.25)
ax.plot(x, fit2, label="Ajuste 2 radicales gaussiano", color = "b", linewidth=1.25)


ax.set_xlabel(r"Campo Magnético (G)", fontsize=16)
ax.set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=16)
ax.tick_params(axis='both', labelsize=15, direction='in', length = 7)
ax.legend(fontsize=10)

plt.savefig(r"../comparacion.pdf")
plt.savefig(r"../comparacion.png",dpi=600)
#table = np.vstack((x[ini:fin], y[ini:fin],fit)) #sirve para guardar tablas de graficas 
#np.savetxt(r"../ajuste_un_radical.txt", table.T, delimiter=' ', newline='\n')

plt.show()