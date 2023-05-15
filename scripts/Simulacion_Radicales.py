# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 15:11:41 2022

@author: ignacio
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import stats
import matplotlib.patches as patches

#constantes
Ah = 3.3751375 #constante hiperfina H2, están en G
An = 16.179725 #constante hiperfina N2
G =(9.454E+9)*(6.623E-34)/(( 2.0029125)*(9.27E-28))
y0 = 0.07578734272609323
a = 0.509190606012936 #ancho de cada gaussiana


t = np.array([ 10,30,60,90])
I = np.array([ 0.150463 , 0.323436, 0.694551, 0.549866])
#definimos los centros según los parámetros a ajustar

#definimos el conjunto de gaussianas

def boltzmann(x,A1,A2,x0,dx):
    #x son los datos, el resto son los parámetros a ajustar

    y = A2 + (A1-A2)/(1+np.exp((x0-x)/dx))
    return y

def gaussian(x,I,y0):
    #x son los datos, el resto son los parámetros a ajustar
    #definimos los centros de cada gaussiana según los parámetros a ajustar

    # los anchos de todas las gaussianas se consideran iguales a "a" (fijo)
    # el centro de la gaussiana i-ésima está dado por xi

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

def recta(x,m,h):
    y = m*x + h
    return y

data0 = np.loadtxt("Corona_t=90min.dat")
data1 = np.loadtxt("Corona_t=90min_2especies.dat")
data2 = np.loadtxt("Corona_t=90min_2especies_rad1.dat")
data3 = np.loadtxt("Corona_t=90min_2especies_rad2.dat")

# Obtain xdata and ydata
x0 = data0[:,0]
y0 = data0[:,1]
x1 = data1[:,0]
y1 = data1[:,1]
x2 = data2[:,0]
y2 = data2[:,1]
x3 = data3[:,0]
y3 = data3[:,1]


plt.subplot(2, 1, 1)
plt.plot(x0,y0,'k')
plt.plot(x1,y1,'r')
plt.xlim(3340,3400)
plt.ylim(-0.04,  0.21)
plt.text(3342, 0.195, r'A',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
plt.legend([r"Medición", r"Simulación" ], loc='lower left', fontsize=12)
plt.tick_params(labelbottom = False)


plt.subplot(2, 1, 2)
plt.plot(x2,y2,'r')
plt.plot(x3,y3,'b--')
plt.xlim(3340,3400)
plt.ylim(0.02,  0.16)
plt.text(3331, 0.18, 'Señal EPR [u.a.]', va='center', rotation='vertical', fontsize=16)
plt.text(3342, 0.15, r'B',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
plt.legend([r"Radical 1", r"Radical 2"], loc='lower left', fontsize=12)
plt.xlabel(r"$B_0$ [G]", fontsize=16)



plt.subplot_tool()
plt.show()

























r"""
pguess = [ 1, 1,  150,  3.11041987e+01]
# Fit the data 
popt, pcov = curve_fit(boltzmann, t, I_arr, p0 = pguess)

# Results
A1,A2,x0,dx  = popt[0], popt[1], popt[2], popt[3]

#############################################################################3

fig, ax = plt.subplots()
props = dict(facecolor='lightgray', alpha=0)

plt.errorbar(t, I_arr, yerr = I_err,  ecolor='k', capsize=4, elinewidth=1.5, fmt=" ")

t2 = np.linspace(0,180,1000)
fit2 = boltzmann(t2, A1,A2,x0,dx)
plt.plot(t2,fit2, 'r')

p1 = patches.FancyArrowPatch((0, 0.035), (80, 0.035), arrowstyle='<|-|>', mutation_scale=20)
ax.add_patch(p1)
ax.text(0.05, 0.95, r'B' , transform=ax.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax.text(0.035, 0.30, r'Lag time = $(80 \pm 10)$ min ' , transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, rotation = 0)

plt.ylabel(r"Señal EPR [u.a.]", fontsize=14)
plt.xlabel(r"$t$ [min]", fontsize=14)

plt.savefig(r"C:\Users\ignacio\Documents\Instituto Balseiro\Tercer_Cuatrimestre\Experimental_III\EPR_Lembo\Figuras\Corona_T_boltz.pdf")

######################################################################

fig, ax = plt.subplots()

plt.errorbar(t, I_arr, yerr = I_err,  ecolor='k', capsize=4, elinewidth=1.5, fmt=" ")

t3 = np.linspace(0,90,1000)
fit3 = recta(t3, 0.00026, 0.00568)
plt.plot(t3,fit3, 'r')

t4 = np.linspace(70,180,1000)
fit4 = recta(t4, 0.00121, -0.07267 )
plt.plot(t4,fit4, 'r')

p1 = patches.FancyArrowPatch((0, 0.030), (83.5, 0.030), arrowstyle='<|-|>', mutation_scale=20)
ax.add_patch(p1)
ax.text(0.05, 0.95, r'A' , transform=ax.transAxes, fontsize=24, verticalalignment='top', fontweight='bold', color="black")
ax.text(0.035, 0.30, r'Lag time = $(83 \pm 2)$ min ' , transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props, rotation = 0)

plt.ylabel(r"Señal EPR [u.a.]", fontsize=14)
plt.xlabel(r"$t$ [min]", fontsize=14)

plt.savefig(r"C:\Users\ignacio\Documents\Instituto Balseiro\Tercer_Cuatrimestre\Experimental_III\EPR_Lembo\Figuras\Corona_T_rectas.pdf")
###############################################################################################

"""