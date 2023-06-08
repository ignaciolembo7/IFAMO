import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.patches as patches

def lorentzian(x, AOHn, AOHh, G, I1, y0, a1):

    x1 = (G-AOHh/2-AOHn) 
    x2 = (G-AOHh/2)
    x3 = (G-AOHh/2+AOHn)
    x4 = (G+AOHh/2-AOHn)
    x5 = (G+AOHh/2)
    x6 = (G+AOHh/2+AOHn)

    #x13 = (G+ADMPO)
    #x14 = (G)
    #x15 = (G-ADMPO)
    
    pref1 = -(x-x1)
    pref2 = -(x-x2)
    pref3 = -(x-x3)
    pref4 = -(x-x4)
    pref5 = -(x-x5)
    pref6 = -(x-x6)
    #pref13 = -(x-x13)
    #pref14 = -(x-x14)
    #pref15 = -(x-x15)

    # Radical OH
    y1 = I1*a1*pref1*(4*(x-x1)**2 + a1**2)**(-2)
    y2 = I1*a1*pref2*(4*(x-x2)**2 + a1**2)**(-2)
    y3 = I1*a1*pref3*(4*(x-x3)**2 + a1**2)**(-2)
    y4 = I1*a1*pref4*(4*(x-x4)**2 + a1**2)**(-2)
    y5 = I1*a1*pref5*(4*(x-x5)**2 + a1**2)**(-2)
    y6 = I1*a1*pref6*(4*(x-x6)**2 + a1**2)**(-2)
        

    # Interacción DMPO
    #y13 = I2*a2*pref13*(4*(x-x13)**2 + a2**2)**(-2)
    #y14 = I2*a2*pref14*(4*(x-x14)**2 + a2**2)**(-2)
    #y15 = I2*a2*pref15*(4*(x-x15)**2 + a2**2)**(-2)

    y = y1+y2+y3+y4+y5+y6+y0

    return y

# Crear la figura y los subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6), gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 1]})

# Cargar y mostrar la primera imagen PNG en la subfigura 1
img1 = mpimg.imread('DMPO-OH.png')
axes[0, 1].imshow(img1)
axes[0, 1].axis('off')  # Opcional: ocultar los ejes
axes[0, 1].set_title('DMPO-OH')

# Cargar y mostrar la segunda imagen PNG en la subfigura 2
img2 = mpimg.imread('DMPO-CH3.png')
axes[1, 1].imshow(img2)
axes[1, 1].axis('off')  # Opcional: ocultar los ejes
axes[1, 1].set_title(r'DMPO-CH$_3$')



# Generar algunos datos de ejemplo para las otras subfiguras
x = np.linspace(3320, 3390, 1000)
yOH = lorentzian(x, 15, 14.6, 3352, 1, 0, 2)
yCH3 = lorentzian(x, 15.3, 22.3, 3352, 1, 0, 2)

# Plot 3
axes[0, 0].plot(x, yOH, color = "r")
#axes[0, 0].set_title('DMPO-OH')

# Plot 4
axes[1, 0].plot(x, yCH3, color = "r")
#axes[1, 0].set_title('DMPO-CH3')

# Ajustar la disposición de los subplots
plt.tight_layout()
plt.text(0, 20, r'D',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
plt.text(0, -810, r'C',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
plt.text(-1310, -810, r'A',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
plt.text(-1310, 20, r'B',fontsize=18, verticalalignment='top', fontweight='bold', color="black")
axes[0, 0].tick_params(axis='both', labelsize=10, direction='in', length = 3)
axes[1, 0].tick_params(axis='both', labelsize=10, direction='in', length = 3)
axes[1, 0].set_xlabel(r"Campo Magnético (G)", fontsize=9)
axes[1, 0].set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=9)
axes[0, 0].set_xlabel(r"Campo Magnético (G)", fontsize=9)
axes[0, 0].set_ylabel(r"$\mathrm{dI_{EPR}/dH}  $", fontsize=9)
# Mostrar la figura en pantalla
plt.savefig(r"../results/simulacion.pdf")
plt.savefig(r"../results/simulacion.png",dpi=600)
plt.show()