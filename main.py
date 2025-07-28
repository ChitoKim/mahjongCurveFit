import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
han = np.arange(1, 14, 1)
fu = np.arange(20, 150, 10)
H, F = np.meshgrid(han, fu)

def point_func(han, fu):
  return 16 * fu * (2 ** han) * ((np.heaviside(han, 1) - np.heaviside(han - 3, 1))) \
  + ((16 * fu * (2 ** han)) * (np.heaviside(fu, 1) - np.heaviside(fu - 70, 1)) + 8000 * np.heaviside(fu - 70, 1)) * (np. heaviside(han - 3 , 1) - np. heaviside(han - 4, 1)) \
  + ((16 * fu * (2 ** han)) * (np.heaviside(fu, 1) - np.heaviside(fu - 40 , 1)) + 8000 * np.heaviside(fu - 40, 1)) * (np.heaviside(han - 4, 1) - np.heaviside(han - 6, 1)) \
  + 12000 * (np.heaviside(han - 6, 1) - np.heaviside(han - 8, 1)) \
  + 16000 * (np.heaviside(han - 8, 1) - np.heaviside(han - 11, 1)) \
  + 24000 * (np.heaviside(han - 11, 1) - np.heaviside(han - 13, 1)) \
  + 32000 * (np.heaviside(han - 13, 1))                                                                                                                  

def linear_model(X, a, b, c):
  (h, f) = X
  return a * h + b * f + c
np.set_printoptions(suppress=True)
T = 100 * np.ceil(0.01* point_func(H, F))
#T_oya = 100 * np.ceil(.015 * point_func(H, F))
#print(T)
xdata = np.vstack((H.ravel(), F.ravel()))
tdata = T.ravel()
#print(xdata)
popt, pcov = curve_fit(linear_model, xdata, tdata, maxfev = 10000)
print(popt, pcov)
TFIT = linear_model((H, F), *popt)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(H, F, T, alpha = 0.5, cmap='viridis')
ax.plot_surface(H, F, TFIT.reshape(H.shape), alpha=0.5, cmap='plasma')
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(H, F, abs(100 * (1 - TFIT / T)), alpha=0.5, cmap='viridis')
plt.show()


