import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1. / (1+np.exp(-x))

x = np.linspace(-9,9,1000) # Erzeuge X-Werte zwischen -9 und 9

plt.figure(figsize=(9,6))

# Beide Funktionen plotten:
# heaviside in orange
# sigmoid in rot
plt.plot(x,np.heaviside(x,1.0), label='heaviside', color='orange')
plt.plot(x,sigmoid(x), label='sigmoid', color='r')

# Grid und Legende anzeigen
plt.grid(True)
plt.legend()
plt.show()