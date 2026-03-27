import numpy as np

#constantes 
g = np.array([0, -10, 0]) # gravedad (m/s^2)
#condiciones iniciales
v0 = np.array([10, 0, 30]) # vel. inicial (m/s)
x0 = np.array([0, 100, 0]) #posición inicial (m)
total_time = 5


#Hacerlo con forward euler
dt = 0.01
t = 0.0
t0 = 0.0
v_func = lambda t: v0 + g * t
x_func = lambda t:x0 + v0 * (t - t0) + 0.5 * g * (t - t0)**2


dt = 0.01

while t < total_time:
    if abs(t - round(t)) < 1e-6: 
        print(f"t: {t:.2f} s,\t v: {v_func(t)} m/s\t x: {x_func(t)} m")
    
    t += dt
