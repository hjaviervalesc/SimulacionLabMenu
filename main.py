import numpy as np

#constantes 
g = np.array([0, -10, 0]) # gravedad (m/s^2)
#condiciones iniciales
v0 = np.array([10, 0, 30]) # vel. inicial (m/s)
x0 = np.array([0, 100, 0]) #posición inicial (m)
total_time = 5


#Hacerlo con forward euler
dt = 1
t = 0.0
t0 = 0.0
# v_func = lambda t: 
v_prev = v0
x_prev = x0

def v_func(t):
    vtdt = v_prev + g * dt
    return vtdt
    

def x_func(t):
    xtdt = x_prev + v_prev * dt
    return xtdt

# x_func = lambda t: 


while t < total_time:
    v_lag = v_func(t)
    x_lag = x_func(t)
 
    print(f"t: {t:.2f} s,\t v: {v_lag} m/s\t x: {x_lag} m")
    
    v_prev = v_lag
    x_prev = x_lag

    t += dt
