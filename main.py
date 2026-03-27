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
v_prev = v0
x_prev = x0

def v_func():
    vtdt = v_prev + g * dt
    return vtdt
    
def x_func():
    xtdt = x_prev + v_prev * dt
    return xtdt

print(f"t: {t:.2f} s,\t v: {v0} m/s\t x: {x0} m")

while t < total_time:
    t += dt
    

    v_helper = v_func()
    x_helper = x_func()
 
    print(f"t: {t:.2f} s,\t v: {v_helper} m/s\t x: {x_helper} m")
    
    v_prev = v_helper
    x_prev = x_helper

