import numpy as np

#constantes 
g = np.array([0, -10, 0]) # gravedad (m/s^2)
#condiciones iniciales
v0 = np.array([10, 0, 30]) # vel. inicial (m/s)
x0 = np.array([0, 100, 0]) #posición inicial (m)
m = 1 #Masa
d = 0.4 #Coeficiente de resistencia al aire
total_time = 5


#Hacerlo con forward euler
dt = 1
t = 0.0
t0 = 0.0
v_prev = v0
x_prev = x0
a_prev = g

def g_func():
    acc = g- ((d/m) * v_prev)
    return acc
    

def v_func():
    vtdt = v_prev + a_prev * dt
    return vtdt
    
def x_func():
    xtdt = x_prev + v_prev * dt
    return xtdt

print(f"t: {t:.2f} s,\t v: {v0} m/s\t x: {x0} m")

while t < total_time:
    t += dt
    
    a_prev = g_func()
    v_helper = v_func()
    x_helper = x_func()
 
    print(f"t: {t:.2f} s,\t v: {np.round(v_helper,1)} m/s\t x: {np.round(x_helper,1)} m")
    
    v_prev = v_helper
    x_prev = x_helper