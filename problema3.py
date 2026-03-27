import numpy as np

#constantes 
g = np.array([0, -10, 0]) # gravedad (m/s^2)   ///Fg
#condiciones iniciales
v0 = np.array([10, 0, 30]) # vel. inicial (m/s)
x0 = np.array([0, 100, 0]) #posición inicial (m)
v0viento = np.array([-12.5, 0, 0])
total_time = 5
m=1

#Hacerlo con forward euler
dt = 1
t = 0.0
t0 = 0.0
v_prev = v0
x_prev = x0
vviento_prev = v0viento
a_prev=g-0.4/m*(v_prev-vviento_prev)

def a_func():
    a=g-0.4/m*(v_prev-vviento_prev)
    return a

def v_func(a_prev):
    vtdt = v_prev + a_prev * dt
    return vtdt
    
def x_func():
    xtdt = x_prev + v_prev * dt
    return xtdt

print(f"t: {t:.2f} s,\t v: {v0} m/s\t x: {x0} m")

while t < total_time:
    t += dt
    
    a_prev = a_func()
    v_helper = v_func(a_prev)
    x_helper = x_func()
 
    print(f"t: {t:.2f} s,\t v: {np.round(v_helper,1)} m/s\t x: {np.round(x_helper,1)} m")
    v_prev = v_helper
    x_prev = x_helper
    

