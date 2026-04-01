import taichi as ti
import numpy as np

#IDEA DEL EJERCICIO
#El planteamiento es muy similar al apartado anterior
#Se pide añadir resistencia del aire, más fuerzas afectan a la bola
#Aceleración no será solo gravedad, calcular aceleración total y aplicar

#Inicializo Taichi
ti.init(arch=ti.cpu) 

#Discretizo la gravedad
g = ti.Vector([0.0, -10.0, 0.0])
a_helper = g

#Velocidad y posición en el momento inicial
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=())
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=())

v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
x_prev[None] = ti.Vector([0.0, 100.0, 0.0])

#Parámetros físicos
m = 1 
d = 0.4 

#Parámetros de la simulación
total_time = 5

#Delta Time reducido para poder ver la simulación
#Para ver resultados del enunciado, poner a 1
dt = 0.03
t = 0.0

##FUNCIONES
#Cálculo de velocidad mediante Euler
@ti.func
def v_func(v, a):
    #Ahora se usa aceleración total en vez de gravedad
    return v + a * dt

#Cálculo de posición mediante Euler
@ti.func
def x_func(x, v):
    return x + v * dt

#Cálculo de aceleración
@ti.func
def g_func(v):
    a = g- ((d/m) * v)
    return a

#Función UPDATE del proyecto (ejecutada cada frame)
@ti.kernel
def step():

    #Calculo parámetros (Velocidad, Posicion, Gravedad)
    a_helper = g_func(v_prev[None])

    #La velocidad depende de la aceleración total
    v_helper = v_func(v_prev[None], a_helper)
    x_helper = x_func(x_prev[None], v_prev[None])
    
    #Actualizo valores
    v_prev[None] = v_helper
    x_prev[None] = x_helper

#Reset del programa
def reset():
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])
    t = 0.0

##PROGRAMA PRINCIPAL
if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")

    #Bucle de simulación
    while t < total_time:
        #Calculo tiempo
        t += dt
        #Update
        step() 
        #Saco resultado 
        print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")

