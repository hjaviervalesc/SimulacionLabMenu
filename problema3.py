import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)

#Problema parecido al problema 2, pero ahora se añade una fuerza más, la fuerza del viento.

#constantes 
g = ti.Vector([0.0, -10.0, 0.0])
v_viento = ti.Vector([-12.5, 0.0, 0.0])
m=1
d=0.4

#tiempos
total_time = 5.0
dt = 0.03
t = 0.0

#campos, donde guardamos la velocidad y posición del frame anterior
v_prev = ti.Vector.field(3, dtype=ti.f32, shape=())
x_prev = ti.Vector.field(3, dtype=ti.f32, shape=())

#condiciones iniciales
v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
x_prev[None] = ti.Vector([0.0, 100.0, 0.0])


@ti.func
def a_func(v, v_viento):
    #Por la segunda ley de Newton. Actuan, tres fuerzas, la gravedad, la fuerza de resistencia del aire 
    #y la fuerza del viento. La aceleración es la suma de las fuerzas dividido por la masa, a = F/m.
    return g - d/m * (v - v_viento)

@ti.func
def v_func(v, a):
    return v + a * dt

@ti.func
def x_func(x, v):
    return x + v * dt

@ti.kernel
def step():
    #Utilizamos fordward Euler, es decir, calculo los parámetros a través de los parámetros del frame anterior.
    #La aceleración depende de la velocidad y del viento
    a_helper = a_func(v_prev[None], v_viento)
    #La velocidad depende de la aceleración total
    v_helper = v_func(v_prev[None], a_helper)
    #La posición depende de la velocidad anterior.
    x_helper = x_func(x_prev[None], v_prev[None])

    #Actualizamos valores
    v_prev[None] = v_helper
    x_prev[None] = x_helper


def reset():
    v_prev[None] = ti.Vector([10.0, 0.0, 30.0])
    x_prev[None] = ti.Vector([0.0, 100.0, 0.0])

if __name__ == "__main__":
    print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")

    #Por cada paso de tiempo, actualizamos y mostramos resultados.
    while t < total_time:
        t += dt
        step()
        print(f"t: {t:.2f} s,\t v: {np.round(v_prev[None],1)} m/s\t x: {np.round(x_prev[None],1)} m")
    

